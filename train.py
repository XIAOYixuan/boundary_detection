import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from config import Config
from data import BucketBatchSampler, UttDataset, collate
from features import W2V2Front, ssl_state
from model import TriDet, TriDetLoss


def dev_loss(front, model, loss_fn, loader, device, mult) -> tuple:
    model.eval()
    with torch.no_grad():
        tot = utt = bnd = n = 0
        for wav, n_sample, n_label, gt_segs, y_utt in loader:
            f, mask = front(wav.to(device),
                            n_sample.to(device), 
                            n_label.to(device),
                            mult)
            model_out = model(f, mask)
            seg_list = [s.to(device) for s in gt_segs]
            y_utt = y_utt.to(device)
            bound = loss_fn.cfg.end_bound(n_sample).float().to(device)
            l = loss_fn(model_out, seg_list, 
                        y_utt, bound)
            tot += l["total"].item() * len(wav)
            utt += l["utt"].item() * len(wav)
            bnd += l["bnd"].item() * len(wav)
            n += len(wav)
    model.train()
    cfg = loss_fn.cfg
    return (tot - cfg.lambda_utt * utt - cfg.lambda_bnd * bnd) / n, utt / n, bnd / n


def main():
    cfg = Config.from_args()
    assert cfg.data_dir and cfg.seglab_dir and cfg.w2v_ckpt

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda')
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.run_dir/"log.txt", "a") as log:
        def say(msg: str) -> None:
            print(msg, flush=True)
            print(msg, file=log, flush=True)

        mult = 1 << (cfg.levels - 1)
        train = UttDataset(cfg, "train")
        dev = UttDataset(cfg, "dev")

        g = torch.Generator().manual_seed(cfg.seed)
        dev_sub = Subset(dev, 
                         torch.randperm(len(dev), generator=g)[:cfg.dev_utts].tolist())

        if cfg.bucket:
            bucket_sampler = BucketBatchSampler(train.ds.n_samples, 
                                                cfg.batch, 
                                                cfg.bucket_chunk, cfg.seed)
            train_loader = DataLoader(train, batch_sampler=bucket_sampler,
                                        num_workers=cfg.num_workers,
                                        collate_fn=collate)
        else:
            train_loader = DataLoader(train, cfg.batch,
                                        shuffle=True, 
                                        num_workers=cfg.num_workers, drop_last=True,
                                        collate_fn=collate)

        dev_loader = DataLoader(dev_sub,
                                cfg.batch,
                                num_workers=cfg.num_workers,
                                collate_fn=collate
                                )
        say(f"train utts: {len(train)}, dev utt {len(dev_sub)}")
        say(f"batches/epoch {len(train_loader)}")

        model = TriDet(cfg).to(device)

        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed)
            front = W2V2Front(cfg).to(device)

        loss_fn = TriDetLoss(cfg)
        utt_head = list(model.gate.parameters())
        head_set = {id(p) for p in utt_head}
        tridet_trunk = [p for p in model.parameters() if id(p) not in head_set]
        ssl_param = [p for p in front.parameters() if p.requires_grad]
        groups = [
            {
                "params": list(model.parameters()), "lr": cfg.lr,
                "weight_decay": cfg.weight_decay
            },
            {
                "params": ssl_param, 
                "lr": cfg.ssl_lr,
                "weight_decay": cfg.ssl_wd
            }
        ]
        optimizer = torch.optim.AdamW(groups)
        scaler = torch.amp.GradScaler("cuda")

        total, warm = cfg.epochs * len(train_loader), cfg.warmup_epochs * len(train_loader)
        # mimic the original tridet settings
        schedular = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s + 1) / max(warm, 1) if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
        )

        best = float('inf')
        no_improve = 0
        step = 0
        for ep in range(cfg.epochs):
            if cfg.bucket:
                bucket_sampler.set_epoch(ep)

            for wave, n_sample, n_label, gt_segs, y_utt in train_loader:
                step += 1
                feat, mask = front(wave.to(device), 
                                    n_sample.to(device),
                                    n_label.to(device),
                                    mult)
                model_out = model(feat, mask)
                loss = loss_fn(
                    model_out,
                    [s.to(device) for s in gt_segs],
                    y_utt.to(device),
                    cfg.end_bound(n_sample).float().to(device)
                )

                optimizer.zero_grad()
                scaler.scale(loss["total"]).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(tridet_trunk + ssl_param, 1.0)
                torch.nn.utils.clip_grad_norm_(utt_head, 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedular.step()

                if step % cfg.log_every == 0:
                    say(f"{step}/{ep}"
                        f"total loss: {loss['total']:.4f}"
                        f"reg {loss['reg']:.4f}"
                        f"cls pos and neg: {loss['cls_pos']} / {loss['cls_neg']}"
                        f"bnd {loss['bnd']}"
                        f"utt {loss['utt']}"
                        )

            dev_loc_loss, dev_utt, dev_bnd = dev_loss(front, model, 
                                                            loss_fn, dev_loader, device, mult)
            state = {'model': model.state_dict(),
                            'ssl': ssl_state(front),
                            'cfg': vars(cfg)}
            torch.save(state, cfg.run_dir / 'ckpt_last.pt')

            if dev_loc_loss < best:
                best = dev_loc_loss
                no_improve = 0
                torch.save(state, cfg.run_dir / 'ckpt_best.pt')
            else:
                no_improve += 1

            say(f"epoch {ep} done: dev loc loss {dev_loc_loss} dev utt {dev_utt}, dev bnd {dev_bnd}")

            if cfg.patience and no_improve >= cfg.patience:
                say(f"early stop triggered at epoch {ep}")
                break


if __name__ == '__main__':
    main()

                    
                    