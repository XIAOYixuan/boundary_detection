import numpy as np
import torch

from config import Config
from model import TriDet
from data import ContLabels
from features import W2V2Front
from core.metrics import boundary_mae 
from core.range_eer import range_eer
from postprocess import utt_proposals

@torch.no_grad()
def infer_utt(model: TriDet, 
              feat: torch.Tensor,
              end_bound: float, 
              cfg: Config,
              device: torch.device):
    T = feat.shape[1]
    mult = 1 << (cfg.levels - 1)
    # pad to this
    Tm = -(-T//mult) * mult
    x = torch.nn.functional.pad(feat, (0, Tm-T))[None].to(device)
    mask = torch.zeros(1, Tm, dtype=torch.bool, device=device)
    mask[:, :T] = True
    out = model(x, mask)
    cls_list = [c[0].cpu().numpy() for c in out['cls']]
    seg_list = [s[0].cpu().numpy() for s in out['seg']]
    msk_list = [m[0].cpu().numpy() for m in out['masks']]
    props = utt_proposals(cls_list, seg_list, msk_list,
                          end_bound=end_bound,
                          cfg=cfg)
    return props, float(out['gate'][0])


def main():
    cli = Config.from_args()
    device = torch.device('cuda')
    ckpt_path = cli.ckpt or str(cli.run_dir / 'ckpt_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # resetting cfg
    cfg = Config(**dict(ckpt['cfg'])) 
    cfg.split, cfg.ckpt, cfg.name = cli.split, cli.ckpt, cli.name
    cfg.gate_thr = cli.gate_thr
    for k in ('data_dir', 'seglab_dir', 'w2v_ckpt', 'index_path'):
        if getattr(cli, k):
            setattr(cfg, k, getattr(cli, k))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    model = TriDet(cfg).to(device).eval()
    model.load_state_dict(ckpt['model'])

    ds = ContLabels(cfg, cfg.split)
    w2v = W2V2Front.from_ckpt(cfg, ckpt).to(device)
    offset = cfg.grid_offset

    ids, props, gts = [], [], []
    gates = np.empty(len(ds.ids))

    for u in range(len(ds.ids)):
        f = w2v.single(ds.paths[u])
        n = ds.n_lab[u]
        # get feature frame count
        T = n*160//cfg.hop+1 if n else f.shape[1]
        f = torch.nn.functional.pad(f[:, :T], (0, max(0, T-f.shape[1])))
        bound = cfg.end_bound(ds.n_samples[u]) if n else T - offset
        prop, gates[u] = infer_utt(model, f, bound, cfg, device)
        props.append(prop)
        ids.append(ds.ids[u])
        gts.append(ds.segs[u])

    # for binary utt deepfake detection
    # if gated, we skip computing range eer for the utt
    gated = []
    fakes = np.flatnonzero(ds.spoof)
    for p, g in zip(props, gates):
        if g >= cfg.gate_thr:
            gated.append(p)
        else:
            gated.append(np.zeros((0, 3), np.float32))
    fake_props = [gated[u] for u in fakes]
    fake_gts = [gts[u] for u in fakes]
    ms, me, matched = boundary_mae(fake_props, fake_gts)
    print(f'ms {ms} me {me} matched {matched}')

    for resolution, (eer, thresh, fpr, fnr) in range_eer(gated, gts, ds.n_lab).items():
        print(f"RANGE_EER@{resolution}ms {eer: .4f} thresh {thresh:.4f} "
              f"fpr {fpr: .4f} fnr {fnr:.4f}")
        
    def obj(x: list) -> np.ndarray:
        a = np.empty(len(x), object)
        a[:] = x
        return a

    np.savez(cfg.run_dir / f"proposals_{cfg.split}.npz",
             ids=np.array(ids), 
             props=obj(props),
             gts=obj(gts),
             gates=gates)

if __name__ == '__main__':
    main()