import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

NEG = -1e4

class MaskedGroupNorm(nn.Module):

    def __init__(self, groups: int, C: int):
        super().__init__()
        self.g = groups
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        # B, g=16, c=16, T
        xg = x.view(B, self.g, C // self.g, T)
        m = mask.view(B, 1, 1, T).float()
        # B, 1, 1, 1
        n = (m.sum(3, keepdim=True) * xg.shape[2]).clamp_min(1)
        # B, g, 1, 1
        mu = (xg*m).sum((2, 3), keepdim=True) / n
        var = ((xg-mu).pow(2)*m).sum((2, 3), keepdim=True) / n
        xhat = ((xg-mu)/(var+1e-5).sqrt()).view(B, C, T)
        out = (xhat*self.weight[:, None]+self.bias[:, None]) * mask[:, None]
        return out




class SGPBlock(nn.Module):

    def __init__(self, C:int,
                 w: int, 
                 k: float):
        super().__init__()
        kw = math.ceil(k*w)
        # make sure we have an odd kernel
        # so the position won't change
        # it makes the decoding easier
        kw = kw if kw % 2 else kw + 1
        self.ln = nn.LayerNorm(C)
        self.fc = nn.Conv1d(C, C, 1)
        self.gfc = nn.Conv1d(C, C, 1)
        self.psi = nn.Conv1d(C, C, w, padding= w//2, groups=C)
        self.convw = nn.Conv1d(C, C, w, padding= w//2, groups=C)
        self.convkw = nn.Conv1d(C, C, kw, padding=kw//2, groups=C)
        self.gn = MaskedGroupNorm(16, C) 
        self.ffn = nn.Sequential(
            nn.Conv1d(C, 4*C, 1),
            nn.ReLU(),
            nn.Conv1d(4*C, C, 1)
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # 1. phi(global)*FC + psi*(Conv_w+conv_kw) -> residual connection
        # 2. then groupnorm + FFN -> residual connection
        
        # x: B, 256, T
        # B, 1, T
        m = mask[:, None].float()

        # global: B, T, 256 -> B, 256, T
        z = self.ln(x.transpose(1, 2)).transpose(1, 2) * m
        # B, 256, 1 -> len
        g = z.sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1)
        g = self.gfc(g)
        # phi*(global)*FC
        partA = torch.relu(g) * self.fc(z)
        # psi*(Conv_w+conv_kw)  
        partB = self.psi(z) * (self.convw(z) + self.convkw(z)) 
        # residual 
        y = partA + partB + x
        # groupnorm + FFN 
        partC = self.ffn(self.gn(y, mask))
        return (y + partC) * m



class MaskedConvHead(nn.Module):

    def __init__(self, C: int,
                 out: int, 
                 layers: int,
                 std: float = 0.0,
                 prior: float = 0.0,
                 final_scale: float = 1.0,
                 zero_final: bool = False,
                 levels: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(C, C, 3, padding=1) for _ in range(layers-1)]
            + [nn.Conv1d(C, out, 3, padding=1)]
        )
        # TODO: ugly writing, basically it means each 
        # layer should have its own head
        # otherwise, level 6's gradient is way too powerful
        # due to the scale.
        # but at the beginning, we use only one head, that head
        # was stored in self.convs (for level 0 only), so now we are
        # extending heads for the 5 levels
        self.finals = nn.ModuleList(
            nn.Conv1d(C, out, 3, padding=1) for _ in range(levels-1)
        )
        if std:
            # head_s, head_e
            for c in self.convs:
                nn.init.normal_(c.weight, 0.0, std)
                nn.init.zeros_(c.bias)
        if final_scale != 1.0:
            # also for head_s, head_e
            # if we don't add this
            # when decoding, the two prob vectors are close to a one-hot
            # vector, aka the model always picks one bin as the endpoint
            with torch.no_grad():
                self.convs[-1].weight.mul_(final_scale)
        if prior:
            # head_cls, prior init from paper
            nn.init.constant_(self.convs[-1].bias, -math.log((1-prior)/prior))
        if zero_final:
            # residual head, output 0 at init, aka no residual
            for c in [self.convs[-1], *self.finals]:
                nn.init.zeros_(c.weight)
                nn.init.zeros_(c.bias)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor,
                level: int = 0) -> torch.Tensor:
        # B, 1, T
        m = mask[:, None].float()
        for c in self.convs[:-1]:
            x = torch.relu(c(x))*m
        if level == 0:
            return self.convs[-1](x)*m
        else:
            return self.finals[level-1](x)*m

class UttHead(nn.Module):
    """
    ASP pooling
    """

    def __init__(self, C: int,
                 att: int,
                 emb: int):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(C, att), nn.Tanh(), nn.Linear(att, 1))
        self.emb = nn.Sequential(nn.Linear(2*C, emb), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(emb, 1), nn.Sigmoid())

    def weights(self, 
                h: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        # get the attention weights: B, T, att -> B, T, 1
        score = self.att(h.transpose(1, 2))[..., 0]
        return score.masked_fill(~m, float("-inf")).softmax(-1)

    def forward(self, 
                h: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        # h: B, C, T
        # m: B, T
        a = self.weights(h, m)[:, None] # B, 1, T
        mu = (h*a).sum(-1) # B, C
        var = (h*h*a).sum(-1) - mu*mu # B, C
        sd = var.clamp_min(1e-6).sqrt() # B, C
        e = self.emb(torch.cat([mu, sd], 1)) # B, 2C -> B, emb
        out = self.out(e)[:, 0]
        return out
        

class TriDet(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        C, Bn = cfg.emb_C, cfg.bins

        self.proj = nn.ModuleList([
            # [B, 1024, Tm] -> [B, C=256, T]
            nn.Conv1d(cfg.feat_dim, C, 3, padding=1), 
            # B, 256, T
            nn.Conv1d(C, C, 3, padding=1)
        ])

        self.blocks = nn.ModuleList(
            SGPBlock(C, cfg.sgp_w, cfg.sgp_k) for _ in range(cfg.levels)
        )

        # to locate the boundary via regressing
        self.head_cls = MaskedConvHead(C, 1, 
                                       cfg.head_layers, 
                                       prior=0.01)
        # -- used to predict probabilty
        self.head_s = MaskedConvHead(C, 1, 
                                     cfg.head_layers, 
                                     std=0.1,
                                     final_scale=0.01)
        self.head_e = MaskedConvHead(C, 1, 
                                     cfg.head_layers, 
                                     std=0.1,
                                     final_scale=0.01)
        self.head_off = MaskedConvHead(C, 2*(Bn+1),
                                       cfg.head_layers)
        # utterance binary prediction
        self.gate = UttHead(C, cfg.utt_att, cfg.utt_emb)
        # -- used to correct the predicted boundary
        self.head_res = MaskedConvHead(C, 2, 
                                       cfg.head_layers,
                                       zero_final=True,
                                       levels=cfg.levels)
        self.register_buffer("bin_idx", torch.arange(Bn+1).float(),
                             persistent=False)

    def _pyramid(self, x: torch.Tensor,
                 mask: torch.Tensor) -> list:
        """
        x = B, 1024, Tm
        6 levels
        x: [B, 1024, T]
        mask: [B, T]
        """
        for cov in self.proj:
            x = torch.relu(cov(x)) * mask[:, None]
        # x: [B, 256, T]
        feats = []
        for i, sgp in enumerate(self.blocks):
            if i:
                x = F.max_pool1d(x, 2)
                mask = F.max_pool1d(mask.float()[:, None], 2)[:, 0] > 0.5
            x = sgp(x, mask)
            feats.append((x, mask))
        return feats


    def _gate(self, feats: list) -> torch.Tensor:
        # for utterance prediction
        # feats: list of [B, 256, T] (six levels)
        # B, 256, total length
        h = torch.cat([f for f, _ in feats], -1)
        # B, total length
        m = torch.cat([m for _, m in feats], -1)
        return self.gate(h, m)

    def _decode(self, fs: torch.Tensor,
                fe: torch.Tensor,
                fc: torch.Tensor,
                res: torch.Tensor,
                mask: torch.Tensor,
                stride: int):
        """
        fs: p of start, [B, T]
        fe: p of end, [B, T]
        fc: p of correction, [B, 2*bins+2, T]
        res: residual, [B, T, 2]
        mask: B, T
        stride 
        Predict the segments for each i

        P_s = softmax(fs[t-B..t] + fc[t, 0])
        ds = sum(b*P_s[b])
        P_e = softmax(fe[t..t+B] + fc[t, 1])
        de = sum(b*P_e[b])
        r = res_range*tanh(res)
        start = (t-ds+r[, 0])*stride
        end = (t+de+r[, 1])*stride
        """
        bins = self.cfg.bins # 16
        B, T = fs.shape
        fc = fc.view(B, 2, bins+1, T) # B, 2, 17, T
        # padding
        pad = fc.new_full((B, bins), NEG)
        # B, T, 17, [pad, pad, 14, 13, 12, ..., 1, t]
        ws = torch.cat([pad, fs], 1).unfold(1, bins+1, 1).flip(-1) 
        # pad right
        fe_ext = torch.cat([fe, pad], 1)
        i = torch.arange(B, device=fe_ext.device)
        last_col = mask.sum(1).clamp_min(1) - 1
        fe_ext[i, last_col+1] = fe[i, last_col]
        we = fe_ext.unfold(1, bins+1, 1)  # B, T, 17
        
        # B, T, 17
        p_s = (ws + fc[:, 0].transpose(1, 2)).softmax(-1)
        p_e = (we + fc[:, 1].transpose(1, 2)).softmax(-1)
        d_s = (p_s * self.bin_idx).sum(-1)
        d_e = (p_e * self.bin_idx).sum(-1)
        r = self.cfg.res_range * torch.tanh(res)
        t = torch.arange(T, device=r.device, dtype=r.dtype)
        pred_s = (t - d_s + r[..., 0])*stride
        pred_e = (t + d_e + r[..., 1])*stride
        segs = torch.stack([pred_s, pred_e], -1)
        return segs

    def forward(self, x: torch.Tensor, 
                mask: torch.Tensor) -> dict:
        #x: shape [B, 1024, Tm]
        # Tm is the multiple of 32
        feats = self._pyramid(x, mask)
        gate = self._gate(feats)

        lcls, lseg, lmasks = [], [], []
        for layer, (feat, m) in enumerate(feats):
            fd = feat.detach()

            fs = self.head_s(fd, m)[:, 0].masked_fill(~m, NEG) # B, T
            fe = self.head_e(fd, m)[:, 0].masked_fill(~m, NEG) # B, T
            # probability correction B, 34, T
            offset = self.head_off(feat, m) 
            # relative dist correction B, T, 2
            res = self.head_res(feat, m, layer).transpose(1, 2) 

            cls = self.head_cls(feat, m)[:, 0] # B, T

            seg = self._decode(
                fs, fe, offset, res, m, 1 << layer
            )

            lcls.append(cls)
            lseg.append(seg)
            lmasks.append(m)

        return {
            "gate": gate,
            "cls": lcls,
            "seg": lseg,
            "masks": lmasks
        }


def assign(segs: torch.Tensor, 
            Tls: list, 
            ranges: list,
            radius: float) -> list:
    """
    center-sampling assignments.
    segs: [G, 2]: gt segments
    Tls: number of instants at pyramid level l, Tls[l] -> int
    ranges: the range of each level
    radius: focus radius

    for each level, the function performs a task 
    similar to frame-level prediction, to predict
    whether the instant t is in the center of the
    segment

    t is in a segment iff 
    - within center += radius * stride
    - in the level range()
    """
    G = len(segs)
    if not G:
        # all bonafide
        return [torch.full((T,), -1, 
                           dtype=torch.long,
                           device=segs.device)
                           for T in Tls]

    device = segs.device
    dur = segs[:, 1] - segs[:, 0]
    center = segs.mean(1)
    out = []
    hit = torch.zeros(G, dtype=torch.long, device=device)

    for l, (T, (low, high)) in enumerate(zip(Tls, ranges)):
        stride = float (1<<l)
        # [T]
        x = torch.arange(T, device=device, dtype=segs.dtype) * stride
        # [T, 1] - [1, G] -> [T, G]
        d_start = x[:, None] - segs[None, :, 0]
        # T, G
        d_end = segs[None, :, 1] - x[:, None]
        d_center = (x[:, None] - center[None]).abs()
        far = torch.maximum(d_start, d_end)

        # [T, G]
        candidate = (d_start >= 0) & (d_end >= 0) \
                    & (far >= low) & (far < high) \
                    & (d_center <= radius * stride)
        # [T, G]
        d_TG = dur[None].expand(T, G).masked_fill(~candidate, float('inf'))
        # [T], we always pick the shortest seg among all candidates
        idx = d_TG.argmin(1)
        idx[~candidate.any(1)] = -1
        out.append(idx)

        hit += torch.bincount(idx[idx>=0], minlength=G)

    # force assignment
    # for a gt, if nothing hits, we force assign a center instant i
    # we search for the level that can cover the sgment, 
    # then pick an instant within that level (we loop over instants
    # from the one closest to center)
    # TODO: the final instant can be outside of the segment, should be
    # fine if 'no hit' is caused by too narrow segment
    # useful if the segment is too narraw (< 1 step)
    lows = torch.tensor([r[0] for r in ranges], device=device, dtype=segs.dtype)
    # [0, 4, 8, 16, 32, 64]
    for g in torch.nonzero(hit == 0).flatten().tolist():
        l = int(torch.searchsorted(lows, (dur[g]/2).reshape(1), right=True)[0]) - 1
        # T
        x = torch.arange(Tls[l], device=device, dtype=segs.dtype) * (1<<l)
        d_center = (x - center[g]).abs().argsort().tolist()
        for i in d_center:
            owner = int(out[l][i])
            if owner < 0 or hit[owner] > 1:
                out[l][i] = g
                hit[g] += 1
                if owner >= 0:
                    hit[owner] -= 1
                break
    return out


def balanced_bce(pred: torch.Tensor, gt: torch.Tensor):
    # for utterance head prediction
    # pred: B, gt: B
    pos = (gt>0.5)
    n, npos = len(gt), pos.sum().clamp_min(1)
    nneg = (n-npos).clamp_min(1)
    # g*lop(p), (1-g)*log(1-p)
    lbce = (n/npos) * pos *  (pred+1e-6).log() + \
           (n/nneg) * (~pos) * (1-pred+1e-6).log()
    return -lbce.sum()/n
            
class TriDetLoss:

    def __init__(self, cfg: Config):
        # adapted TriDet loss
        # + multi tasking
        self.cfg = cfg

    def __call__(self, out: dict,
                 segs: list,
                 y_utt: torch.Tensor, 
                 bound: torch.Tensor) -> dict:
        """
        out: model outputs, per pyramid level we have
          - cls, list of [B, T]
          - seg: list of [B, T, 2], decoded start end
          - masks: list of [B, T]
          - gate: [B]

        segs: gt fake segment in base steps
        y_utt: [B] 1 fake 0 bonafide 
        bound: [B] per-item end bound (used to solve the case where fake segments collide with the 
        last frame)
        """
        cfg = self.cfg

        Tls = [c.shape[1] for c in out['cls']]
        # B, total length(sum of tls)
        logits = torch.cat(out['cls'], dim=1) 
        # B, total length, [start, end]
        pred = torch.cat(out['seg'], 1)

        # prepare probability vectors
        # -- clipping, -offset < s 
        # -- e < end_bound
        # -- B, total length
        p_s = torch.maximum(pred[..., 0], pred.new_tensor(-cfg.grid_offset))
        # B, total length
        p_e = torch.minimum(pred[..., 1], bound.to(pred)[:, None])
        # B, total length
        pred = torch.stack([p_s, p_e], -1)

        # prepare for instant binary classification
        msk = torch.cat(out['masks'], 1)
        # each item: 6 levels, each level: shape T, after cat: total length
        assignments = [torch.cat(assign(s, Tls, cfg.level_ranges, cfg.fcos_radius))
                       for s in segs]
        # B, total length
        idx = torch.stack(assignments)
        # binary cls labels
        pos = (idx >= 0)  & msk
        neg = (idx < 0) & msk
        y = pos.float()
        p = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        pt = p*y + (1-p)*(1-y)
        floss = (cfg.focal_alpha*y + (1-cfg.focal_alpha)*(1-y)) \
                * (1-pt).pow(cfg.focal_gamma)*ce

        # shape, B, total length, 2, to compute IoU
        # --- s: [G, 2] i: total length
        gt = [s[i.clamp_min(0)] if len(s) else s.new_zeros(len(i), 2) 
              for s, i in zip(segs, idx)]
        gt = torch.stack(gt)
        # inter: min(end) - max(st)
        inter = (torch.minimum(pred[..., 1], gt[..., 1])
                -torch.maximum(pred[..., 0], gt[..., 0])).clamp_min(0)
        # union:  
        union = ((pred[..., 1] - pred[..., 0]) + (gt[..., 1] - gt[..., 0])
                 - inter).clamp_min(1e-6)
        tiou = inter/union

        # utt head
        l_utt = balanced_bce(out['gate'], y_utt.to(out['gate'].dtype))

        n_pos = pos.sum().clamp_min(1)
        n_neg = neg.sum().clamp_min(1)
        l_pos = (tiou.detach() * floss * pos).sum() / n_pos
        l_neg = (floss * neg).sum() / n_neg
        l_reg = ((1-tiou) * pos).sum() / n_pos

        # boundary direct regression
        err = (pred-gt).abs() * (cfg.grid * 1e3 / cfg.bnd_tau_ms)
        hub = torch.where(err < 1, 0.5*err*err, err-0.5).sum(-1)
        l_bnd = (hub*pos).sum()/n_pos

        total = l_pos + l_reg + l_neg + cfg.lambda_bnd*l_bnd + cfg.lambda_utt * l_utt

        return {
            "total": total,
            "utt": l_utt,
            "cls_pos": l_pos, 
            "cls_neg": l_neg,
            "reg": l_reg,
            "bnd": l_bnd,
            "n_pos": int(pos.sum()),
            "n_neg":int(neg.sum()) 
        }

