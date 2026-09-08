"""
Fast implementation of deepfense's range eer.
Only works for range@x ms
x is defined as hop in the interface
"""

import numpy as np

def gt_spoof_mask(gts: np.ndarray,
                  n_frames: int,
                  hop: float = 0.01) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for st, ed in np.asarray(gts, dtype=np.float64).reshape(-1, 2):
        ist, ied = max(0, round(st/hop)), min(n_frames, round(ed/hop))
        mask[ist:ied] = True
    return mask


def _downsample(spoof: np.ndarray,
                scores: np.ndarray,
                f: int):
    # to the position every f steps
    pad = (f - len(spoof)%f) %f
    spoof_pad = np.pad(spoof, (0, pad)).reshape(-1, f).any(1)
    score_pad = np.pad(scores, (0, pad), constant_values=np.inf).reshape(-1, f).min(1)
    return spoof_pad, score_pad


def rasterize(props: np.ndarray,
              n_frames: int,
              hop: float = 0.01): 
    # simple way to assign score
    # pick the max
    # TODO: max fusion
    s = np.zeros(n_frames, dtype=np.float64)
    for st, ed, score in np.asarray(props, dtype=np.float64).reshape(-1, 3):
        ist, ied = max(0, round(st/hop)), min(n_frames, round(ed/hop))
        if ied > ist:
            np.maximum(s[ist: ied], score, out=s[ist: ied])
    return s


def _fast_aggregate(spoof: np.ndarray,
                    score: np.ndarray,
                    thresh: float):
    hyp = (score <= thresh) # positive
    n_fake, n_real = int(spoof.sum()), int((~spoof).sum())
    fnr = float((spoof & ~hyp).sum() / n_fake)  if n_fake else 0.
    fpr = float((~spoof & hyp).sum() / n_real)  if n_real else 0.
    return fpr, fnr


def _search(spoof: np.ndarray,
            score: np.ndarray,
            prec: float,
            max_iters: int):
    # return eer, thresh, fpr, fnr
    sorted_scores = np.sort(score)
    lo_p, hi_p = 0.0, 100.0
    thresh = float(np.percentile(sorted_scores, 50.0))
    f_lo = _fast_aggregate(spoof, score, float(sorted_scores[0]))
    f_mid = _fast_aggregate(spoof, score, thresh)

    # lo = 0, FNR 1, FPR = 0, g < 0
    # hi = 100, FNR 0, FPR = 1, g > 0

    best = (abs(f_mid[0]-f_mid[1]), thresh, *f_mid)
    for _ in range(max_iters):
        if abs(f_mid[0]-f_mid[1]) <= prec:
            break
        if (f_lo[0] - f_lo[1]) * (f_mid[0] - f_mid[1]) < 0:
            # lo, g < 0, mid > 0 -> hi small
            hi_p = (lo_p + hi_p) / 2
        else:
            lo_p = (lo_p + hi_p) / 2
            f_lo = f_mid
        thresh = float(np.percentile(sorted_scores, (lo_p+hi_p) /2))
        f_mid = _fast_aggregate(spoof, score, thresh)

        if abs(f_mid[0] - f_mid[1]) <= best[0]:
            best = (abs(f_mid[0]-f_mid[1]), thresh, *f_mid)
    _, thresh, fpr, fnr = best
    return (fpr + fnr)/2, thresh, fpr, fnr
    
def range_eer(props: list,
              gts: list, 
              n_frames: list,
              factors: tuple = (1, 2), # 10ms and 20ms
              prec: float = 1e-4,
              hop: float = 0.01,
              max_iters: int = 64):
    
    spoof_masks = []
    spoof_scores = []
    for p, g, n in zip(props, gts, n_frames):
        if n:
            spoof_masks.append(gt_spoof_mask(g, n, hop))
            spoof_scores.append(-rasterize(p, n, hop))
    out = {}
    for f in factors:
        if f == 1:
            spoof = np.concatenate(spoof_masks)
            score = np.concatenate(spoof_scores)
        else:
            pairs = [_downsample(s, v, f) for s, v in zip(spoof_masks, spoof_scores)]
            spoof = np.concatenate([a for a, _ in pairs])
            score = np.concatenate([b for _, b in pairs])

        if spoof.all() or not spoof.any():
            out[round(hop * f * 1000)] = (float("nan"), )*4
        else:
            # eer, threshold, fpr, fnr
            out[round(hop * f * 1000)] = _search(spoof, score, 
                                                 prec, max_iters)
    return out