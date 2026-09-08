import numpy as np

from core.nms import soft_nms

def utt_proposals(cls:list, seg: list, 
                  mask: list, end_bound: float,
                  cfg) -> np.ndarray:
    # prepare the proposal for utt
    # cls, seg, mask, results from 6 levels
    off = cfg.grid_offset
    rows = []
    for lc, ls, lm in zip(cls, seg, mask):
        score = 1.0 / (1.0 + np.exp(-np.clip(lc[lm], -30, 30)))
        start_end = np.clip(ls[lm], -off, end_bound)
        keep = start_end[:, 1] > start_end[:, 0]
        if keep.any():
            rows.append(np.concatenate([
                start_end[keep], score[keep, None]
            ], 1))
    if not rows:
        return np.zeros((0, 3), np.float32)
    props = np.concatenate(rows)
    props = props[np.argsort(-props[:, 2])][:cfg.pre_nms_topk]
    # to seconds
    props[:, :2] = (props[:, :2]+off) * cfg.grid
    return soft_nms(props.astype(np.float32), cfg.soft_nms_sigma, cfg.top_k)
