import numpy as np

TIOUS = np.arange(0.5, 1.0, 0.05)

def segment_iou(a: np.ndarray,
                b: np.ndarray) -> np.ndarray:
    # a, b [K, 2]
    # IoU matrix
    inter = np.clip(
        np.minimum(a[:, None, 1], b[None, :, 1])
        - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    union = a[:, None, 1] - a[:, None, 0] + b[None, :, 1] - b[None, :, 0] - inter
    return inter / union

def boundary_mae(props: list, gts: list, top_k: int=100) -> tuple:
    # props: list of [K, 3] (start, end, score)
    # gts:   list of [K, 2]  (start, end)
    ds, de, n_gt = [], [], 0
    for p, g in zip(props, gts):
        n_gt += len(g)
        if not len(g) or not len(p):
            continue
        p = p[np.argsort(-p[:, 2])][:top_k]
        iou = segment_iou(g, p[:, :2])
        best, ok = iou.argmax(1), iou.max(1) >= 0.5
        ds += list(np.abs(g[ok, 0] - p[best[ok], 0]))
        de += list(np.abs(g[ok, 1] - p[best[ok], 1]))
    if not ds:
        return float('nan'), float('nan'), 0.0
    # start, end, match
    return 1e3 * np.mean(ds), 1e3 * np.mean(de), len(ds)/max(n_gt, 1)