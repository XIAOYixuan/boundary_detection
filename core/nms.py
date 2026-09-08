import numpy as np

from core.metrics import segment_iou

def soft_nms(props: np.ndarray, sigma: float, top_k: int) -> np.ndarray:
    props = props.copy()
    keep = []

    # props: [(start, end, score)], shape [N, 3]
    while len(props) and len(keep) < top_k:
        # greedy, always pick the max score
        i = int(props[:, 2].argmax())
        keep.append(props[i])
        p, props = props[i], np.delete(props, i, 0)
        if len(props):
            iou = segment_iou(props[:, :2], p[None, :2])[:, 0]
            props[:, 2] *= np.exp(-iou ** 2 / sigma) 
    
    if len(keep) == 0:
        return np.zeros((0, 3), np.float32)
    return np.stack(keep)

