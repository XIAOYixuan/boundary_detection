import numpy as np
import pandas as pd

SEGLAB = {"train": "train", "dev": "dev", "test": "eval"}

def fake_segments(lab: np.ndarray) -> np.ndarray:
    """
    turn frame level labels to [N, 2]
    N, 2 -> N segments, (start, end)
    """
    lab = (lab == '0').astype(np.int8)
    f = np.concatenate(([0], lab, [0]))
    d = np.diff(f)
    st = np.flatnonzero(d==1)*0.01
    ed = np.flatnonzero(d==-1)*0.01
    return np.stack([st, ed], 1)


class PartialSpoof:

    def __init__(self, cfg, split):
        df = pd.read_parquet(f"{cfg.data_dir}/{split}.parquet")
        # partialspoof and partialedit
        # we need this to make comparison
        # rangeEER @ 10ms
        lab = np.load(f"{cfg.seglab_dir}/{SEGLAB[split]}_seglab_0.01.npy",
                      allow_pickle=True).item()
        self.ids = df.ID.str.removeprefix("partialspoof_").tolist()
        self.paths = df.path.tolist()
        self.spoof = (df.label == "spoof").to_numpy()
        self.segs = []
        self.n_lab = []
        for i in self.ids:
            if i in lab:
                self.segs.append(fake_segments(lab[i]))
                self.n_lab.append(len(lab[i]))
            else:
                self.segs.append(np.zeros((0, 2)))
                self.n_lab.append(0)
