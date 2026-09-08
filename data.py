import json 

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
import soundfile as sf

from config import Config
from core.partialspoof import PartialSpoof
from features import W2V2Front


class ContLabels(PartialSpoof):
    """
    Used for partialedit. 
    If conff.cont_labels is 1, then we should use the exact 
    region times. Note, the duration inferred from the provided
    label doesn't match the exact audio duration, so using 
    this wrapper, will change the gt labels for the segments 
    that collide the audio end
    """
    def __init__(self, cfg: Config, split: str):
        super().__init__(cfg, split)
        self._n = None
        assert cfg.index_path or not cfg.cont_labels, \
            "cont_labels 1 need the index path"
        if not cfg.index_path:
            return

        idx = pd.read_parquet(cfg.index_path, columns=["ID", "n_samples", "segments"])
        idx = idx.set_index("ID").reindex(self.ids)
        assert not idx.segments.isna().any(), \
            f"{int(idx.segments.isna().sum())} ids missing in {cfg.index_path}"

        self._n = [int(v) for v in idx.n_samples]
        if not cfg.cont_labels:
            return

        # recomputing the segment, use the exact region
        self.segs = [self.clip(json.loads(js), cfg.dur_bound(n))
                     for js, n in zip(idx.segments, self._n)]

    @staticmethod
    def clip(regions: list, hi: float) -> np.ndarray:
        # make sure the labels start and end are all valid
        # 0<= start < end < audio duration
        a = np.asarray(regions, np.float64).reshape(-1, 2)
        a = np.stack([a[:, 0].clip(0.0), a[:, 1].clip(None, hi)], 1) if len(a) else a
        a = a[(a[:, 0] < hi) & (a[:, 1] > a[:, 0])] if len(a) else np.zeros((0, 2))
        # sort by start
        return a[np.argsort(a[:, 0], kind="stable")] if len(a) else np.zeros((0, 2))


    @property
    def n_samples(self) -> list:
        if self._n is None:
            self._n = [sf.info(p).frames for p in self.paths]
        return self._n


class BucketBatchSampler(Sampler):

    def __init__(self, 
                 n_samples: list,
                 batch: int, 
                 chunk: int, 
                 seed: int):
        super().__init__()

        self.n = np.asarray(n_samples) # [N]
        self.batch = batch
        self.chunk = chunk
        self.seed = seed
        self.epoch = 0
        assert chunk % batch == 0 and batch > 0, (chunk, batch)

    def batches(self, epoch: int) -> list:
        rng = np.random.default_rng([self.seed, epoch])
        perm = rng.permutation(len(self.n))
        # list of batches
        out = []

        for c in range(0, len(perm), self.chunk):
            ch = perm[c: c+self.chunk]
            ch = ch[np.argsort(self.n[ch], kind='stable')]

            for b in range(0, len(ch) - self.batch+1, self.batch):
                out += [ch[b: b+self.batch].tolist()]

        rng.shuffle(out)
        return out

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self.batches(self.epoch))

    def __len__(self):
        return len(self.n) // self.batch

class UttDataset(Dataset):

    def __init__(self, cfg: Config, split: str):
        self.cfg, self.ds = cfg, ContLabels(cfg, split)

    def __len__(self):
        return len(self.ds.ids)

    def __getitem__(self, index):
        cfg = self.cfg
        x = W2V2Front.wave(self.ds.paths[index], cfg.sr) 
        segs = self.ds.segs[index] # [G, 2]
        segs = torch.from_numpy(
            (segs / cfg.grid - cfg.grid_offset).clip(-cfg.grid_offset)
        ).float()
        # [G, 2]
        return x, len(x), self.ds.n_lab[index], segs, float(self.ds.spoof[index])


def collate(batch):
    xs, ns, ms, segs, ys = zip(*batch)
    waves = torch.zeros(len(batch), max(ns))
    for i, x in enumerate(xs):
        waves[i, :len(x)] = x
    return waves, torch.tensor(ns), torch.tensor(ms), list(segs), torch.tensor(ys)

