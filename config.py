import argparse
from dataclasses import dataclass, fields
from pathlib import Path

@dataclass
class Config:
    sr: int = 16000 # sampling rate
    hop: int = 320 # resolution 20ms
    # for partialspoof, it's 10ms cuz they provided the label
    #  at 10ms
    # for partialedit and llamapartialspoof, we use 0.001
    precision: float = 0.01 

    # w2v2 hyperparam
    w2v_ckpt: str = "" # set in the config
    w2v_layer: int = 24
    w2v_dim: int = 1024
    ft_extractor: int = 0 # mark whether we ft the cov extractor
    ft_layers: str = "" # empty to finetune all

    # w2v2 ft hyper param
    ssl_lr: float = 1e-5
    ssl_wd: float = 0.0 # weight decay

    # adpated tridet model hyper param
    emb_C: int = 256 # embedding channels
    # -- sgp for segment pred
    levels: int = 6
    bins: int = 16
    sgp_w: int = 1
    sgp_k:float = 1.5
    # -- frame-level pred head
    head_layers: int = 3
    fcos_radius: float = 1.5
    # -- offset 
    res_range:float  = 0.5 # half of the resolution, 0 is none
    # -- end window pad 
    end_ext: int = 1

    # gate model hyper param, utt level head
    lambda_utt: float = 1.0 
    utt_detach: int = 0
    utt_att: int = 128
    utt_emb: int = 256
    gate_lr: float = 1e-3
    gate_wd: float = 1e-4
    gate_epochs: int = 5

    # alignment
    # NOTE: only works for w2v2, 
    # remember to recalculate this when using 
    # dsp features
    pad_frames: int = 1
    grid_offset: float = -0.375 # used to center the frame


    # boundary exact regression
    lambda_bnd: float = 0.05
    bnd_tau_ms: float = 20.0

    # loss smoothing
    neg_norm: int = 1
    norm_init: float = 50.0
    norm_mom: float = 0.9 # momentum

    # training hyper-param
    batch: int = 8
    # 1 : we sort the utterance, shorter first
    # used for datasets whose audio duration 
    # varies by a lot (llamapartialspoof)
    bucket: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 30
    warmup_epochs: int = 5
    ## early stop
    patience: int = 3
    dev_utts: int = 2000
    ## focal loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    ## others
    seed: int = 17
    num_workers:int = 4
    log_every: int = 50
    # used for llamapartialspoof, 
    # otherwise it won't fit the 48g gpu card
    grad_ckpt: int = 0

    # inference hyper-param
    pre_nms_topk: int = 200
    soft_nms_sigma: float = 0.5
    top_k: int = 100
    ## two step eval, used to mute the segment prediction
    gate_thr: float = 0.0 # mute all if gate < gate_thr
    init_ckpt: str = ""

    # eval
    split: str = "dev" 
    ckpt: str = ""
    name: str = "v8"
    data_dir:str = "" # parquet files
    seglab_dir: str = "" # seglab npy provided by partialspoof
    cont_labels: int = 0 # continuous label
    index_path: str = "" # parquet with columns ID, n_samples, segments, if cont_label is True, then we need this

    out_dir: str = ""

    @property
    def grid(self) -> float:
        return float(self.hop) / self.sr

    @property
    def bucket_chunk(self) -> int:
        return 100 * self.batch

    @property
    def feat_dim(self) -> int:
        return self.w2v_dim

    @property
    def ft_layer_list(self) -> list:
        if not self.ft_layers:
            return list(range(24))
        a, b = (int(v) for v in self.ft_layers.split('-'))
        return list(range(a, b+1))

    def dur_bound(self, n_samples):
        # n_samples: can be a batch (used in train.py) or int
        # wave length rounded up to the precision (in seconds)
        spp = int(round(self.sr * self.precision))
        q = -(-n_samples // spp)
        return (q.double() if hasattr(q, 'double') else float(q)) * self.precision

    def end_bound(self, n_samples):
        # TODO: to reproduce partialspoof, the end should 
        # be defined by the label npz, not the audio duration
        # there is some duration mismatch issue between
        # audio duration and duration inferred from label
        return self.dur_bound(n_samples) / self.grid - self.grid_offset

    @property
    def level_ranges(self) -> list:
        # lv1 = [0, 4)
        # lv2 = [4, 8)
        # lv3 = [8, 16)
        # lv4 = [16, 32
        # lv5 =  32, 64
        # lv6 = 64, inf
        # lvi(i>1) = 2^i, 2^(i+1)
        lo = [0.0] + [float(1 << l) for l in range(2, self.levels+1)]
        hi = lo[1:] + [float("inf")]
        return list(zip(lo, hi))

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / "runs" / self.name

    @classmethod
    def from_args(cls, argv=None) -> "Config":
        ap = argparse.ArgumentParser()
        for f in fields(cls):
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
        return cls(**vars(ap.parse_args(argv)))


