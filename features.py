import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from w2v2 import load_xlsr


def _group(name: str) -> tuple:
    """
    based on the name, return which layer has it
    -1, no layer
    """
    if name.startswith("feature_extractor."):
        return "extractor", -1
    if name.startswith(("post_extract_proj.", "layer_norm.")):
        return "proj_ln", -1
    if name.startswith("encoder.pos_conv."):
        return "pos_conv", -1
    if name.startswith("encoder.layers."):
        return "layers", int(name.split(".")[2])
    if name.startswith("encoder.layer_norm."):
        return "final_ln", -1
    if name == "mask_emb":
        return "mask_emb", -1
    raise KeyError(name)


def ssl_state(front: nn.Module) -> dict:
    return {
        k: v.detach().cpu().clone()
          for k, v in front.ssl.state_dict().items()
    }


def set_trainable(ssl: nn.Module, cfg: Config) -> dict:
    """
    freeze all except for the layers defined in cfg
    """
    ssl.requires_grad_(False)
    layers = set(cfg.ft_layer_list)
    on = {"extractor": bool(cfg.ft_extractor), 
          "proj_ln": 0 in layers,
          "pos_conv": 0 in layers,
          "final_ln": True,
          "mask_emb": False}

    for n, p in ssl.named_parameters():
        g, i = _group(n)
        if g == "layers":
            p.requires_grad = i in layers
        else:
            p.requires_grad = on[g]


class W2V2Front(nn.Module):
    """
    waveform to [B, 1024, Tm]
    then mean-normalized, left-padded, batch padded
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ssl = load_xlsr(cfg.w2v_ckpt)
        self.ft = set_trainable(self.ssl, cfg)
        self.ssl.encoder.grad_ckpt = bool(cfg.grad_ckpt)

    @classmethod
    def from_ckpt(cls, cfg: Config, ckpt: dict) -> "W2V2Front":
        """Frozen frontend for eval"""
        m = cls(cfg)
        m.ssl.load_state_dict(ckpt['ssl'], strict=True)
        return m.eval().requires_grad_(False)

    @staticmethod
    def wave(path: str, sr: int=16000) -> torch.Tensor:
        """
        mimic fairseq's logic to read wave
        fairseq normalized = True
        """
        w, r = sf.read(path, dtype='float32', always_2d=True)
        assert r == sr, f"{path}: sr {r}"
        x = torch.from_numpy(w.mean(1))
        return F.layer_norm(x, x.shape)

    def _hidden(self, waves: torch.Tensor,
                n_samples: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=waves.is_cuda):
            hidden = self.ssl(waves, layer=self.cfg.w2v_layer, n_samples=n_samples)
        return hidden.float().transpose(1, 2)

    def forward(self, waves: torch.Tensor,
                n_samples: torch.Tensor, 
                n_lab: torch.Tensor,
                mult: int 
                ) :
        """
        We need to pad the label frames to a multiple of 
        'mult' (32 for 6 levels), then we mask out the 
        remaining one
        """
        cfg = self.cfg
        f = self._hidden(waves, n_samples) # [B, 1024, T'_max]
        Tp = self.ssl.frames(n_samples.to(f.device)) #[B]
        # mask
        valid = torch.arange(f.shape[2], device=f.device)[None] < Tp[:, None] #[B, Tmax]
        # mask out the invalid part, then compute mean
        # then subtract the f by mean 
        # mean - normalization
        f = f - (f*valid[:, None]).sum(2, keepdim=True) / Tp[:, None, None] # [B, 1024, Tmax]
        # left pad
        f = F.pad((f * valid[:, None]), (cfg.pad_frames, 0))

        # prepare masks for tridet
        n_lab = n_lab.to(f.device) # [B]
        # TODO: should use n_samples to compute
        # for fast verification, we just use n_lab to 
        # compute how many 20ms-grides are there
        instant_length = n_lab * 160 // cfg.hop + 1
        fallback = cfg.pad_frames + Tp 
        T = torch.where(n_lab > 0, instant_length, fallback)
        Tm = -((-int(T.max()))//mult) * mult
        mask = torch.arange(Tm, device=f.device)[None] < T[:, None] # [B, Tm]

        f = F.pad(f, (0, max(0, Tm - f.shape[2])))[:, :, :Tm]
        f = f * mask[:, None]
        return f, mask

    @torch.no_grad()
    def single(self, path: str) -> torch.Tensor:
        """
        infer a single audio
        """
        cfg = self.cfg
        x = self.wave(path, cfg.sr)[None].to(next(self.parameters()).device)
        m = self._hidden(x, None)[0].cpu() # 1024, T
        m = m-m.mean(1, keepdim=True)
        return F.pad(m, (cfg.pad_frames, 0))

if __name__ == "__main__":
    from w2v2 import _Wav2Vec2SSL
    ssl = _Wav2Vec2SSL()
    set_trainable(ssl, Config())
    param = sum(p.requires_grad for p in ssl.parameters())
    print('num of param', param)
    numel = sum(p.numel() for p in ssl.parameters() if p.requires_grad)
    print('numel', numel)