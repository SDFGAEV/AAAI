import hashlib
from ..mineclip import MineCLIP


def load(cfg, device):
    cfg = cfg.copy()
    ckpt = cfg.pop("ckpt")
    assert (
        hashlib.md5(open(ckpt["path"], "rb").read()).hexdigest() == ckpt["checksum"]
    ), "broken ckpt"

    model = MineCLIP(**cfg)
    # Fix: "geqrf_cpu" not implemented for 'Half' — QR decomposition used by
    # orthogonal_ init doesn't support fp16 on CPU. Force float32 for init,
    # then cast back to the target dtype after moving to device.
    model = model.float().to(device)
    model.load_ckpt(ckpt["path"], strict=True)
    return model
