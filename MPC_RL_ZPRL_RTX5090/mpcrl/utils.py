from __future__ import annotations
import os, random
from contextlib import nullcontext
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def configure_accelerator(hardware: dict | None = None) -> torch.device:
    """Configure PyTorch execution for CUDA/RTX 50-series while keeping CPU fallback."""
    hardware = hardware or {}
    device = pick_device()
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = bool(hardware.get('allow_tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(hardware.get('allow_tf32', True))
        torch.backends.cudnn.benchmark = bool(hardware.get('cudnn_benchmark', True))
        torch.set_float32_matmul_precision(str(hardware.get('matmul_precision', 'high')))
    return device


def amp_dtype_from_name(name: str | None):
    name = (name or 'fp32').lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    return None


def autocast_context(device: torch.device | str, precision: str = 'fp32'):
    device = torch.device(device)
    dtype = amp_dtype_from_name(precision)
    if device.type == 'cuda' and dtype is not None:
        return torch.autocast(device_type='cuda', dtype=dtype)
    return nullcontext()


def accelerator_summary(device: torch.device, precision: str) -> str:
    if device.type != 'cuda':
        return f'device=cpu precision=fp32'
    p = torch.cuda.get_device_properties(device)
    cc = f'{p.major}.{p.minor}'
    mem = p.total_memory / 1024**3
    return f'device={p.name} compute_capability={cc} vram={mem:.1f}GiB precision={precision}'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
