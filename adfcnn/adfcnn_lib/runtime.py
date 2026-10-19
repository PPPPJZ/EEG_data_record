"""设备选择和固定随机种子；Windows 不依赖 shell 或多进程。"""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("device 只能为 cpu、cuda 或 cuda:N")
    index = 0 if device.index is None else device.index
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"请求的 NVIDIA GPU {index} 不可用。请检查驱动及 CUDA 版 PyTorch；"
            "如仅调试 CPU，可显式指定 --device cpu。"
        )
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def set_seed(seed: int) -> None:
    # 在首次 CUDA 运算前配置 cuBLAS；相同软件/硬件环境下可重复。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
