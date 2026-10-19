"""Feedback 可长期复用的推理器，仅加载一次模型。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .config import (
    CHECKPOINT_VERSION, MODEL_VERSION, PREPROCESSING_VERSION,
    ModelConfig, PreprocessingConfig,
)
from .model import ADFCNN
from .preprocessing import preprocess_window
from .runtime import resolve_device


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    probabilities: tuple[float, ...]


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda:0"):
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 teacher 模型: {path}")
        # 文件仅含 Tensor 与 Python 基础类型，不需要导入训练机器上的路径。
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("teacher 文件必须为 checkpoint 字典")
        for key, expected in (
            ("checkpoint_version", CHECKPOINT_VERSION), ("model_version", MODEL_VERSION),
            ("preprocessing_version", PREPROCESSING_VERSION),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"teacher 文件的 {key} 与此代码版本不兼容")
        self.preprocessing = PreprocessingConfig(**payload["preprocessing_config"])
        self.model_config = ModelConfig(**payload["model_config"])
        self.device = resolve_device(device)
        self.model = ADFCNN(self.model_config)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

    def predict(self, raw_uv: np.ndarray) -> int:
        """输入原始 (750, 8) uV，返回 0=左手或 1=右手。"""
        return self.predict_details(raw_uv).class_id

    @torch.inference_mode()
    def predict_details(self, raw_uv: np.ndarray) -> Prediction:
        inputs = preprocess_window(raw_uv, self.preprocessing)
        tensor = torch.from_numpy(inputs[np.newaxis]).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("模型输出包含非有限数值")
        class_id = int(probabilities.argmax())
        return Prediction(
            class_id=class_id, class_name=self.preprocessing.class_names[class_id],
            probabilities=tuple(float(value) for value in probabilities),
        )
