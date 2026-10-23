"""teacher 权重与推理约定保存为一个跨平台文件。"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import mne
import numpy as np
import scipy
import sklearn
import torch
import h5py
from .config import (
    CHECKPOINT_VERSION, MODEL_VERSION, PREPROCESSING_VERSION,
    ModelConfig, TrainingConfig,
)
from .data import SessionData
from .training import TrainingResult


def save_teacher(
    destination: str | Path, session: SessionData, result: TrainingResult,
    training: TrainingConfig, model_config: ModelConfig,
) -> Path:
    path = Path(destination).expanduser().resolve()
    if path == session.source_path.resolve():
        raise ValueError("模型输出路径不可与输入 H5 相同")
    train = result.train_indices
    if not np.array_equal(np.sort(train), np.arange(len(session.labels))):
        raise ValueError("teacher 必须使用全部有效带标签 trial 训练")
    if result.final_epoch != training.epochs:
        raise ValueError("teacher 必须完成全部配置 epochs")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_state_dict": result.state_dict,
        "model_config": asdict(model_config),
        "preprocessing_config": asdict(session.preprocessing),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(session.source_path),
        "session_summary": session.summary,
        "training_config": asdict(training),
        "training_protocol": {
            "data": "all_usable_labeled_trials",
            "validation": False,
            "test": False,
            "class_resampling": "none",
            "checkpoint_selection": "final_epoch",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": training.epochs,
            "scheduler_eta_min": 0.0,
        },
        "training_seed": result.training_seed,
        "train_class_counts": np.bincount(session.labels[train], minlength=2).tolist(),
        "train_trial_ids": session.trial_ids[train].tolist(),
        "final_epoch": result.final_epoch,
        "final_train_loss": result.final_train_loss,
        "final_learning_rate": result.final_learning_rate,
        "software_versions": {
            module.__name__: str(module.__version__)
            for module in (torch, np, scipy, mne, h5py, sklearn)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先完成临时文件，再替换目标，避免中断留下不完整 checkpoint。
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
