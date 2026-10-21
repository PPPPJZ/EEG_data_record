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
    N_FOLDS, ModelConfig, TrainingConfig,
)
from .data import SessionData, split_fold_indices
from .training import FoldResult


def save_teacher(
    destination: str | Path, session: SessionData, result: FoldResult,
    fold_ids: np.ndarray, test_fold: int, split_seed: int,
    training: TrainingConfig, model_config: ModelConfig,
) -> Path:
    path = Path(destination).expanduser().resolve()
    if len(fold_ids) != len(session.labels):
        raise ValueError("fold_ids 与 trial 数量不一致")
    train = result.train_indices
    validation = result.validation_indices
    test = result.test_indices
    original_train, original_validation, original_test = split_fold_indices(
        fold_ids, test_fold
    )
    if (
        not set(train).issubset(original_train)
        or not set(validation).issubset(original_validation)
        or not np.array_equal(np.sort(test), original_test)
    ):
        raise ValueError("训练结果的平衡划分与 fold_ids 不一致")
    class_balance = {
        key: value for key, value in result.class_balance.items()
        if key not in {"dropped_train_indices", "dropped_validation_indices"}
    }
    class_balance["dropped_train_trial_ids"] = session.trial_ids[
        result.class_balance["dropped_train_indices"]
    ].tolist()
    class_balance["dropped_validation_trial_ids"] = session.trial_ids[
        result.class_balance["dropped_validation_indices"]
    ].tolist()
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
        "split_seed": split_seed,
        "training_seed": result.training_seed,
        "test_fold": test_fold,
        "validation_fold": test_fold % N_FOLDS + 1,
        "class_balance": class_balance,
        "train_trial_ids": session.trial_ids[train].tolist(),
        "validation_trial_ids": session.trial_ids[validation].tolist(),
        "test_trial_ids": session.trial_ids[test].tolist(),
        "best_epoch": result.best_epoch,
        "validation_accuracy": result.validation_accuracy,
        "validation_loss": result.validation_loss,
        "test_accuracy": result.test_accuracy,
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
