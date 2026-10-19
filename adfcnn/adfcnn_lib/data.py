"""解析原始 H5 trial，固定读取 shuffle，并构建分层五折。"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .config import LOAD_SHUFFLE_SEED, N_FOLDS, PreprocessingConfig
from .preprocessing import preprocess_window


@dataclass(frozen=True)
class SessionData:
    source_path: Path
    inputs: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    preprocessing: PreprocessingConfig
    summary: dict


def read_json(source: h5py.File, key: str) -> dict:
    value = source[key][()]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError(f"{key} 必须包含 JSON 对象")
    return payload


def load_session(
    data_file: str | Path, preprocessing: PreprocessingConfig | None = None
) -> SessionData:
    """只读 H5；不缓存、不写文件，不使用 metadata.output_dir。"""
    config = PreprocessingConfig() if preprocessing is None else preprocessing
    path = Path(data_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 H5 数据文件: {path}")
    with h5py.File(path, "r") as source:
        required = ("metadata/json", "protocol/json", "eeg/raw_uv", "trials/table")
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"H5 缺少字段: {missing}")
        protocol = read_json(source, "protocol/json")
        metadata = read_json(source, "metadata/json")
        if protocol.get("sample_rate_hz") != config.sampling_rate_hz:
            raise ValueError("H5 采样率必须为 250 Hz")
        if protocol.get("channel_count") != len(config.channel_names):
            raise ValueError("H5 必须包含 8 个通道")
        if tuple(protocol.get("channel_labels", [])) != config.channel_names:
            raise ValueError("H5 通道顺序必须为 CH1–CH8")
        if protocol.get("signal_unit") != config.signal_unit:
            raise ValueError("H5 原始信号单位必须为 uV")
        expected_codes = {str(i): name for i, name in enumerate(config.class_names)}
        if protocol.get("class_codes") != expected_codes:
            raise ValueError("H5 类别映射必须为 0=left_hand、1=right_hand")
        raw = source["eeg/raw_uv"]
        if raw.ndim != 2 or raw.shape[1] != len(config.channel_names):
            raise ValueError(f"原始 EEG 必须为 [samples, 8]，实际为 {raw.shape}")
        trials = source["trials/table"][:]
        fields = {"trial_id", "class_id", "completed", "cue_start_sample", "trial_end_sample"}
        if trials.ndim != 1 or not fields.issubset(trials.dtype.names or ()):
            raise ValueError(f"trials/table 必须包含 {sorted(fields)}")
        if len(np.unique(trials["trial_id"])) != len(trials):
            raise ValueError("trial_id 不可重复")
        # 先按 trial_id 固定初始顺序，再同步 shuffle 数据和标签。
        trials = trials[np.argsort(trials["trial_id"], kind="stable")]
        completed = np.asarray(trials["completed"], dtype=bool)
        starts = np.asarray(trials["cue_start_sample"], dtype=np.int64)
        ends = np.asarray(trials["trial_end_sample"], dtype=np.int64)
        valid = (
            completed & np.isin(trials["class_id"], [0, 1]) & (starts >= 0)
            & (ends <= len(raw)) & (starts <= ends - config.window_samples)
            & (starts <= len(raw) - config.window_samples)
        )
        rows = trials[valid]
        counts = np.bincount(rows["class_id"].astype(np.int64), minlength=2)
        if counts.min() < N_FOLDS:
            raise ValueError(
                f"有效左手/右手 trial 数为 {counts.tolist()}，分层五折要求每类至少 5 个"
            )
        ordered_starts = np.sort(rows["cue_start_sample"])
        if np.any(np.diff(ordered_starts) < config.window_samples):
            raise ValueError("不同 trial 的 3 秒窗口发生重叠，无法安全划分 train/val/test")
        permutation = np.random.default_rng(LOAD_SHUFFLE_SEED).permutation(len(rows))
        rows = rows[permutation]
        inputs = []
        for row in rows:
            start = int(row["cue_start_sample"])
            window = raw[start:start + config.window_samples]
            try:
                inputs.append(preprocess_window(window, config))
            except ValueError as error:
                raise ValueError(f"trial {int(row['trial_id'])}: {error}") from error
        summary = {
            "subject_id": str(metadata.get("subject_id", path.parent.parent.name)),
            "session": path.parent.name,
            "status": str(metadata.get("status", "unknown")),
            "planned_trials": len(trials),
            "completed_trials": int(completed.sum()),
            "usable_trials": len(rows),
            "excluded_trials": len(trials) - len(rows),
            "class_counts": counts.tolist(),
            "load_shuffle_seed": LOAD_SHUFFLE_SEED,
        }
    return SessionData(
        source_path=path, inputs=np.stack(inputs),
        labels=np.asarray(rows["class_id"], dtype=np.int64),
        trial_ids=np.asarray(rows["trial_id"], dtype=np.int64),
        preprocessing=config, summary=summary,
    )


def make_folds(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels 必须为一维 0/1 数组")
    if np.bincount(labels, minlength=2).min() < N_FOLDS:
        raise ValueError("分层五折要求每类至少 5 个 trial")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_ids = np.empty(len(labels), dtype=np.int64)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        fold_ids[held_out] = fold
    return fold_ids


def split_fold_indices(fold_ids: np.ndarray, test_fold: int) -> tuple[np.ndarray, ...]:
    """test=K，val=K%5+1，其余三折 train。"""
    folds = np.asarray(fold_ids)
    if folds.ndim != 1 or set(np.unique(folds)) != set(range(1, N_FOLDS + 1)):
        raise ValueError("fold_ids 必须为包含 1–5 的一维数组")
    if test_fold not in range(1, N_FOLDS + 1):
        raise ValueError("test_fold 必须为 1–5")
    validation_fold = test_fold % N_FOLDS + 1
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    validation = np.flatnonzero(folds == validation_fold)
    test = np.flatnonzero(folds == test_fold)
    return train, validation, test
