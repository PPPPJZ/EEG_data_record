"""训练与推理共用的窗口处理，不读取相邻 trial 的数据。"""
from __future__ import annotations
import mne
import numpy as np
from .config import PreprocessingConfig


def preprocess_window(raw_uv: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """(750, 8) 原始 uV -> (1, 8, 750) float32 模型输入。"""
    values = np.asarray(raw_uv)
    expected = (config.window_samples, len(config.channel_names))
    if values.shape != expected:
        raise ValueError(f"原始窗口必须为 {expected}，实际为 {values.shape}")
    if values.dtype.kind not in "fiu":
        raise ValueError("EEG 必须为实数数值数组")
    values = np.asarray(values.T, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("原始 EEG 窗口包含 NaN 或 Inf")
    filtered = mne.filter.filter_data(
        values, sfreq=config.sampling_rate_hz, l_freq=None, h_freq=config.lowpass_hz,
        filter_length=config.filter_length, h_trans_bandwidth=config.transition_bandwidth,
        method=config.filter_method, phase=config.filter_phase, fir_window=config.fir_window,
        fir_design=config.fir_design, pad=config.filter_pad, n_jobs=1, copy=True, verbose="ERROR",
    )
    mean = filtered.mean(axis=-1, keepdims=True)
    std = filtered.std(axis=-1, keepdims=True, ddof=0)
    standardized = (filtered - mean) / np.maximum(std, config.standardization_std_floor)
    output = np.ascontiguousarray(standardized[np.newaxis], dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("预处理产生了非有限数值")
    return output
