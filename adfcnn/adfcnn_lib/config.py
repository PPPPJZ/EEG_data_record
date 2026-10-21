"""独立 ADFCNN 的显式参数与文件格式约定。"""
from __future__ import annotations
from dataclasses import dataclass
import math

# 不使用采集文件内的随机种子。读取 shuffle 和实验划分互不依赖。
LOAD_SHUFFLE_SEED = 42
RESULTS_SEEDS = (42, 2024, 3407)
TEACHER_SEED = 42
N_FOLDS = 5
TEACHER_TEST_FOLD = 1
TEACHER_FILENAME = "adfcnn_teacher.pt"
CHECKPOINT_VERSION = 1
MODEL_VERSION = "adfcnn_750_v1"
PREPROCESSING_VERSION = "raw_uv_fir40_window_zscore_v1"

@dataclass(frozen=True)
class PreprocessingConfig:
    """参数写入 teacher 文件，推理时从文件恢复。"""
    sampling_rate_hz: int = 250
    window_samples: int = 750
    channel_names: tuple[str, ...] = (
        "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"
    )
    class_names: tuple[str, ...] = ("left_hand", "right_hand")
    signal_unit: str = "uV"
    lowpass_hz: float = 40.0
    filter_method: str = "fir"
    filter_length: str = "auto"
    transition_bandwidth: str = "auto"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    filter_phase: str = "zero"
    filter_pad: str = "reflect_limited"
    standardization_std_floor: float = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        if self.sampling_rate_hz != 250 or self.window_samples != 750:
            raise ValueError("此模型要求 250 Hz、750 点的 3 秒窗口")
        if self.channel_names != tuple(f"CH{i}" for i in range(1, 9)):
            raise ValueError("通道顺序必须为 CH1–CH8")
        if self.class_names != ("left_hand", "right_hand") or self.signal_unit != "uV":
            raise ValueError("类别必须为左手/右手，信号单位必须为 uV")
        if not math.isfinite(self.lowpass_hz) or not 0 < self.lowpass_hz < 125:
            raise ValueError("低通频率必须在 (0, 125) Hz 内")
        if self.filter_method != "fir" or self.filter_phase != "zero":
            raise ValueError("此版本采用零相位 FIR 窗口滤波")
        if not math.isfinite(self.standardization_std_floor) or self.standardization_std_floor <= 0:
            raise ValueError("标准差下限必须为正数")

@dataclass(frozen=True)
class ModelConfig:
    num_channels: int = 8
    num_classes: int = 2
    window_samples: int = 750
    features: int = 8
    dropout: float = 0.25
    convolution_max_norm: float = 2.0
    log_floor: float = 1e-6

    def __post_init__(self) -> None:
        if (self.num_channels, self.num_classes, self.window_samples, self.features) != (8, 2, 750, 8):
            raise ValueError("此版本固定为 8 通道、2 类、750 点、8 个特征")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")
        if (not math.isfinite(self.convolution_max_norm) or not math.isfinite(self.log_floor)
                or self.convolution_max_norm <= 0 or self.log_floor <= 0):
            raise ValueError("权重范数上限和 log 下限必须为正数")

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.075
    device: str = "cuda:0"
    num_workers: int = 0
    balance_train_validation: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size 必须为正整数，num_workers 不能为负")
        if not isinstance(self.balance_train_validation, bool):
            raise ValueError("balance_train_validation 必须为布尔值")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正数")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay 必须为非负数")
