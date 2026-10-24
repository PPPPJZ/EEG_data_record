"""集中定义运动想象采集系统的固定参数。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# 新 EEG：本程序作为服务器，厂家程序向此地址推送 eeg_data。
EEG_WS_HOST: Final = "127.0.0.1"
EEG_WS_PORT: Final = 8765
EEG_WS_CHANNELS: Final = 32
EEG_WS_TIMEOUT_SECONDS: Final = 3.0
EEG_WS_MAX_MESSAGE_BYTES: Final = 8 * 1024 * 1024
EEG_WS_MAX_CHUNK_SAMPLES: Final = 2500
# 按 Teacher 训练时的电极顺序填写厂家通道号（从 1 开始），必须恰好 8 个。
# 只影响模型输入；显示与保存始终保留全部采集通道。
TEACHER_CHANNELS: Final = ()

# 旧 BLE 解码参数，仅供历史协议工具与 8 通道演示使用。
TARGET_DEVICE_NAME: Final = "Motren_E"
EEG_DATA_UUID: Final = "fec26ec4-6d71-4442-9f81-55bc21d658d6"
CONTROL_UUID: Final = "7812a8f6-515a-4698-8bdc-a751b65c8e2c"
START_ACQUISITION: Final = b"\x02"
STOP_ACQUISITION: Final = b"\x03"

CHANNEL_COUNT: Final = 8
CHANNEL_LABELS: Final = tuple(f"CH{index}" for index in range(1, 9))
SAMPLE_RATE_HZ: Final = 250
SAMPLES_PER_PACKET: Final = 5
ADC_UV_PER_LSB: Final = (4.5 / 8.0) * 1_000_000.0 / (2**23)

# 实时滤波，沿用 pure_MI 的两级带通
NOTCH_HZ: Final = 50.0
NOTCH_Q: Final = 30.0
BANDPASS_LOW_HZ: Final = 5.0
BANDPASS_HIGH_HZ: Final = 30.0
BANDPASS_ORDER: Final = 4
BANDPASS_CASCADE_COUNT: Final = 2

# 实验时序
PREPARATION_SECONDS: Final = 5.0
REST_SECONDS: Final = 2.5
CUE_SECONDS: Final = 1.5
IMAGERY_BLACK_SECONDS: Final = 4.0
TASK_SECONDS: Final = CUE_SECONDS + IMAGERY_BLACK_SECONDS
TRIAL_SECONDS: Final = REST_SECONDS + TASK_SECONDS
# self_learning：先自主选侧，再进行固定时长的运动想象与光标反馈。
SELF_LEARNING_REST_SECONDS: Final = 3.0
SELF_LEARNING_TASK_SECONDS: Final = 10.0
SELF_LEARNING_TRIAL_SECONDS: Final = SELF_LEARNING_REST_SECONDS + SELF_LEARNING_TASK_SECONDS
DEFAULT_LEARNING_TRIALS: Final = 100
DEFAULT_BREAK_EVERY_TRIALS: Final = 20
BREAK_INSTRUCTION_SECONDS: Final = 3.0
BREAK_COUNTDOWN_SECONDS: Final = 30.0
BREAK_CHIME_REMAINING_SECONDS: Final = 6.0

# 类别与事件编码
CLASS_NONE: Final = -1
CLASS_LEFT: Final = 0
CLASS_RIGHT: Final = 1
CLASS_NAMES: Final = {
    CLASS_LEFT: "left_hand",
    CLASS_RIGHT: "right_hand",
}

# 实验模式与触觉刺激；每 0.25 s 切换一次，完整高低周期为 0.5 s
MODE_PURE_MI: Final = "pure_MI"
MODE_TACTILE_MI: Final = "tactile_MI"
MODE_SELF_LEARNING: Final = "self_learning"
HAPTIC_DEVICE_NAMES: Final = {CLASS_LEFT: "HapticL", CLASS_RIGHT: "HapticR"}
HAPTIC_LEVEL_UUID: Final = "a2300001-4d18-4d1c-86b6-6c4b5d0134a1"
DEFAULT_HAPTIC_INTENSITY: Final = 0.8
HAPTIC_INTENSITY_DIFFERENCE: Final = 0.15
HAPTIC_STEP_SECONDS: Final = 0.25
HAPTIC_TEST_SECONDS: Final = 2.0
HAPTIC_WRITE_TIMEOUT_SECONDS: Final = 1.0
HAPTIC_CONNECT_TIMEOUT_SECONDS: Final = 15.0

EVENT_RECORDING_START: Final = 100
EVENT_TRIAL_START: Final = 110
EVENT_CUE_LEFT: Final = 210
EVENT_CUE_RIGHT: Final = 220
EVENT_CUE_END: Final = 300
EVENT_SELF_LEARNING_START: Final = 230
EVENT_TRIAL_END: Final = 400
EVENT_BREAK_INSTRUCTION_START: Final = 500
EVENT_BREAK_MONITOR_START: Final = 510
EVENT_BREAK_PAUSED: Final = 520
EVENT_BREAK_COUNTDOWN_RESTARTED: Final = 530
EVENT_BREAK_CHIME: Final = 540
EVENT_BREAK_END: Final = 550
EVENT_BREAK_DEVICE_DISCONNECTED: Final = 560
EVENT_BREAK_SIGNAL_RESTORED: Final = 570
EVENT_RECORDING_COMPLETE: Final = 900
EVENT_RECORDING_ABORTED: Final = 990
EVENT_NAMES: Final = {
    EVENT_RECORDING_START: "recording_start",
    EVENT_TRIAL_START: "trial_start_rest",
    EVENT_CUE_LEFT: "cue_left",
    EVENT_CUE_RIGHT: "cue_right",
    EVENT_CUE_END: "cue_end_imagery_black_start",
    EVENT_SELF_LEARNING_START: "self_learning_control_start",
    EVENT_TRIAL_END: "trial_end",
    EVENT_BREAK_INSTRUCTION_START: "break_instruction_start",
    EVENT_BREAK_MONITOR_START: "break_monitor_start",
    EVENT_BREAK_PAUSED: "break_paused",
    EVENT_BREAK_COUNTDOWN_RESTARTED: "break_countdown_restarted",
    EVENT_BREAK_CHIME: "break_chime",
    EVENT_BREAK_END: "break_end",
    EVENT_BREAK_DEVICE_DISCONNECTED: "break_device_disconnected",
    EVENT_BREAK_SIGNAL_RESTORED: "break_signal_restored",
    EVENT_RECORDING_COMPLETE: "recording_complete",
    EVENT_RECORDING_ABORTED: "recording_aborted",
}

# 实时服务
PREVIEW_SECONDS: Final = 5
DISPLAY_LIMIT_UV: Final = 50.0
SAMPLE_BROADCAST_HZ: Final = 15.0
STATE_BROADCAST_HZ: Final = 20.0
SCAN_TIMEOUT_SECONDS: Final = 10.0
RECONNECT_DELAY_SECONDS: Final = 2.0

# 数据存储
DATA_FILENAME: Final = "motor_imagery.h5"
SESSION_README_FILENAME: Final = "README.md"
HDF5_SCHEMA_NAME: Final = "motren_motor_imagery"
HDF5_SCHEMA_VERSION: Final = "1.4"
HDF5_WRITE_BLOCK_SAMPLES: Final = SAMPLE_RATE_HZ


@dataclass(frozen=True)
class EEGStreamConfig:
    """集中定义接收协议；不对厂家数值乘 ADC 系数或做重采样。"""

    host: str = EEG_WS_HOST
    port: int = EEG_WS_PORT
    channels: int = EEG_WS_CHANNELS
    channel_labels: tuple[str, ...] = ()
    timeout_seconds: float = EEG_WS_TIMEOUT_SECONDS
    teacher_channels: tuple[int, ...] = TEACHER_CHANNELS

    def __post_init__(self) -> None:
        if not self.host.strip() or not 0 <= self.port <= 65535:
            raise ValueError("EEG WebSocket 地址或端口无效")
        if type(self.channels) is not int or not 1 <= self.channels <= 256:
            raise ValueError("EEG 通道数必须在 1–256 之间")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("EEG 断流超时必须为有限正数")
        labels = tuple(self.channel_labels) or tuple(f"CH{i + 1}" for i in range(self.channels))
        if len(labels) != self.channels or len(set(labels)) != self.channels or not all(labels):
            raise ValueError("EEG 通道标签必须与通道数一致且不重复")
        object.__setattr__(self, "channel_labels", labels)
        selected = tuple(self.teacher_channels)
        if selected and (
            len(selected) != 8 or len(set(selected)) != 8
            or any(type(i) is not int or not 1 <= i <= self.channels for i in selected)
        ):
            raise ValueError("Teacher 通道映射须为 8 个不重复、从 1 开始的有效采集通道号")
        object.__setattr__(self, "teacher_channels", selected)

    def model_channels(self, *, demo: bool) -> tuple[int, ...]:
        if self.teacher_channels:
            return self.teacher_channels
        if demo and self.channels == 8:
            return tuple(range(1, 9))
        raise ValueError(
            "自主学习需先配置 TEACHER_CHANNELS 或 --teacher-channels："
            "按 Teacher 训练电极顺序指定 8 个采集通道号，不能自动取前 8 通道"
        )

    def metadata(self, *, demo: bool) -> dict:
        return {
            "source": "simulated" if demo else "websocket",
            "host": self.host,
            "port": self.port,
            "message_type": "eeg_data",
            "input_layout": "channels,samples",
            "input_unit": "uV",
            "value_conversion": "none; float32 storage",
            "timeout_seconds": self.timeout_seconds,
            "teacher_channels": list(
                self.teacher_channels or (tuple(range(1, 9)) if demo and self.channels == 8 else ())
            ),
        }
