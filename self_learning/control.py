"""独立的光标控制数学与固定长度原始 EEG 环形缓冲。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

WINDOW_SAMPLES = 750
CHANNELS = 8
DEFAULT_CONFIG_PATH = Path(__file__).with_name("control.yaml")


@dataclass(frozen=True)
class ControlConfig:
    inference_interval_ms: float = 40
    ema_alpha: float = 0.2
    dead_zone: float = 0.1
    max_velocity: float = 0.2
    cursor_min_x: float = 0.05
    cursor_max_x: float = 0.95

    def __post_init__(self):
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in asdict(self).values()
        ):
            raise ValueError("光标参数必须是有限数值")
        if self.inference_interval_ms <= 0 or not 0 < self.ema_alpha <= 1:
            raise ValueError("推理间隔必须为正数，ema_alpha 必须在 (0, 1]")
        if not 0 <= self.dead_zone < 1 or self.max_velocity <= 0:
            raise ValueError("dead_zone 必须在 [0, 1)，max_velocity 必须为正数")
        if not 0 <= self.cursor_min_x <= 0.5 <= self.cursor_max_x <= 1:
            raise ValueError("光标边界必须在 [0, 1] 内且包含中心 0.5")
        if self.cursor_min_x == self.cursor_max_x:
            raise ValueError("光标左右边界不可相同")

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "ControlConfig":
        try:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"control.yaml 语法错误：{exc}") from exc
        if not isinstance(values, dict):
            raise ValueError("control.yaml 必须是参数映射")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"control.yaml 包含未知参数：{exc}") from exc


class RawRingBuffer:
    """采集端 (8, N) -> Predictor 的 (750, 8)，始终复制完整快照。"""
    def __init__(self):
        self.data = np.empty((WINDOW_SAMPLES, CHANNELS), dtype=np.float32)
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0
        self.position = 0

    def append(self, raw_uv: np.ndarray):
        raw = np.asarray(raw_uv, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != CHANNELS or not np.isfinite(raw).all():
            raise ValueError("原始 EEG 必须为有限的 (8, N) 微伏数组")
        n = raw.shape[1]
        self.total += n
        if n >= WINDOW_SAMPLES:
            self.data[:] = raw[:, -WINDOW_SAMPLES:].T
            self.position = 0
            self.count = WINDOW_SAMPLES
            return
        first = min(n, WINDOW_SAMPLES - self.position)
        self.data[self.position:self.position + first] = raw[:, :first].T
        if n > first:
            self.data[:n - first] = raw[:, first:].T
        self.position = (self.position + n) % WINDOW_SAMPLES
        self.count = min(WINDOW_SAMPLES, self.count + n)

    def snapshot(self) -> np.ndarray | None:
        if self.count < WINDOW_SAMPLES:
            return None
        return np.concatenate((self.data[self.position:], self.data[:self.position]))


class CursorController:
    """按实际时间积分；新概率到来前持续使用上次速度。"""
    def __init__(self, config: ControlConfig):
        self.config = config
        self.reset()

    def reset(self, now: float | None = None):
        self.x = 0.5
        self.smooth = 0.0
        self.signal = 0.0
        self.control = 0.0
        self.velocity = 0.0
        self.last_time = now

    def advance(self, now: float) -> float:
        if self.last_time is not None:
            dt = max(0.0, now - self.last_time)
            self.x = float(np.clip(
                self.x + self.velocity * dt,
                self.config.cursor_min_x, self.config.cursor_max_x,
            ))
        self.last_time = now
        return self.x

    def update(self, p_right: float, now: float):
        if not math.isfinite(p_right) or not 0 <= p_right <= 1:
            raise ValueError("P(Right) 必须在 [0, 1]")
        self.advance(now)
        self.signal = 2 * p_right - 1
        a = self.config.ema_alpha
        self.smooth = a * self.signal + (1 - a) * self.smooth
        magnitude = max(0.0, abs(self.smooth) - self.config.dead_zone)
        self.control = math.copysign(magnitude / (1 - self.config.dead_zone), self.smooth)
        self.velocity = self.config.max_velocity * self.control

    def freeze(self, now: float):
        self.advance(now)
        self.velocity = 0.0
