"""接收厂家 ws_server_example.py 协议，复用原显示滤波器。"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from config import (
    EEGStreamConfig, EEG_WS_MAX_CHUNK_SAMPLES, EEG_WS_MAX_MESSAGE_BYTES,
    RECONNECT_DELAY_SECONDS, SAMPLE_RATE_HZ,
)
from eeg import RealtimeEEGFilter, SamplesCallback, StatusCallback


class WebSocketEEGSource:
    """单一推流客户端；有效 EEG 才算连接，断流沿用实验断连处理。"""

    def __init__(
        self, on_samples: SamplesCallback, on_status: StatusCallback,
        *, config: EEGStreamConfig | None = None,
    ) -> None:
        self.config = config or EEGStreamConfig()
        self.on_samples, self.on_status = on_samples, on_status
        self.filter = RealtimeEEGFilter(channels=self.config.channels)
        self.malformed_packets = 0
        self.running = True
        self.connected = False
        self.status = "等待启动 EEG 接收"
        self.server = None
        self._client = None
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self.running = False
        self._stop.set()

    def _status(self, text: str, connected: bool) -> None:
        self.status = text
        self.connected = connected
        self.on_status(text, connected)

    def decode(self, message: str | bytes) -> np.ndarray | None:
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise ValueError("消息必须为 JSON 对象")
        if payload.get("type") != "eeg_data":
            return None
        rate = payload.get("sample_rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate != SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate 必须为 {SAMPLE_RATE_HZ} Hz，收到 {rate!r}")
        # 先检查类型，避免把字符串、null 或布尔值隐式变成 EEG 数值。
        raw = np.asarray(payload.get("data"))
        if raw.ndim != 2 or raw.shape[0] != self.config.channels:
            raise ValueError(f"data 必须为 ({self.config.channels}, samples)，收到 {raw.shape}")
        if not 1 <= raw.shape[1] <= EEG_WS_MAX_CHUNK_SAMPLES:
            raise ValueError(f"每块必须包含 1–{EEG_WS_MAX_CHUNK_SAMPLES} 个采样点")
        if raw.dtype.kind not in "iuf" or any(
            type(value) is bool for row in payload["data"] for value in row
        ):
            raise ValueError("EEG 只能包含数值")
        raw = raw.astype(np.float64)
        if not np.isfinite(raw).all() or np.any(np.abs(raw) > np.finfo(np.float32).max):
            raise ValueError("EEG 包含非有限值或超出 float32 存储范围")
        return raw

    async def _handle(self, websocket, _path=None) -> None:
        # 第二个客户端不能覆盖状态或把另一台设备的数据混入同一文件。
        if self._client is not None:
            await websocket.close(code=1013, reason="An EEG producer is already connected")
            return
        self._client = websocket
        self.filter.reset()
        self._status("厂家已连接，等待有效 EEG 数据", False)
        loop = asyncio.get_running_loop()
        last_valid = loop.time()
        last_error = "EEG 推流连接已断开，等待厂家重新连接"
        try:
            while self.running:
                remaining = self.config.timeout_seconds - (loop.time() - last_valid)
                if remaining <= 0:
                    raise asyncio.TimeoutError
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                try:
                    raw = self.decode(message)
                except (ValueError, TypeError, OverflowError) as exc:
                    self.malformed_packets += 1
                    # 一旦丢弃坏 EEG 块便形成缺口，不继续把 trial 标记为完整。
                    last_error = f"EEG 数据协议错误：{exc}；等待厂家重新连接"
                    self._status(last_error, False)
                    await websocket.close(code=1008, reason="Invalid EEG data")
                    return
                if raw is None:
                    continue  # 非 EEG 消息不会延长有效信号超时。
                filtered = self.filter.process(raw)
                last_valid = loop.time()
                if not self.connected:
                    self._status(
                        f"已接收厂家 EEG · {self.config.channels} 通道 · {SAMPLE_RATE_HZ} Hz", True,
                    )
                self.on_samples(raw, filtered)
        except asyncio.TimeoutError:
            last_error = f"EEG 超过 {self.config.timeout_seconds:g} 秒未收到有效数据，等待厂家重新连接"
            self._status(last_error, False)
            await websocket.close(code=1011, reason="EEG stream timeout")
        except ConnectionClosed:
            pass
        finally:
            self._client = None
            self._status(last_error if self.running else "EEG 接收服务已停止", False)

    async def run(self) -> None:
        address = f"ws://{self.config.host}:{self.config.port}"
        while self.running:
            try:
                async with websockets.serve(
                    self._handle, self.config.host, self.config.port,
                    max_size=EEG_WS_MAX_MESSAGE_BYTES, max_queue=16,
                    close_timeout=1,
                ) as server:
                    self.server = server
                    self._status(f"正在监听 {address}，等待厂家推送 EEG", False)
                    await self._stop.wait()
            except asyncio.CancelledError:
                raise
            except OSError as exc:
                self._status(f"EEG WebSocket 监听失败 {address}：{exc}", False)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=RECONNECT_DELAY_SECONDS)
                except asyncio.TimeoutError:
                    pass
            finally:
                self.server = None
