"""Motren_E 的 BLE 数据解码、实时滤波和模拟数据源。"""

from __future__ import annotations

import asyncio
import struct
from collections import deque
from collections.abc import Callable
from contextlib import suppress

import numpy as np
from bleak import BleakClient, BleakScanner
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi

from config import (
    ADC_UV_PER_LSB,
    BANDPASS_CASCADE_COUNT,
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    CHANNEL_COUNT,
    CONTROL_UUID,
    EEG_DATA_UUID,
    NOTCH_HZ,
    NOTCH_Q,
    RECONNECT_DELAY_SECONDS,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_PACKET,
    SCAN_TIMEOUT_SECONDS,
    START_ACQUISITION,
    STOP_ACQUISITION,
    TARGET_DEVICE_NAME,
)

BLOCK_STRUCT = struct.Struct("<9i")
PACKET_SIZE = 1 + SAMPLES_PER_PACKET * BLOCK_STRUCT.size
FLAG_PACKET_END = 0x01
FLAG_HAS_EEG_DATA = 0x02

SamplesCallback = Callable[[np.ndarray, np.ndarray], None]
StatusCallback = Callable[[str, bool], None]


class PacketDecodeError(ValueError):
    """BLE 通知不符合设备协议。"""


class EEGPacketDecoder:
    """将一条完整设备通知解码为 8 x 5 的原始微伏数据。"""

    def __init__(self) -> None:
        self._pending = [
            deque(maxlen=SAMPLES_PER_PACKET) for _ in range(CHANNEL_COUNT)
        ]

    def reset(self) -> None:
        for channel in self._pending:
            channel.clear()

    def feed(self, data: bytes | bytearray) -> np.ndarray | None:
        if not data:
            raise PacketDecodeError("empty BLE notification")

        flags = data[0]
        if flags & FLAG_HAS_EEG_DATA:
            if len(data) != PACKET_SIZE:
                raise PacketDecodeError(
                    f"bad EEG packet length: got {len(data)}, expected {PACKET_SIZE}"
                )
            payload = memoryview(data)[1:]
            for sample_index in range(SAMPLES_PER_PACKET):
                block = BLOCK_STRUCT.unpack_from(
                    payload, sample_index * BLOCK_STRUCT.size
                )
                for channel_index, digital_value in enumerate(block[1:]):
                    self._pending[channel_index].append(
                        digital_value * ADC_UV_PER_LSB
                    )
        elif len(data) != 1:
            raise PacketDecodeError("notification has data but no EEG-data flag")

        if not flags & FLAG_PACKET_END:
            return None

        if {len(channel) for channel in self._pending} != {SAMPLES_PER_PACKET}:
            self.reset()
            raise PacketDecodeError("packet-end flag arrived without five samples")

        decoded = np.asarray(self._pending, dtype=np.float64)
        self.reset()
        return decoded


class RealtimeEEGFilter:
    """50 Hz 陷波后串联两次 5-30 Hz 四阶 Butterworth 带通。"""

    def __init__(self, channels: int = CHANNEL_COUNT, fs: int = SAMPLE_RATE_HZ):
        self.channels = channels
        self.notch_b, self.notch_a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=fs)
        single_bandpass_sos = butter(
            BANDPASS_ORDER,
            [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        self.bandpass_sos = np.vstack(
            [single_bandpass_sos] * BANDPASS_CASCADE_COUNT
        )
        self.reset()

    def reset(self) -> None:
        self._notch_state: list[np.ndarray | None] = [None] * self.channels
        self._bandpass_state: list[np.ndarray | None] = [None] * self.channels

    def process(self, samples_uv: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples_uv, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != self.channels:
            raise ValueError(
                f"expected ({self.channels}, samples), got {samples.shape}"
            )
        if samples.shape[1] == 0:
            return samples.copy()

        filtered = np.empty_like(samples)
        for channel_index, channel in enumerate(samples):
            notch_state = self._notch_state[channel_index]
            if notch_state is None:
                notch_state = lfilter_zi(self.notch_b, self.notch_a) * channel[0]
            notched, notch_state = lfilter(
                self.notch_b, self.notch_a, channel, zi=notch_state
            )
            self._notch_state[channel_index] = notch_state

            bandpass_state = self._bandpass_state[channel_index]
            if bandpass_state is None:
                bandpass_state = sosfilt_zi(self.bandpass_sos) * notched[0]
            bandpassed, bandpass_state = sosfilt(
                self.bandpass_sos, notched, zi=bandpass_state
            )
            self._bandpass_state[channel_index] = bandpass_state
            filtered[channel_index] = bandpassed
        return filtered


class BLEEEGSource:
    """自动重连的 Motren_E 数据源，同时输出原始和滤波数据。"""

    def __init__(
        self,
        on_samples: SamplesCallback,
        on_status: StatusCallback,
        *,
        scan_lock: asyncio.Lock | None = None,
    ) -> None:
        self.on_samples = on_samples
        self.on_status = on_status
        self.decoder = EEGPacketDecoder()
        self.filter = RealtimeEEGFilter()
        self.running = True
        self.malformed_packets = 0
        self._scan_lock = scan_lock if scan_lock is not None else asyncio.Lock()

    def stop(self) -> None:
        self.running = False

    @staticmethod
    def _matches_target(device, advertisement_data) -> bool:
        advertised_name = getattr(advertisement_data, "local_name", None)
        return TARGET_DEVICE_NAME in {device.name, advertised_name}

    def _on_notification(self, _sender, data: bytearray) -> None:
        try:
            raw_uv = self.decoder.feed(data)
            if raw_uv is not None:
                self.on_samples(raw_uv, self.filter.process(raw_uv))
        except (PacketDecodeError, ValueError, struct.error):
            self.malformed_packets += 1

    async def _find_device(self):
        self.on_status(f"正在扫描 {TARGET_DEVICE_NAME}", False)
        async with self._scan_lock:
            return await BleakScanner.find_device_by_filter(
                self._matches_target,
                timeout=SCAN_TIMEOUT_SECONDS,
            )

    async def _collect(self, device) -> None:
        disconnected = asyncio.Event()
        loop = asyncio.get_running_loop()

        def on_disconnected(_client) -> None:
            loop.call_soon_threadsafe(disconnected.set)

        self.on_status(f"正在连接 {TARGET_DEVICE_NAME}", False)
        async with BleakClient(
            device,
            disconnected_callback=on_disconnected,
            timeout=30.0,
        ) as client:
            if client.services.get_characteristic(EEG_DATA_UUID) is None:
                raise RuntimeError("EEG characteristic is unavailable")
            if client.services.get_characteristic(CONTROL_UUID) is None:
                raise RuntimeError("control characteristic is unavailable")

            self.decoder.reset()
            self.filter.reset()
            await client.start_notify(EEG_DATA_UUID, self._on_notification)
            await client.write_gatt_char(
                CONTROL_UUID, START_ACQUISITION, response=False
            )
            self.on_status(f"已连接 {TARGET_DEVICE_NAME}", True)

            try:
                while self.running and client.is_connected:
                    try:
                        await asyncio.wait_for(disconnected.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    break
            finally:
                if client.is_connected:
                    with suppress(Exception):
                        await client.write_gatt_char(
                            CONTROL_UUID, STOP_ACQUISITION, response=False
                        )
                    with suppress(Exception):
                        await client.stop_notify(EEG_DATA_UUID)

    async def run(self) -> None:
        while self.running:
            try:
                device = await self._find_device()
                if device is None:
                    self.on_status(
                        f"未发现 {TARGET_DEVICE_NAME}，正在重试", False
                    )
                else:
                    await self._collect(device)
                    if self.running:
                        self.on_status("蓝牙已断开，正在自动重连", False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 采集服务必须持续重连
                self.on_status(f"蓝牙错误：{exc}", False)

            if self.running:
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)


class SimulatedEEGSource:
    """用于无设备端到端测试的 8 通道模拟源。"""

    def __init__(
        self,
        on_samples: SamplesCallback,
        on_status: StatusCallback,
        *,
        time_scale: float = 1.0,
        seed: int = 7,
    ) -> None:
        self.on_samples = on_samples
        self.on_status = on_status
        self.time_scale = time_scale
        self.running = True
        self.filter = RealtimeEEGFilter()
        self.rng = np.random.default_rng(seed)
        self.sample_index = 0
        self.phases = np.linspace(0.0, np.pi, CHANNEL_COUNT, endpoint=False)
        self.malformed_packets = 0

    def stop(self) -> None:
        self.running = False

    async def run(self) -> None:
        self.on_status("已连接模拟 Motren_E", True)
        loop = asyncio.get_running_loop()
        last_update = loop.time()
        sample_budget = 0.0

        while self.running:
            now = loop.time()
            sample_budget += (
                (now - last_update) * SAMPLE_RATE_HZ / self.time_scale
            )
            last_update = now
            chunk_size = (
                int(sample_budget) // SAMPLES_PER_PACKET * SAMPLES_PER_PACKET
            )
            if chunk_size < SAMPLES_PER_PACKET:
                await asyncio.sleep(0.002)
                continue
            sample_budget -= chunk_size

            indices = np.arange(self.sample_index, self.sample_index + chunk_size)
            time_seconds = indices / SAMPLE_RATE_HZ
            raw_uv = np.empty((CHANNEL_COUNT, chunk_size), dtype=np.float64)
            for channel in range(CHANNEL_COUNT):
                phase = self.phases[channel]
                mu = (14.0 + channel) * np.sin(
                    2 * np.pi * (10.0 + channel * 0.07) * time_seconds + phase
                )
                background = 8.0 * np.sin(
                    2 * np.pi * 6.0 * time_seconds + phase / 2
                )
                line_noise = 15.0 * np.sin(2 * np.pi * 50.0 * time_seconds)
                noise = self.rng.normal(0.0, 4.0, chunk_size)
                raw_uv[channel] = mu + background + line_noise + noise

            self.sample_index += chunk_size
            self.on_samples(raw_uv, self.filter.process(raw_uv))
            await asyncio.sleep(0)
