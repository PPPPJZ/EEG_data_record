"""使用真实本地 WebSocket 验证厂家协议，无硬件、无受试者数据。"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from app import ApplicationService, SessionForm
from config import EEGStreamConfig
from eeg import RealtimeEEGFilter
from load_session import load_session
from self_learning.runtime import FeedbackRuntime
from test_pipeline import wait_until
from ws_eeg import WebSocketEEGSource

# 独立测试配置：本机随机端口，32 通道、250 Hz，不接触真实设备。
CHANNELS = 32
CHUNK = 25
SAMPLE_RATE = 250
TIMEOUT = 0.3


def message(data, *, sample_rate=SAMPLE_RATE):
    return json.dumps({"type": "eeg_data", "data": data.tolist(), "sample_rate": sample_rate})


def address(source):
    port = source.server.sockets[0].getsockname()[1]
    return f"ws://127.0.0.1:{port}"


class DecoderTest(unittest.TestCase):
    def test_bad_shapes_rates_and_values_rejected(self):
        source = WebSocketEEGSource(lambda *_: None, lambda *_: None)
        for data, rate in (
            (np.zeros((8, 5)), 250), (np.zeros((32, 0)), 250),
            (np.zeros((32, 5)), 500), (np.zeros((32, 5)), True),
            (np.full((32, 5), np.nan), 250), (np.full((32, 5), np.inf), 250),
            (np.full((32, 5), "123"), 250), (np.ones((32, 5), dtype=bool), 250),
            (np.full((32, 5), 1e40), 250), (np.zeros(32), 250),
        ):
            with self.subTest(shape=data.shape, rate=rate), self.assertRaises(ValueError):
                source.decode(message(data, sample_rate=rate))
        mixed = np.ones((32, 5)).tolist()
        mixed[0][0] = True
        with self.assertRaises(ValueError):
            source.decode(json.dumps({"type": "eeg_data", "sample_rate": 250, "data": mixed}))
        self.assertIsNone(source.decode('{"type":"status"}'))

    def test_teacher_mapping_and_batched_window(self):
        stream = EEGStreamConfig(teacher_channels=(32, 30, 28, 26, 24, 22, 20, 18))
        self.assertEqual(stream.model_channels(demo=False)[0], 32)
        with self.assertRaisesRegex(ValueError, "通道"):
            EEGStreamConfig().model_channels(demo=False)
        with self.assertRaises(ValueError):
            EEGStreamConfig(teacher_channels=(1,) * 8)
        runtime = FeedbackRuntime("mock", mock=True)
        try:
            runtime.ready = True
            runtime.begin()
            for second in (1.0, 2.0, 3.0):
                with patch("self_learning.runtime.time.perf_counter", return_value=second):
                    runtime.append(np.full((8, 250), second), int(second * 250))
            window = runtime.ring.snapshot()
            self.assertEqual(window.shape, (750, 8))
            np.testing.assert_array_equal(window[:, 0], np.repeat([1, 2, 3], 250))
            with patch("self_learning.runtime.time.perf_counter", return_value=6.0):
                runtime.append(np.zeros((8, 250)), 1000)
            self.assertIsNone(runtime.ring.snapshot())
        finally:
            runtime.executor.shutdown()


class WebSocketSourceTest(unittest.IsolatedAsyncioTestCase):
    async def test_teacher_receives_exact_selected_raw_channels(self):
        selected = (32, 30, 28, 26, 24, 22, 20, 18)
        with tempfile.TemporaryDirectory() as temp:
            service = ApplicationService(
                demo=True, mock_teacher=True,
                stream=EEGStreamConfig(teacher_channels=selected),
            )
            try:
                await service.configure_session(SessionForm(
                    experiment_mode="self_learning", learning_trials=1,
                    subject_id="MAPPING", experiment_date="2026-10-24",
                    output_dir=temp,
                ))
                await service._stop_source()
                await wait_until(lambda: service.feedback.ready)
                service.feedback.begin()
                raw = np.arange(32 * 750, dtype=np.float64).reshape(32, 750)
                service._on_samples(raw, np.zeros_like(raw))
                np.testing.assert_array_equal(
                    service.feedback.ring.snapshot(), raw[[i - 1 for i in selected]].T,
                )
                self.assertEqual(service.state_payload()["eeg"]["teacher_channels"], list(selected))
            finally:
                await service.stop()

    async def test_all_channels_unscaled_display_filter_single_producer_and_reconnect(self):
        chunks, states = [], []
        source = WebSocketEEGSource(
            lambda raw, filtered: chunks.append((raw, filtered)),
            lambda text, ready: states.append((text, ready)),
            config=EEGStreamConfig(port=0, timeout_seconds=1),
        )
        task = asyncio.create_task(source.run())
        try:
            await wait_until(lambda: source.server is not None)
            async with websockets.connect(address(source)) as client:
                self.assertFalse(source.connected)
                rng = np.random.default_rng(8)
                raw = rng.normal(50, 100, (CHANNELS, 300))
                await client.send(message(raw[:, :23]))
                await client.send(message(raw[:, 23:]))
                await wait_until(lambda: len(chunks) == 2)
                np.testing.assert_array_equal(np.hstack([c[0] for c in chunks]), raw)
                np.testing.assert_allclose(
                    np.hstack([c[1] for c in chunks]), RealtimeEEGFilter(channels=32).process(raw),
                )
                async with websockets.connect(address(source)) as second:
                    await second.wait_closed()
                    self.assertEqual(second.close_code, 1013)
                self.assertTrue(source.connected)
            await wait_until(lambda: source._client is None)
            self.assertFalse(source.connected)
            async with websockets.connect(address(source)) as client:
                await client.send(message(raw))
                await wait_until(lambda: len(chunks) == 3)
                np.testing.assert_allclose(chunks[-1][1], RealtimeEEGFilter(channels=32).process(raw))
        finally:
            source.stop()
            await task

    async def test_silent_socket_and_bad_packet_disconnect_without_hiding_failure(self):
        states = []
        source = WebSocketEEGSource(
            lambda *_: None, lambda text, ready: states.append((text, ready)),
            config=EEGStreamConfig(port=0, timeout_seconds=TIMEOUT),
        )
        task = asyncio.create_task(source.run())
        try:
            await wait_until(lambda: source.server is not None)
            async with websockets.connect(address(source)) as client:
                await client.send(message(np.zeros((CHANNELS, CHUNK))))
                await wait_until(lambda: source.connected)
                await client.send('{"type":"status"}')
                await client.wait_closed()
            self.assertFalse(source.connected)
            self.assertTrue(any("未收到有效数据" in text for text, _ in states))
            await wait_until(lambda: source._client is None)
            async with websockets.connect(address(source)) as client:
                await client.send(message(np.zeros((CHANNELS, CHUNK))))
                await wait_until(lambda: source.connected)
                await client.send(message(np.zeros((CHANNELS, CHUNK)), sample_rate=500))
                await client.wait_closed()
            self.assertFalse(source.connected)
            self.assertEqual(source.malformed_packets, 1)
            self.assertTrue(any("250 Hz" in text and not ready for text, ready in states))
        finally:
            source.stop()
            await task

    async def test_complete_real_protocol_stores_only_received_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            service = ApplicationService(stream=EEGStreamConfig(port=0))
            recorded = []
            sender = None
            await service.start()
            try:
                await service.configure_session(SessionForm(
                    subject_id="WS-TEST", experiment_date="2026-10-24",
                    left_trials=1, right_trials=0, output_dir=temp,
                ))
                source = service.source
                await wait_until(lambda: source.server is not None)
                callback = source.on_samples

                def capture(raw, filtered):
                    if service._recording_active:
                        recorded.append(raw.copy())
                    callback(raw, filtered)

                source.on_samples = capture
                async with websockets.connect(address(source)) as client:
                    async def send():
                        index = 0
                        while True:
                            t = (index + np.arange(CHUNK)) / SAMPLE_RATE
                            raw = np.arange(CHANNELS)[:, None] + 80 * np.sin(2 * np.pi * 50 * t)
                            await client.send(message(raw))
                            index += CHUNK
                            await asyncio.sleep(CHUNK / SAMPLE_RATE)

                    sender = asyncio.create_task(send())
                    await wait_until(lambda: service.state_payload()["can_start"])
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == "complete", timeout=18)
                    sender.cancel()
                    with suppress(asyncio.CancelledError):
                        await sender
                session = load_session(Path(temp) / "motor_imagery.h5")
                self.assertEqual(session["status"], "complete")
                self.assertEqual(session["protocol"]["trial_seconds"], 8)
                self.assertEqual(session["protocol"]["channel_count"], CHANNELS)
                np.testing.assert_array_equal(
                    session["raw_uv"], np.hstack(recorded).T.astype(np.float32),
                )
                self.assertIsNone(session["filtered_uv"])
                self.assertTrue(session["trials"]["completed"].all())
                with h5py.File(Path(temp) / "motor_imagery.h5") as handle:
                    self.assertEqual(set(handle["eeg"]), {"raw_uv"})
            finally:
                if sender:
                    sender.cancel()
                    with suppress(asyncio.CancelledError, ConnectionClosed):
                        await sender
                await service.stop()

    async def test_stream_loss_rolls_back_trial_and_allows_raw_only_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            service = ApplicationService(stream=EEGStreamConfig(port=0, timeout_seconds=TIMEOUT))
            sender = None
            try:
                await service.configure_session(SessionForm(
                    subject_id="WS-LOSS", experiment_date="2026-10-24",
                    left_trials=2, right_trials=0, output_dir=temp,
                ))
                await wait_until(lambda: service.source.server is not None)
                async with websockets.connect(address(service.source)) as client:
                    async def send():
                        while True:
                            await client.send(message(np.ones((CHANNELS, CHUNK))))
                            await asyncio.sleep(CHUNK / SAMPLE_RATE)
                    sender = asyncio.create_task(send())
                    await wait_until(lambda: service.state_payload()["can_start"])
                    await service.start_experiment()
                    await wait_until(lambda: service.writer.sample_count >= CHUNK)
                    sender.cancel()
                    with suppress(asyncio.CancelledError):
                        await sender
                    await wait_until(lambda: service.last_outcome == "incomplete")
                session = load_session(Path(temp) / "motor_imagery.h5")
                self.assertEqual(session["raw_uv"].shape, (0, CHANNELS))
                self.assertIsNone(session["filtered_uv"])
                self.assertIn("未收到有效数据", session["metadata"]["termination_reason"])
                info = await service.parse_restart(temp)
                self.assertEqual(info["next_trial_id"], 1)
                await service.configure_restart()
                await wait_until(lambda: service.source.server is not None)
                async with websockets.connect(address(service.source)) as client:
                    await client.send(message(np.ones((CHANNELS, CHUNK))))
                    await wait_until(lambda: service.state_payload()["can_start"])
                    await service.start_experiment()
                    await service.abort_experiment("cancel preparation")
            finally:
                if sender:
                    sender.cancel()
                    with suppress(asyncio.CancelledError, ConnectionClosed):
                        await sender
                await service.stop()


if __name__ == "__main__":
    unittest.main()
