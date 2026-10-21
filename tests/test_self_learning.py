"""自主学习控制、原始窗口和采集流程验证，不依赖真实 BLE 外设。"""
from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from app import ApplicationService, SessionForm
from config import (
    BREAK_COUNTDOWN_SECONDS,
    BREAK_INSTRUCTION_SECONDS,
    EVENT_CUE_LEFT,
    EVENT_CUE_RIGHT,
    EVENT_SELF_LEARNING_START,
    SELF_LEARNING_TRIAL_SECONDS,
    TRIAL_SECONDS,
)
from load_session import load_session
from self_learning.control import ControlConfig, CursorController, RawRingBuffer
from self_learning.runtime import FeedbackRuntime


async def wait_until(predicate, timeout=10):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("未达到预期状态")
        await asyncio.sleep(0.01)


class ControlTests(unittest.TestCase):
    def test_direction_ema_dead_zone_and_velocity_integration(self):
        cursor = CursorController(ControlConfig())
        cursor.reset(0)
        cursor.update(1, 0)
        self.assertAlmostEqual(cursor.smooth, 0.2)
        self.assertAlmostEqual(cursor.velocity, 0.2 * (0.2 - 0.1) / 0.9)
        cursor.advance(0.5)
        self.assertAlmostEqual(cursor.x, 0.5 + cursor.velocity * 0.5)
        cursor.reset(0)
        cursor.update(0, 0)
        self.assertLess(cursor.velocity, 0)
        cursor.advance(1)
        self.assertLess(cursor.x, 0.5)
        cursor.reset(0)
        cursor.update(0.7, 0)
        self.assertEqual(cursor.velocity, 0)  # EMA=0.08 在死区内
        cursor.advance(5)
        self.assertEqual(cursor.x, 0.5)

    def test_actual_dt_freeze_reset_and_boundaries(self):
        cursor = CursorController(ControlConfig(ema_alpha=1))
        cursor.reset(2)
        cursor.update(1, 2)
        cursor.advance(2.17)
        self.assertAlmostEqual(cursor.x, 0.534)
        cursor.advance(100)
        self.assertEqual(cursor.x, 0.95)
        cursor.update(0, 100)
        cursor.advance(200)
        self.assertEqual(cursor.x, 0.05)
        cursor.freeze(200)
        cursor.advance(300)
        self.assertEqual(cursor.x, 0.05)
        cursor.reset()
        self.assertEqual(cursor.x, 0.5)
        self.assertEqual(cursor.smooth, 0)

    def test_ring_keeps_exactly_latest_raw_window_and_copies(self):
        ring = RawRingBuffer()
        raw = np.arange(8 * 1200, dtype=np.float32).reshape(8, 1200)
        ring.append(raw[:, :749])
        self.assertIsNone(ring.snapshot())
        ring.append(raw[:, 749:755])
        np.testing.assert_array_equal(ring.snapshot(), raw[:, 5:755].T)
        snapshot = ring.snapshot()
        ring.append(raw[:, 755:])
        np.testing.assert_array_equal(ring.snapshot(), raw[:, -750:].T)
        np.testing.assert_array_equal(snapshot, raw[:, 5:755].T)
        ring.reset()
        self.assertIsNone(ring.snapshot())
        ring.append(raw)
        np.testing.assert_array_equal(ring.snapshot(), raw[:, -750:].T)

    def test_invalid_yaml_parameters_and_raw_values_rejected(self):
        for values in [
            {"ema_alpha": 0}, {"dead_zone": 1}, {"inference_interval_ms": -1},
            {"max_velocity": float("nan")}, {"cursor_min_x": 0.6},
        ]:
            with self.assertRaises(ValueError):
                ControlConfig(**values)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "control.yaml"
            path.write_text("ema_alpha: [", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "语法错误"):
                ControlConfig.load(path)
        with self.assertRaises(ValueError):
            RawRingBuffer().append(np.full((8, 10), np.nan))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_frequency_raw_input_and_stale_result_reset(self):
        rows = []
        runtime = FeedbackRuntime("", mock=True, on_prediction=rows.append)
        await runtime.start()
        try:
            await wait_until(lambda: runtime.ready)
            calls = []

            class SlowPredictor:
                def predict_details(self, window):
                    calls.append((time.perf_counter(), window.copy()))
                    time.sleep(0.01)
                    from types import SimpleNamespace
                    return SimpleNamespace(probabilities=(0.1, 0.9), class_id=1)

            runtime.predictor = SlowPredictor()
            runtime.begin()
            raw = np.arange(8 * 750, dtype=np.float32).reshape(8, 750)
            runtime.append(raw[:, :749], 749)
            await asyncio.sleep(0.06)
            self.assertEqual(len(rows), 0)
            runtime.append(raw[:, 749:], 750)
            await wait_until(lambda: len(rows) == 1)
            np.testing.assert_array_equal(calls[0][1], raw.T)
            for _ in range(6):
                runtime.append(np.ones((8, 10)), runtime.sample_end + 10)
                await asyncio.sleep(0.045)
            self.assertGreaterEqual(len(rows), 5)
            intervals = np.diff([r[0] for r in calls])
            self.assertGreater(float(np.median(intervals)), 0.025)
            self.assertLess(float(np.median(intervals)), 0.09)
            runtime.begin(trial_id=2, context="trial")
            await asyncio.sleep(0.06)
            self.assertEqual(runtime.prediction_count, 0)
            self.assertEqual(runtime.cursor.x, 0.5)
            self.assertIsNone(runtime.ring.snapshot())
        finally:
            await runtime.close()

    async def test_initialization_failure_keeps_model_red(self):
        runtime = FeedbackRuntime("does-not-exist.pt")
        await runtime.start()
        try:
            await wait_until(lambda: bool(runtime.error), timeout=20)
            self.assertFalse(runtime.ready)
            with self.assertRaises(ValueError):
                runtime.begin()
        finally:
            await runtime.close()

    async def test_inflight_prediction_is_discarded_after_new_trial(self):
        from threading import Event
        from types import SimpleNamespace
        started, release = Event(), Event()
        rows = []
        runtime = FeedbackRuntime("", mock=True, on_prediction=rows.append)
        await runtime.start()
        try:
            await wait_until(lambda: runtime.ready)

            class BlockingPredictor:
                def predict_details(self, window):
                    started.set()
                    release.wait(timeout=2)
                    return SimpleNamespace(probabilities=(0, 1), class_id=1)

            runtime.predictor = BlockingPredictor()
            runtime.begin(trial_id=1, context="trial")
            runtime.append(np.ones((8, 750)), 750)
            await wait_until(started.is_set)
            runtime.begin(trial_id=2, context="trial")
            release.set()
            await asyncio.sleep(0.1)
            self.assertEqual(rows, [])
            self.assertIsNone(runtime.last_prediction)
            self.assertEqual(runtime.cursor.x, 0.5)
        finally:
            release.set()
            await runtime.close()

    async def test_mock_session_persists_unknown_labels_and_only_imagery_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "learning"
            service = ApplicationService(demo=True, mock_teacher=True, time_scale=0.1)
            await service.start()
            try:
                await service.configure_session(SessionForm(
                    experiment_mode="self_learning", subject_id="S-LEARNING",
                    experiment_date="2026-10-20", learning_trials=2, break_every_trials=1,
                    left_trials=0, right_trials=0, teacher_model_path="mock",
                    output_dir=str(output),
                ))
                await wait_until(lambda: service.state_payload()["can_start"])
                timing = service.state_payload()["protocol_timing"]
                self.assertEqual(timing["standard_trial_seconds"], TRIAL_SECONDS)
                self.assertEqual(
                    timing["self_learning_trial_seconds"],
                    SELF_LEARNING_TRIAL_SECONDS,
                )
                self.assertEqual(
                    timing["interblock_break_seconds"],
                    BREAK_INSTRUCTION_SECONDS + BREAK_COUNTDOWN_SECONDS,
                )
                self.assertIsNone(service.haptics)
                self.assertEqual(service.sequence, [-1, -1])
                await service.test_model()
                await wait_until(lambda: service.feedback.test_result is not None)
                self.assertEqual(service.feedback.test_result["count"], 10)
                await service.preview_feedback("start")
                await wait_until(lambda: service.feedback.prediction_count > 0)
                await service.preview_feedback("reset")
                self.assertEqual(service.feedback.cursor.x, 0.5)
                await service.preview_feedback("stop")
                await service.start_experiment()
                await wait_until(lambda: service.stage == "break_monitor")
                samples = service.writer.sample_count
                await service.toggle_break_countdown()
                await asyncio.sleep(0.1)
                self.assertEqual(service.writer.sample_count, samples)
                self.assertFalse(service.feedback.enabled)
                await service.toggle_break_countdown()
                await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()
            session = load_session(output / "motor_imagery.h5")
            self.assertEqual(session["status"], "complete")
            self.assertEqual(session["trials"]["class_id"].tolist(), [-1, -1])
            self.assertEqual(
                session["protocol"]["trial_seconds"],
                SELF_LEARNING_TRIAL_SECONDS,
            )
            self.assertEqual(session["raw_uv"].shape, session["filtered_uv"].shape)
            codes = session["events"]["event_code"].tolist()
            self.assertEqual(codes.count(EVENT_SELF_LEARNING_START), 2)
            self.assertNotIn(EVENT_CUE_LEFT, codes)
            self.assertNotIn(EVENT_CUE_RIGHT, codes)
            self.assertTrue(session["feedback_config"]["mock"])
            for trial in session["trials"]:
                feedback = session["feedback"][session["feedback"]["trial_id"] == trial["trial_id"]]
                self.assertGreater(len(feedback), 0)
                self.assertTrue(np.all(feedback["window_start_sample"] >= trial["cue_start_sample"]))
                self.assertTrue(np.all(feedback["window_end_sample"] <= trial["trial_end_sample"]))
                self.assertTrue(np.all(feedback["window_end_sample"] - feedback["window_start_sample"] == 750))
            self.assertIn("class_id=-1", (output / "README.md").read_text(encoding="utf-8"))

    async def test_disconnect_removes_incomplete_learning_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = ApplicationService(demo=True, mock_teacher=True, time_scale=0.1)
            await service.start()
            try:
                await service.configure_session(SessionForm(
                    experiment_mode="self_learning", subject_id="S-ABORT",
                    experiment_date="2026-10-20", learning_trials=2,
                    output_dir=temporary,
                ))
                await wait_until(lambda: service.state_payload()["can_start"])
                await service.start_experiment()
                await wait_until(lambda: service.stage == "self_learning")
                service._on_source_status("test disconnect", False)
                await wait_until(lambda: service.stage == "device_check")
                self.assertFalse(service.feedback.enabled)
                self.assertTrue(service.session_locked)
            finally:
                await service.stop()
            session = load_session(Path(temporary) / "motor_imagery.h5")
            self.assertEqual(session["status"], "incomplete")
            self.assertEqual(len(session["raw_uv"]), 0)
            self.assertEqual(len(session["feedback"]), 0)
            self.assertEqual(session["trials"]["rest_start_sample"].tolist(), [-1, -1])
            self.assertEqual(session["trials"]["class_id"].tolist(), [-1, -1])
