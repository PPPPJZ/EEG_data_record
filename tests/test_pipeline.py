from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import ApplicationService, SessionForm
from config import (
    EVENT_BREAK_CHIME,
    EVENT_BREAK_COUNTDOWN_RESTARTED,
    EVENT_BREAK_DEVICE_DISCONNECTED,
    EVENT_BREAK_END,
    EVENT_BREAK_INSTRUCTION_START,
    EVENT_BREAK_MONITOR_START,
    EVENT_BREAK_PAUSED,
    EVENT_BREAK_SIGNAL_RESTORED,
    EVENT_CUE_LEFT,
    EVENT_RECORDING_COMPLETE,
    EVENT_RECORDING_START,
)
from load_session import load_session


async def wait_until(predicate, *, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0.01)


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_interblock_break_pauses_writes_and_continue_resets_timer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "break_session"
            # Windows 默认定时粒度约 15.6 ms，避免把整段实验压缩至该量级。
            service = ApplicationService(demo=True, time_scale=0.1)
            await service.start()
            try:
                await service.configure_session(
                    SessionForm(
                        subject_id="S-BREAK",
                        experiment_date="2026-09-21",
                        description="interblock break test",
                        left_trials=2,
                        right_trials=0,
                        break_every_trials=1,
                        output_dir=str(output_dir),
                    )
                )
                await wait_until(lambda: service.device_connected)
                await service.start_experiment()
                await wait_until(lambda: service.stage == "break_monitor")

                samples_at_break = service.writer.sample_count
                await asyncio.sleep(0.05)
                self.assertEqual(service.writer.sample_count, samples_at_break)
                self.assertLess(service._break_remaining_s, 30.0)

                await service.toggle_break_countdown()
                paused_remaining = service._break_remaining_s
                await asyncio.sleep(0.05)
                self.assertAlmostEqual(
                    service._break_remaining_s,
                    paused_remaining,
                    places=3,
                )

                await service.toggle_break_countdown()
                self.assertGreater(service._break_remaining_s, 29.0)

                service._on_source_status("test disconnect", False)
                self.assertEqual(service.stage, "break_monitor")
                self.assertTrue(service._break_paused)
                service._on_source_status("test reconnect", True)
                await wait_until(lambda: service._has_valid_signal)
                self.assertTrue(service._break_paused)
                self.assertTrue(service.state_payload()["break_can_continue"])
                await service.toggle_break_countdown()
                self.assertGreater(service._break_remaining_s, 29.0)
                await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()

            session = load_session(output_dir / "motor_imagery.h5")
            event_codes = session["events"]["event_code"].tolist()
            break_codes = {
                EVENT_BREAK_INSTRUCTION_START,
                EVENT_BREAK_MONITOR_START,
                EVENT_BREAK_PAUSED,
                EVENT_BREAK_COUNTDOWN_RESTARTED,
                EVENT_BREAK_CHIME,
                EVENT_BREAK_END,
                EVENT_BREAK_DEVICE_DISCONNECTED,
                EVENT_BREAK_SIGNAL_RESTORED,
            }
            break_events = session["events"][
                np.isin(session["events"]["event_code"], list(break_codes))
            ]

            self.assertTrue(break_codes.issubset(set(event_codes)))
            self.assertEqual(len(set(break_events["sample_index"].tolist())), 1)
            self.assertEqual(session["trials"]["block_id"].tolist(), [1, 2])
            self.assertGreater(session["raw_uv"].shape[0], 3_600)
            self.assertLess(session["raw_uv"].shape[0], 4_400)
            self.assertGreater(
                session["quality"]["wall_duration_s"],
                session["quality"]["actual_duration_s"] + 30.0,
            )

    async def test_abort_during_preparation_leaves_output_directory_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "preparation_cancelled"
            service = ApplicationService(demo=True, time_scale=1.0)
            await service.start()
            try:
                await service.configure_session(
                    SessionForm(
                        subject_id="S-PREP",
                        experiment_date="2026-09-21",
                        description="preparation abort test",
                        left_trials=1,
                        right_trials=0,
                        output_dir=str(output_dir),
                    )
                )
                await wait_until(lambda: service.device_connected)
                await service.start_experiment()
                self.assertEqual(service.stage, "preparing")
                await service.abort_experiment("test preparation abort")
                self.assertEqual(service.stage, "device_check")
                self.assertFalse(service.session_locked)
            finally:
                await service.stop()

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

    async def test_demo_pipeline_creates_complete_readable_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "complete_session"
            service = ApplicationService(demo=True, time_scale=0.1)
            await service.start()
            try:
                await service.configure_session(
                    SessionForm(
                        subject_id="S-DEMO",
                        experiment_date="2026-09-21",
                        description="end-to-end test",
                        left_trials=1,
                        right_trials=0,
                        output_dir=str(output_dir),
                    )
                )
                await wait_until(lambda: service.device_connected)
                await service.start_experiment()
                await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()

            session = load_session(output_dir / "motor_imagery.h5")
            event_codes = set(session["events"]["event_code"].tolist())
            self.assertEqual(session["status"], "complete")
            self.assertEqual(session["raw_uv"].shape[1], 8)
            self.assertEqual(
                session["raw_uv"].shape,
                session["filtered_uv"].shape,
            )
            self.assertGreater(session["raw_uv"].shape[0], 1_700)
            self.assertLess(session["raw_uv"].shape[0], 2_300)
            self.assertIn(EVENT_RECORDING_START, event_codes)
            self.assertIn(EVENT_CUE_LEFT, event_codes)
            self.assertIn(EVENT_RECORDING_COMPLETE, event_codes)
            self.assertTrue(bool(session["trials"][0]["completed"]))

    async def test_escape_style_abort_preserves_incomplete_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "incomplete_session"
            service = ApplicationService(demo=True, time_scale=0.02)
            await service.start()
            try:
                await service.configure_session(
                    SessionForm(
                        subject_id="S-ABORT",
                        experiment_date="2026-09-21",
                        description="abort test",
                        left_trials=1,
                        right_trials=1,
                        output_dir=str(output_dir),
                    )
                )
                await wait_until(lambda: service.device_connected)
                await service.start_experiment()
                await wait_until(lambda: service.stage in {"rest", "cue"})
                await asyncio.sleep(0.03)
                await service.abort_experiment("test abort")
                self.assertEqual(service.stage, "device_check")
                self.assertTrue(service.session_locked)
            finally:
                await service.stop()

            session = load_session(output_dir / "motor_imagery.h5")
            self.assertEqual(session["status"], "incomplete")
            self.assertGreater(session["raw_uv"].shape[0], 0)
            self.assertEqual(
                session["quality"]["termination_reason"],
                "test abort",
            )


if __name__ == "__main__":
    unittest.main()
