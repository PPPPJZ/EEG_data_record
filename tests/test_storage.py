from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from config import (
    CHANNEL_COUNT,
    EEGStreamConfig,
    CLASS_LEFT,
    EVENT_CUE_END,
    EVENT_CUE_LEFT,
    EVENT_RECORDING_START,
    EVENT_TRIAL_END,
    EVENT_TRIAL_START,
)
from experiment import build_trial_plans, create_session_config
from load_session import extract_trial, load_session
from storage import SessionWriter, ensure_empty_output_dir, write_session_readme


class SessionWriterTest(unittest.TestCase):
    def test_writes_self_contained_hdf5_and_trial_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "session"
            config = create_session_config(
                subject_id="S001",
                experiment_date="2026-09-21",
                description="storage test",
                left_trials=1,
                right_trials=0,
                output_dir=str(output_dir),
                random_seed=9,
            )
            plans = build_trial_plans([CLASS_LEFT])
            stream = EEGStreamConfig(channels=CHANNEL_COUNT)
            writer = SessionWriter(config, plans, stream=stream)
            writer.begin_recording()
            writer.append_event(
                event_code=EVENT_RECORDING_START,
                trial_id=-1,
                class_id=-1,
                planned_time_s=0.0,
                actual_time_s=0.0,
            )
            rest_index = writer.append_event(
                event_code=EVENT_TRIAL_START,
                trial_id=1,
                class_id=CLASS_LEFT,
                planned_time_s=0.0,
                actual_time_s=0.0,
            )
            writer.mark_trial_boundary(
                trial_id=1,
                boundary="rest_start",
                actual_time_s=0.0,
                sample_index=rest_index,
            )

            raw = np.arange(500 * CHANNEL_COUNT, dtype=np.float64).reshape(
                CHANNEL_COUNT, 500
            )
            writer.append_samples(raw[:, :250])
            cue_index = writer.append_event(
                event_code=EVENT_CUE_LEFT,
                trial_id=1,
                class_id=CLASS_LEFT,
                planned_time_s=1.0,
                actual_time_s=1.0,
            )
            writer.mark_trial_boundary(
                trial_id=1,
                boundary="cue_start",
                actual_time_s=1.0,
                sample_index=cue_index,
            )
            writer.append_samples(raw[:, 250:])
            cue_end_index = writer.append_event(
                event_code=EVENT_CUE_END,
                trial_id=1,
                class_id=CLASS_LEFT,
                planned_time_s=2.0,
                actual_time_s=2.0,
            )
            writer.mark_trial_boundary(
                trial_id=1,
                boundary="cue_end",
                actual_time_s=2.0,
                sample_index=cue_end_index,
            )
            trial_end_index = writer.append_event(
                event_code=EVENT_TRIAL_END,
                trial_id=1,
                class_id=CLASS_LEFT,
                planned_time_s=2.0,
                actual_time_s=2.0,
            )
            writer.mark_trial_boundary(
                trial_id=1,
                boundary="trial_end",
                actual_time_s=2.0,
                sample_index=trial_end_index,
            )
            quality = writer.finalize(
                status="complete",
                actual_duration_s=2.0,
                malformed_packets_at_end=0,
            )
            write_session_readme(
                config,
                plans,
                status="complete",
                quality=quality,
                stream=stream,
            )

            session = load_session(output_dir / "motor_imagery.h5")
            whole_trial = extract_trial(session, 1)
            motor_imagery = extract_trial(
                session,
                1,
                signal="raw_uv",
                window="motor_imagery",
            )

            self.assertEqual(session["status"], "complete")
            self.assertEqual(session["raw_uv"].shape, (500, CHANNEL_COUNT))
            np.testing.assert_allclose(session["raw_uv"], raw.T)
            self.assertIsNone(session["filtered_uv"])
            with self.assertRaisesRegex(ValueError, "未保存"):
                extract_trial(session, 1, signal="filtered_uv")
            self.assertEqual(len(session["trials"]), 1)
            self.assertTrue(bool(session["trials"][0]["completed"]))
            self.assertEqual(whole_trial.shape, (500, CHANNEL_COUNT))
            self.assertEqual(motor_imagery.shape, (250, CHANNEL_COUNT))
            self.assertTrue((output_dir / "README.md").is_file())
            with h5py.File(output_dir / "motor_imagery.h5", "r") as handle:
                self.assertIn("protocol/json", handle)
                self.assertIn("events/table", handle)
                self.assertIn("quality", handle)
                self.assertNotIn("eeg/filtered_uv", handle)
                self.assertEqual(handle.attrs["schema_version"], "1.4")

    def test_rejects_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是空文件夹"):
                ensure_empty_output_dir(output_dir)


if __name__ == "__main__":
    unittest.main()
