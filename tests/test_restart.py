"""只使用临时模拟数据验证断连回退、恢复设置和原文件续采。"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace

import h5py
import numpy as np

from app import ApplicationService, SessionForm
from config import (
    EEGStreamConfig,
    EVENT_RECORDING_START, EVENT_TRIAL_START, EVENT_CUE_LEFT, EVENT_CUE_RIGHT,
    EVENT_CUE_END, EVENT_TRIAL_END, EVENT_RECORDING_ABORTED,
    EVENT_BREAK_INSTRUCTION_START, EVENT_BREAK_MONITOR_START, EVENT_BREAK_END,
)
from experiment import create_session_config, build_trial_sequence, build_trial_plans
from load_session import load_session
from restart import parse_resume_path
from storage import SessionWriter, write_session_readme
from test_pipeline import wait_until

# 本脚本所有模拟时序显式配置，与真实数据和硬件隔离。
TIME_SCALE = 0.05
BLOCK_SIZE = 20


def fixture(path: Path, *, completed=18, total=21, break_state="none", channels=8):
    config = create_session_config(
        subject_id="RESTART-TEST", experiment_date="2026-10-20", description="恢复原配置",
        left_trials=total // 2, right_trials=total - total // 2,
        output_dir=str(path), random_seed=71, break_every_trials=BLOCK_SIZE,
    )
    plans = build_trial_plans(build_trial_sequence(config), config.break_every_trials)
    stream = EEGStreamConfig(channels=channels)
    writer = SessionWriter(config, plans, demo=True, time_scale=TIME_SCALE, stream=stream)
    writer.begin_recording()
    writer.append_event(event_code=EVENT_RECORDING_START, trial_id=-1, class_id=-1,
                        planned_time_s=0, actual_time_s=0)
    wall_offset = 0

    def boundary(plan, name, code, t):
        desired = round(t * 250)
        size = desired - writer.sample_count
        if size:
            raw = np.full((channels, size), plan.trial_id, np.float32)
            writer.append_samples(raw)
        sample = writer.append_event(
            event_code=code, trial_id=plan.trial_id, class_id=plan.class_id,
            planned_time_s=t, actual_time_s=t, wall_time_s=t + wall_offset,
        )
        writer.mark_trial_boundary(trial_id=plan.trial_id, boundary=name,
                                   actual_time_s=t, sample_index=sample)

    for plan in plans[:completed]:
        boundary(plan, "rest_start", EVENT_TRIAL_START, plan.planned_rest_start_s)
        boundary(plan, "cue_start", EVENT_CUE_LEFT if plan.class_id == 0 else EVENT_CUE_RIGHT, plan.planned_cue_start_s)
        boundary(plan, "cue_end", EVENT_CUE_END, plan.planned_cue_end_s)
        boundary(plan, "trial_end", EVENT_TRIAL_END, plan.planned_trial_end_s)
    if break_state != "none":
        t = plans[completed - 1].planned_trial_end_s
        for code, offset in [(EVENT_BREAK_INSTRUCTION_START, 0), (EVENT_BREAK_MONITOR_START, 3)]:
            writer.append_event(event_code=code, trial_id=completed, class_id=-1,
                                planned_time_s=t, actual_time_s=t, wall_time_s=t + offset)
        wall_offset = 12
        if break_state == "complete":
            wall_offset = 33
            writer.append_event(event_code=EVENT_BREAK_END, trial_id=completed, class_id=-1,
                                planned_time_s=t, actual_time_s=t, wall_time_s=t + wall_offset)
    if break_state != "partial":
        plan = plans[completed]
        boundary(plan, "rest_start", EVENT_TRIAL_START, plan.planned_rest_start_s)
        raw = np.full((channels, 100), -999, np.float32)
        writer.append_samples(raw)
    duration = writer.sample_count / 250
    writer.append_event(event_code=EVENT_RECORDING_ABORTED, trial_id=completed + 1, class_id=-1,
                        planned_time_s=duration, actual_time_s=duration, wall_time_s=duration + wall_offset)
    quality = writer.finalize(status="incomplete", actual_duration_s=duration,
                              wall_duration_s=duration + wall_offset, malformed_packets_at_end=0,
                              reason="fixture disconnect")
    write_session_readme(config, plans, status="incomplete", quality=quality, stream=stream)
    return path / "motor_imagery.h5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RestartStorageTest(unittest.TestCase):
    def test_legacy_filtered_prefix_is_readable_but_never_extended(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "legacy", completed=1, total=2)
            with h5py.File(path, "r+") as root:
                root.attrs["schema_version"] = "1.3"
                protocol = json.loads(root["protocol/json"][()])
                protocol.pop("acquisition")
                protocol.pop("saved_signals")
                root["protocol/json"][()] = json.dumps(protocol)
                root["eeg"].create_dataset(
                    "filtered_uv", data=root["eeg/raw_uv"][:] / 2,
                    maxshape=(None, 8), chunks=(250, 8),
                )
            info = parse_resume_path(str(path))
            info.check_stream(EEGStreamConfig(channels=8), demo=True)
            before = load_session(path)["filtered_uv"].copy()
            writer = SessionWriter.resume(info)
            writer.append_samples(np.full((8, 300), 777.0))
            writer.finalize(status="incomplete", actual_duration_s=9.2, malformed_packets_at_end=0)
            after = load_session(path)
            np.testing.assert_array_equal(after["filtered_uv"], before[:2000])
            self.assertEqual(len(after["raw_uv"]), 2300)
            self.assertEqual(after["protocol"]["legacy_filtered_samples"], 2000)
            parse_resume_path(str(path))  # 再次续采也允许短于 raw 的历史前缀。
            with self.assertRaisesRegex(ValueError, "采集源"):
                info.check_stream(EEGStreamConfig(channels=8), demo=False)

    def test_32_channel_resume_preserves_prefix_and_stream_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "wide", completed=1, total=2, channels=32)
            before = load_session(path)["raw_uv"]
            info = parse_resume_path(str(path))
            info.check_stream(EEGStreamConfig(channels=32), demo=True)
            with self.assertRaisesRegex(ValueError, "通道"):
                info.check_stream(EEGStreamConfig(channels=8), demo=True)
            writer = SessionWriter.resume(info)
            writer.append_samples(np.full((32, 250), 123.0))
            writer.finalize(status="incomplete", actual_duration_s=9, malformed_packets_at_end=0)
            after = load_session(path)
            np.testing.assert_array_equal(after["raw_uv"][:2000], before[:2000])
            self.assertEqual(after["raw_uv"].shape, (2250, 32))
            self.assertIsNone(after["filtered_uv"])

    def test_saved_teacher_hash_checked_before_model_load(self):
        from self_learning.runtime import FeedbackRuntime
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher.pt"
            path.write_bytes(b"changed checkpoint")
            predictor = Mock()
            runtime = FeedbackRuntime(str(path), expected_sha256="original hash",
                                      saved_config={"ema_alpha": 0.42})
            try:
                self.assertEqual(runtime.config.ema_alpha, 0.42)
                with patch.dict("sys.modules", {
                    "adfcnn.adfcnn_lib.inference": SimpleNamespace(Predictor=predictor),
                }):
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        runtime._load()
                    predictor.assert_not_called()
            finally:
                runtime.executor.shutdown()

    def test_parser_read_only_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "session"
            path = fixture(folder)
            before = digest(path), digest(folder / "README.md")
            info = parse_resume_path(str(folder))
            self.assertEqual(info.payload()["completed_trials"], 18)
            self.assertEqual(info.payload()["remaining_trials"], 3)
            self.assertEqual(info.payload()["trials_before_break"], 2)
            self.assertEqual(info.payload()["remaining_breaks"], 1)
            self.assertEqual((digest(path), digest(folder / "README.md")), before)
            for value in ("", "relative", str(folder / "missing.h5"), str(folder / "README.md")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    parse_resume_path(value)
            bad = Path(temp) / "other.h5"
            with h5py.File(bad, "w") as h:
                h.create_dataset("hello", data=[1, 2])
            with self.assertRaisesRegex(ValueError, "不是本系统"):
                parse_resume_path(str(bad))

    def test_legacy_partial_cleanup_preserves_prefix_and_original_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "session")
            renamed = path.with_name("chosen.h5")
            path.rename(renamed)
            before = load_session(renamed)
            info = parse_resume_path(str(renamed))
            writer = SessionWriter.resume(info)
            self.assertEqual(writer.sample_count, 18 * 2000)
            writer.finalize(status="incomplete", actual_duration_s=144, malformed_packets_at_end=0)
            after = load_session(renamed)
            np.testing.assert_array_equal(before["raw_uv"][:36000], after["raw_uv"])
            self.assertIsNone(after["filtered_uv"])
            self.assertEqual(before["trials"][:18].tobytes(), after["trials"][:18].tobytes())
            self.assertTrue(np.all(after["trials"]["rest_start_sample"][18:] == -1))
            self.assertTrue(np.isnan(after["trials"]["actual_rest_start_s"][18:]).all())
            self.assertNotIn(EVENT_RECORDING_ABORTED, after["events"]["event_code"])
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "发生了变化"):
                info.check_unchanged()

    def test_corrupt_completed_boundary_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "session", completed=1, total=3)
            with h5py.File(path, "r+") as h:
                row = h["trials/table"][0]
                row["cue_start_sample"] = 100000
                h["trials/table"][0] = row
            before = digest(path)
            with self.assertRaisesRegex(ValueError, "边界"):
                parse_resume_path(str(path))
            self.assertEqual(digest(path), before)


class RestartPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def ready_resume(self, service, path):
        await service.parse_restart(str(path))
        await service.configure_restart()
        await wait_until(lambda: service.state_payload()["can_start"])

    async def test_trial_19_resumes_two_then_break_and_appends_same_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "session")
            original = load_session(path)
            service = ApplicationService(demo=True, time_scale=TIME_SCALE)
            try:
                # 修改当前默认值不应影响旧文件恢复的时长。
                with patch("experiment.TRIAL_SECONDS", 88), patch("experiment.REST_SECONDS", 22):
                    await self.ready_resume(service, path)
                    self.assertEqual(service.timing["trial_seconds"], 8)
                    self.assertEqual(service.timing["rest_seconds"], 2.5)
                    self.assertEqual(service.config.expected_duration_seconds, 168)
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == "break_monitor")
                    self.assertEqual(service._break_after_trial_id, 20)
                    self.assertEqual(service.state_payload()["completed_trials"], 20)
                    await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()
            result = load_session(path)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["trials"]["completed"].all())
            self.assertEqual(original["trials"][:18].tobytes(), result["trials"][:18].tobytes())
            np.testing.assert_array_equal(original["raw_uv"][:36000], result["raw_uv"][:36000])
            self.assertIsNone(result["filtered_uv"])
            np.testing.assert_array_equal(result["trials"]["class_id"], original["trials"]["class_id"])
            self.assertEqual(result["trials"][18]["rest_start_sample"], 36000)
            self.assertNotIn(EVENT_RECORDING_ABORTED, result["events"]["event_code"])
            self.assertEqual(sum(result["events"]["event_code"] == EVENT_RECORDING_START), 1)
            self.assertTrue(np.all(np.diff(result["events"]["wall_time_s"]) >= 0))
            self.assertTrue(np.all(np.diff(result["events"]["actual_time_s"]) >= 0))
            self.assertTrue(np.all(np.diff(result["events"]["sample_index"]) >= 0))
            self.assertEqual(result["quality"]["termination_reason"], "")
            text = path.with_name("README.md").read_text(encoding="utf-8")
            self.assertIn("`complete`", text)
            self.assertIn("第 19 个 trial", text)
            self.assertEqual(text.count("<!-- interruption:"), 1)
            self.assertEqual({p.name for p in path.parent.iterdir()}, {"motor_imagery.h5", "README.md"})

    async def test_cancelled_preparation_and_reselection_do_not_mutate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp) / "session", completed=0, total=2)
            before = digest(path), digest(path.with_name("README.md"))
            service = ApplicationService(demo=True, time_scale=TIME_SCALE)
            try:
                await self.ready_resume(service, path)
                await service.start_experiment()
                await service.abort_experiment("取消续采准备")
                self.assertEqual(service.last_outcome, "cancelled")
                self.assertEqual((digest(path), digest(path.with_name("README.md"))), before)
                with self.assertRaises(ValueError):
                    await service.parse_restart(str(path.with_name("bad.h5")))
                with self.assertRaisesRegex(ValueError, "先解析"):
                    await service.configure_restart()
                await service.parse_restart(str(path))
            finally:
                await service.stop()

    async def test_completed_or_partial_break_at_boundary(self):
        for break_state in ("none", "partial", "complete"):
            with self.subTest(break_state=break_state), tempfile.TemporaryDirectory() as temp:
                path = fixture(Path(temp) / "session", completed=20, break_state=break_state)
                service = ApplicationService(demo=True, time_scale=TIME_SCALE)
                try:
                    info = await service.parse_restart(str(path))
                    self.assertEqual(info["remaining_breaks"], int(break_state != "complete"))
                    await service.configure_restart()
                    await wait_until(lambda: service.state_payload()["can_start"])
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == "complete")
                finally:
                    await service.stop()
                result = load_session(path)
                self.assertEqual(sum(result["events"]["event_code"] == EVENT_BREAK_END), 1)
                self.assertEqual(result["trials"][20]["rest_start_sample"], 40000)

    async def test_device_disconnect_rolls_back_all_modes_and_can_repeat(self):
        for mode in ("pure_MI", "tactile_MI", "self_learning"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / "session"
                service = ApplicationService(demo=True, time_scale=0.1, mock_teacher=mode == "self_learning")
                try:
                    await service.configure_session(SessionForm(
                        subject_id="DISCONNECT", experiment_date="2026-10-20", experiment_mode=mode,
                        left_trials=2, right_trials=0, learning_trials=2, output_dir=str(folder),
                    ))
                    await wait_until(lambda: service.state_payload()["can_start"])
                    await service.start_experiment()
                    await wait_until(lambda: service._trial_index == 1 and service.stage in {"cue", "self_learning"})
                    if mode == "tactile_MI":
                        service.haptics._disconnected(0)
                    else:
                        service._on_source_status("模拟 EEG 断连", False)
                    await wait_until(lambda: service.last_outcome == "incomplete")
                    path = folder / "motor_imagery.h5"
                    first = load_session(path)
                    self.assertEqual(first["quality"]["completed_trials"], 1)
                    self.assertEqual(len(first["raw_uv"]), first["trials"][0]["trial_end_sample"])
                    self.assertEqual(first["trials"][1]["rest_start_sample"], -1)
                    self.assertNotIn(EVENT_RECORDING_ABORTED, first["events"]["event_code"])
                    if mode == "self_learning":
                        self.assertTrue(np.all(first["feedback"]["trial_id"] == 1))
                    if mode == "tactile_MI":
                        self.assertFalse(np.any(first["haptic_commands"]["trial_id"] == 2))
                    # 续采又断一次，再次继续，已完成的 trial 始终不变。
                    with patch("self_learning.control.ControlConfig.load", side_effect=AssertionError("不得读取新 YAML")):
                        await self.ready_resume(service, path)
                    if mode == "self_learning":
                        self.assertEqual(service.feedback.state_payload()["config"], first["feedback_config"]["config"])
                    if mode == "tactile_MI":
                        self.assertEqual(service.haptics.protocol, first["protocol"]["haptic_stimulation"])
                    await service.start_experiment()
                    await wait_until(lambda: service._trial_index == 1 and service.stage == "rest")
                    service._on_source_status("第二次断连", False)
                    await wait_until(lambda: service.last_outcome == "incomplete")
                    second = load_session(path)
                    np.testing.assert_array_equal(second["raw_uv"], first["raw_uv"])
                    await self.ready_resume(service, path)
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == "complete")
                finally:
                    await service.stop()
                final = load_session(path)
                self.assertEqual(final["status"], "complete")
                self.assertTrue(final["trials"]["completed"].all())
                np.testing.assert_array_equal(final["raw_uv"][:len(first["raw_uv"])], first["raw_uv"])
                history = (folder / "README.md").read_text(encoding="utf-8")
                self.assertEqual(history.count("<!-- interruption:"), 2)
                with self.assertRaisesRegex(ValueError, "complete"):
                    parse_resume_path(str(path))

    async def test_first_trial_disconnect_leaves_no_eeg_then_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "session"
            service = ApplicationService(demo=True, time_scale=TIME_SCALE)
            try:
                await service.configure_session(SessionForm(
                    subject_id="FIRST", experiment_date="2026-10-20",
                    left_trials=1, right_trials=0, output_dir=str(folder),
                ))
                await wait_until(lambda: service.state_payload()["can_start"])
                await service.start_experiment()
                await wait_until(lambda: service.stage == "cue")
                service._on_source_status("first trial lost", False)
                await wait_until(lambda: service.last_outcome == "incomplete")
                path = folder / "motor_imagery.h5"
                self.assertEqual(len(load_session(path)["raw_uv"]), 0)
                await self.ready_resume(service, path)
                await service.start_experiment()
                await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()
            result = load_session(path)
            self.assertEqual(sum(result["events"]["event_code"] == EVENT_RECORDING_START), 1)
            self.assertEqual(result["quality"]["completed_trials"], 1)


if __name__ == "__main__":
    unittest.main()
