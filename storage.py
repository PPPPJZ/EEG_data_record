"""自包含 HDF5 会话文件的流式写入与 README 生成。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from config import (
    BANDPASS_CASCADE_COUNT,
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    EEGStreamConfig,
    CLASS_NAMES,
    DATA_FILENAME,
    DISPLAY_LIMIT_UV,
    EVENT_NAMES,
    HDF5_SCHEMA_NAME,
    HDF5_SCHEMA_VERSION,
    HDF5_WRITE_BLOCK_SAMPLES,
    NOTCH_HZ,
    NOTCH_Q,
    SAMPLE_RATE_HZ,
    SESSION_README_FILENAME,
)
from experiment import SessionConfig, TrialPlan
from haptics import HapticCommand

HAPTIC_COMMAND_DTYPE = np.dtype([
    ("motor_id", "i1"),
    ("trial_id", "<i4"),
    ("context", "S16"),
    ("percentage", "<i2"),
    ("requested_wall_time_s", "<f8"),
    ("acknowledged_wall_time_s", "<f8"),
    ("sample_index", "<i8"),
    ("success", "?"),
    ("error", h5py.string_dtype(encoding="utf-8")),
])

FEEDBACK_DTYPE = np.dtype([
    ("trial_id", "<i4"), ("window_start_sample", "<i8"), ("window_end_sample", "<i8"),
    ("prediction_class_id", "i1"), ("p_right", "<f8"), ("signal", "<f8"),
    ("smooth", "<f8"), ("control", "<f8"), ("velocity", "<f8"), ("cursor_x", "<f8"),
    ("inference_ms", "<f8"), ("wall_time_s", "<f8"), ("actual_time_s", "<f8"),
    ("control_elapsed_s", "<f8"),
])

EVENT_DTYPE = np.dtype(
    [
        ("event_id", "<i4"),
        ("event_code", "<i2"),
        ("trial_id", "<i4"),
        ("class_id", "i1"),
        ("planned_time_s", "<f8"),
        ("actual_time_s", "<f8"),
        ("wall_time_s", "<f8"),
        ("sample_index", "<i8"),
        ("timing_error_ms", "<f8"),
        ("sample_clock_error_ms", "<f8"),
    ]
)

TRIAL_DTYPE = np.dtype(
    [
        ("trial_id", "<i4"),
        ("block_id", "<i4"),
        ("class_id", "i1"),
        ("planned_rest_start_s", "<f8"),
        ("planned_cue_start_s", "<f8"),
        ("planned_cue_end_s", "<f8"),
        ("planned_trial_end_s", "<f8"),
        ("actual_rest_start_s", "<f8"),
        ("actual_cue_start_s", "<f8"),
        ("actual_cue_end_s", "<f8"),
        ("actual_trial_end_s", "<f8"),
        ("rest_start_sample", "<i8"),
        ("cue_start_sample", "<i8"),
        ("cue_end_sample", "<i8"),
        ("trial_end_sample", "<i8"),
        ("completed", "?"),
    ]
)

TRIAL_BOUNDARY_FIELDS = {
    "rest_start": ("actual_rest_start_s", "rest_start_sample"),
    "cue_start": ("actual_cue_start_s", "cue_start_sample"),
    "cue_end": ("actual_cue_end_s", "cue_end_sample"),
    "trial_end": ("actual_trial_end_s", "trial_end_sample"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("实验数据记录路径已存在，但不是文件夹")
    path.mkdir(parents=True, exist_ok=True)
    try:
        next(path.iterdir())
    except StopIteration:
        return
    raise ValueError("实验数据记录文件夹不是空文件夹，已拒绝覆盖")


class SessionWriter:
    """以追加方式写入一个实验的 EEG、事件和 trial 边界。"""

    def __init__(
        self,
        config: SessionConfig,
        plans: list[TrialPlan],
        *,
        malformed_packets_at_start: int = 0,
        demo: bool = False,
        time_scale: float = 1.0,
        feedback_metadata: dict | None = None,
        stream: EEGStreamConfig | None = None,
    ) -> None:
        self.stream = stream or EEGStreamConfig()
        self.channels = self.stream.channels
        ensure_empty_output_dir(config.output_dir)
        self.config = config
        self.plans = plans
        self.path = config.output_dir / DATA_FILENAME
        self.created_at_utc = utc_now_iso()
        self.recording_started_at_utc: str | None = None
        self.closed = False
        self._malformed_packets_at_start = malformed_packets_at_start
        self._event_count = 0
        self._flushed_sample_count = 0
        self._raw_pending: list[np.ndarray] = []
        self._pending_sample_count = 0
        self._runtime = {"demo": demo, "time_scale": time_scale}
        self._feedback_metadata = feedback_metadata
        self._feedback_dataset = None
        self._original_metadata = {}
        self._previous_malformed = 0

        self._file = h5py.File(self.path, "x")
        self._create_schema()

    @classmethod
    def resume(cls, info, *, malformed_packets_at_start: int = 0) -> "SessionWriter":
        """准备完成后才以 r+ 打开原文件；不会创建新的 EEG 文件。"""
        info.check_unchanged()
        self = cls.__new__(cls)
        self.config, self.plans, self.path = info.config, info.plans, info.path
        self.created_at_utc = info.metadata["created_at_utc"]
        self.recording_started_at_utc = info.metadata.get("recording_started_at_utc")
        self.closed = False
        self._malformed_packets_at_start = malformed_packets_at_start
        self._previous_malformed = info.metadata.get("quality", {}).get("malformed_packets_during_session", 0)
        self._runtime = info.metadata["runtime"]
        self._original_metadata = {
            k: v for k, v in info.metadata.items()
            if k not in {"status", "quality", "termination_reason", "ended_at_utc"}
        }
        self._feedback_metadata = info.feedback
        self._raw_pending = []
        self._pending_sample_count = 0
        self._file = h5py.File(self.path, "r+")
        try:
            self._metadata_dataset = self._file["metadata/json"]
            self._raw_dataset = self._file["eeg/raw_uv"]
            self.channels = self._raw_dataset.shape[1]
            # 旧文件已保存的滤波前缀保留；续采从此只追加 raw，不再写滤波数据。
            self._legacy_filtered_dataset = self._file.get("eeg/filtered_uv")
            self._events_dataset = self._file["events/table"]
            self._haptic_dataset = self._file["haptics/commands"]
            self._trials_dataset = self._file["trials/table"]
            self._feedback_dataset = self._file.get("self_learning/feedback")
            self._quality_group = self._file["quality"]
            self._trial_rows = self._trials_dataset[:]
            self._flushed_sample_count = len(self._raw_dataset)
            self._event_count = len(self._events_dataset)
            # 老文件的中断记录只在真正续采时补入 README，解析/调试不修改文件。
            record_interruption(
                self.config.output_dir, trial_id=info.cut.completed + 1,
                completed=info.cut.completed,
                reason=info.metadata.get("termination_reason") or "原实验未正常结束",
                key=info.metadata.get("ended_at_utc") or info.metadata["created_at_utc"],
            )
            self.recovery = self.rollback_incomplete_trials()
            protocol = dict(info.protocol)
            protocol["saved_signals"] = ["raw_uv"]
            if self._legacy_filtered_dataset is not None:
                protocol["legacy_filtered_samples"] = len(self._legacy_filtered_dataset)
            self._file["protocol/json"][()] = json.dumps(protocol, ensure_ascii=False, sort_keys=True)
            self._file.attrs["schema_version"] = HDF5_SCHEMA_VERSION
            self._file.flush()
            return self
        except BaseException:
            self._file.close()
            self.closed = True
            raise

    def rollback_incomplete_trials(self):
        from dataclasses import replace
        from restart import recovery_cut

        self._require_open()
        self.flush_samples()
        cut = recovery_cut(self._trial_rows, self._events_dataset[:])
        self._raw_dataset.resize((cut.samples, self.channels))
        legacy = getattr(self, "_legacy_filtered_dataset", None)
        if legacy is not None:
            legacy.resize((min(cut.samples, len(legacy)), self.channels))
            protocol = json.loads(self._file["protocol/json"][()])
            protocol["legacy_filtered_samples"] = len(legacy)
            self._file["protocol/json"][()] = json.dumps(protocol, ensure_ascii=False, sort_keys=True)
        self._flushed_sample_count = cut.samples
        self._events_dataset.resize((cut.events,))
        self._event_count = cut.events
        commands = self._haptic_dataset[:]
        keep = (
            (commands["sample_index"] <= cut.samples)
            & (commands["trial_id"] <= cut.completed)
            & (
                ((commands["trial_id"] >= 1) & commands["success"])
                | (commands["acknowledged_wall_time_s"] <= cut.wall_seconds)
            )
        ) if cut.completed else np.zeros(len(commands), dtype=bool)
        commands = commands[keep]
        self._haptic_dataset.resize((len(commands),))
        self._haptic_dataset[:] = commands
        if len(commands):
            cut = replace(cut, wall_seconds=max(cut.wall_seconds, float(np.max(commands["acknowledged_wall_time_s"]))))
        if self._feedback_dataset is not None:
            rows = self._feedback_dataset[:]
            rows = rows[(rows["trial_id"] <= cut.completed) & (rows["window_end_sample"] <= cut.samples)]
            self._feedback_dataset.resize((len(rows),))
            self._feedback_dataset[:] = rows
        for field in TRIAL_DTYPE.names:
            if field.startswith("actual_"):
                self._trial_rows[field][cut.completed:] = np.nan
            elif field.endswith("_sample"):
                self._trial_rows[field][cut.completed:] = -1
        self._trial_rows["completed"][cut.completed:] = False
        self._trials_dataset[cut.completed:] = self._trial_rows[cut.completed:]
        self._file.flush()
        return cut

    @property
    def sample_count(self) -> int:
        return self._flushed_sample_count + self._pending_sample_count

    def _create_schema(self) -> None:
        root = self._file
        root.attrs["schema_name"] = HDF5_SCHEMA_NAME
        root.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        root.attrs["status"] = "preparing"
        root.attrs["created_at_utc"] = self.created_at_utc

        metadata_group = root.create_group("metadata")
        self._metadata_dataset = metadata_group.create_dataset(
            "json",
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self._write_metadata_json(status="preparing")

        protocol = {
            **self.config.protocol,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channel_count": self.channels,
            "channel_labels": list(self.stream.channel_labels),
            "signal_unit": "uV",
            "raw_signal": "received values before local software filtering; upstream processing unknown",
            "saved_signals": ["raw_uv"],
            "acquisition": self.stream.metadata(demo=self._runtime["demo"]),
            "filtered_signal": {
                "notch_hz": NOTCH_HZ,
                "notch_q": NOTCH_Q,
                "bandpass_hz": [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                "butterworth_order_per_stage": BANDPASS_ORDER,
                "cascade_count": BANDPASS_CASCADE_COUNT,
            },
            "display_limit_uv": DISPLAY_LIMIT_UV,
            "class_codes": {str(key): value for key, value in CLASS_NAMES.items()},
            "event_codes": {str(key): value for key, value in EVENT_NAMES.items()},
            "sample_index_convention": "zero-based; slices are half-open [start:end)",
            "time_axes": {
                "planned_time_s": "protocol acquisition time; excludes breaks",
                "actual_time_s": (
                    "server active-acquisition time; excludes breaks and pauses"
                ),
                "wall_time_s": (
                    "server elapsed time since recording start; includes breaks"
                ),
            },
        }
        protocol_group = root.create_group("protocol")
        protocol_group.create_dataset(
            "json",
            data=json.dumps(protocol, ensure_ascii=False, sort_keys=True),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        eeg_group = root.create_group("eeg")
        eeg_group.attrs["sample_rate_hz"] = SAMPLE_RATE_HZ
        eeg_group.attrs["units"] = "uV"
        eeg_group.attrs["channel_labels"] = np.asarray(
            self.stream.channel_labels,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset_options = {
            "shape": (0, self.channels),
            "maxshape": (None, self.channels),
            "chunks": (HDF5_WRITE_BLOCK_SAMPLES, self.channels),
            "dtype": np.float32,
            "compression": "lzf",
            "shuffle": True,
        }
        self._raw_dataset = eeg_group.create_dataset("raw_uv", **dataset_options)

        events_group = root.create_group("events")
        events_group.attrs["event_code_map_json"] = json.dumps(
            {str(key): value for key, value in EVENT_NAMES.items()},
            sort_keys=True,
        )
        self._events_dataset = events_group.create_dataset(
            "table",
            shape=(0,),
            maxshape=(None,),
            chunks=(128,),
            dtype=EVENT_DTYPE,
        )

        haptic_group = root.create_group("haptics")
        self._haptic_dataset = haptic_group.create_dataset(
            "commands", shape=(0,), maxshape=(None,), chunks=(128,),
            dtype=HAPTIC_COMMAND_DTYPE,
        )
        if self.config.self_learning_enabled:
            group = root.create_group("self_learning")
            group.create_dataset(
                "config_json", data=json.dumps(self._feedback_metadata, ensure_ascii=False),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            group.attrs["labels"] = "trial class_id=-1 unknown; predictions are not ground truth"
            group.attrs["window"] = "750 raw samples within current trial imagery; end exclusive"
            self._feedback_dataset = group.create_dataset(
                "feedback", shape=(0,), maxshape=(None,), chunks=(256,), dtype=FEEDBACK_DTYPE,
            )

        self._trial_rows = np.empty(len(self.plans), dtype=TRIAL_DTYPE)
        for index, plan in enumerate(self.plans):
            row = self._trial_rows[index]
            row["trial_id"] = plan.trial_id
            row["block_id"] = plan.block_id
            row["class_id"] = plan.class_id
            row["planned_rest_start_s"] = plan.planned_rest_start_s
            row["planned_cue_start_s"] = plan.planned_cue_start_s
            row["planned_cue_end_s"] = plan.planned_cue_end_s
            row["planned_trial_end_s"] = plan.planned_trial_end_s
            for field in (
                "actual_rest_start_s",
                "actual_cue_start_s",
                "actual_cue_end_s",
                "actual_trial_end_s",
            ):
                row[field] = np.nan
            for field in (
                "rest_start_sample",
                "cue_start_sample",
                "cue_end_sample",
                "trial_end_sample",
            ):
                row[field] = -1
            row["completed"] = False

        trials_group = root.create_group("trials")
        trials_group.attrs["class_code_map_json"] = json.dumps(
            {str(key): value for key, value in CLASS_NAMES.items()},
            sort_keys=True,
        )
        self._trials_dataset = trials_group.create_dataset(
            "table",
            data=self._trial_rows,
            dtype=TRIAL_DTYPE,
        )

        self._quality_group = root.create_group("quality")
        self._quality_group.attrs["expected_sample_count"] = int(
            round(self.config.expected_duration_seconds * SAMPLE_RATE_HZ)
        )
        self._quality_group.attrs["malformed_packets_at_start"] = int(
            self._malformed_packets_at_start
        )
        root.flush()

    def _write_metadata_json(self, *, status: str, **extra: Any) -> None:
        metadata = {
            **self._original_metadata,
            **self.config.as_serializable_dict(),
            "status": status,
            "created_at_utc": self.created_at_utc,
            "recording_started_at_utc": self.recording_started_at_utc,
            "runtime": self._runtime,
            **extra,
        }
        self._metadata_dataset[()] = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
        )

    def begin_recording(self) -> None:
        self._require_open()
        self.recording_started_at_utc = self.recording_started_at_utc or utc_now_iso()
        if "ended_at_utc" in self._file.attrs:
            del self._file.attrs["ended_at_utc"]
        self._file.attrs["status"] = "recording"
        self._file.attrs["recording_started_at_utc"] = self.recording_started_at_utc
        self._write_metadata_json(status="recording")
        self._file.flush()

    def append_samples(
        self,
        raw_uv: np.ndarray,
    ) -> None:
        self._require_open()
        raw = np.asarray(raw_uv, dtype=np.float32)
        expected_prefix = (self.channels,)
        if raw.ndim != 2 or raw.shape[0:1] != expected_prefix:
            raise ValueError(f"raw EEG shape must be ({self.channels}, samples)")
        if not np.isfinite(raw).all():
            raise ValueError("raw EEG must contain finite values")
        if raw.shape[1] == 0:
            return

        self._raw_pending.append(raw.copy())
        self._pending_sample_count += raw.shape[1]
        if self._pending_sample_count >= HDF5_WRITE_BLOCK_SAMPLES:
            self.flush_samples()

    def flush_samples(self) -> None:
        self._require_open()
        if not self._raw_pending:
            return
        raw = np.hstack(self._raw_pending).T
        start = self._flushed_sample_count
        end = start + raw.shape[0]
        self._raw_dataset.resize((end, self.channels))
        self._raw_dataset[start:end] = raw
        self._flushed_sample_count = end
        self._raw_pending.clear()
        self._pending_sample_count = 0
        self._file.flush()

    def append_event(
        self,
        *,
        event_code: int,
        trial_id: int,
        class_id: int,
        planned_time_s: float,
        actual_time_s: float,
        wall_time_s: float | None = None,
    ) -> int:
        self._require_open()
        self.flush_samples()
        sample_index = self.sample_count
        if wall_time_s is None:
            wall_time_s = actual_time_s
        timing_error_ms = (actual_time_s - planned_time_s) * 1000.0
        sample_clock_error_ms = (
            sample_index / SAMPLE_RATE_HZ - actual_time_s
        ) * 1000.0
        event_row = np.array(
            (
                self._event_count,
                event_code,
                trial_id,
                class_id,
                planned_time_s,
                actual_time_s,
                wall_time_s,
                sample_index,
                timing_error_ms,
                sample_clock_error_ms,
            ),
            dtype=EVENT_DTYPE,
        )
        self._events_dataset.resize((self._event_count + 1,))
        self._events_dataset[self._event_count] = event_row
        self._event_count += 1
        self._file.flush()
        return sample_index

    def mark_trial_boundary(
        self,
        *,
        trial_id: int,
        boundary: str,
        actual_time_s: float,
        sample_index: int,
    ) -> None:
        self._require_open()
        if boundary not in TRIAL_BOUNDARY_FIELDS:
            raise ValueError(f"unknown trial boundary: {boundary}")
        if not 1 <= trial_id <= len(self._trial_rows):
            raise ValueError(f"trial id out of range: {trial_id}")
        time_field, sample_field = TRIAL_BOUNDARY_FIELDS[boundary]
        row_index = trial_id - 1
        self._trial_rows[row_index][time_field] = actual_time_s
        self._trial_rows[row_index][sample_field] = sample_index
        if boundary == "trial_end":
            self._trial_rows[row_index]["completed"] = True
        self._trials_dataset[row_index] = self._trial_rows[row_index]
        self._file.flush()

    def append_haptic_command(
        self, command: HapticCommand, *, recording_origin: float, time_scale: float
    ) -> None:
        self._require_open()
        index = len(self._haptic_dataset)
        self._haptic_dataset.resize((index + 1,))
        self._haptic_dataset[index] = np.array((
            command.motor_id,
            command.trial_id,
            command.context.encode("ascii"),
            command.percentage,
            (command.requested_at - recording_origin) / time_scale,
            (command.acknowledged_at - recording_origin) / time_scale,
            self.sample_count,
            command.success,
            command.error,
        ), dtype=HAPTIC_COMMAND_DTYPE)
        # 不逐条 flush；跟随 EEG 块、协议事件或最终关闭一起落盘。

    def append_feedback(self, row: dict, *, wall_time_s: float, actual_time_s: float) -> None:
        self._require_open()
        if self._feedback_dataset is None:
            raise ValueError("当前会话未启用自主学习")
        values = {**row, "wall_time_s": wall_time_s, "actual_time_s": actual_time_s}
        index = len(self._feedback_dataset)
        self._feedback_dataset.resize((index + 1,))
        self._feedback_dataset[index] = np.array(
            tuple(values[name] for name in FEEDBACK_DTYPE.names), dtype=FEEDBACK_DTYPE,
        )

    def finalize(
        self,
        *,
        status: str,
        actual_duration_s: float,
        wall_duration_s: float | None = None,
        malformed_packets_at_end: int,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_open()
        if status not in {"complete", "incomplete"}:
            raise ValueError("final status must be complete or incomplete")
        self.flush_samples()

        expected_samples = int(
            round(self.config.expected_duration_seconds * SAMPLE_RATE_HZ)
        )
        recorded_samples = self.sample_count
        sample_duration_s = recorded_samples / SAMPLE_RATE_HZ
        if wall_duration_s is None:
            wall_duration_s = actual_duration_s
        malformed_during_session = self._previous_malformed + max(
            0, malformed_packets_at_end - self._malformed_packets_at_start
        )
        quality = {
            "expected_sample_count": expected_samples,
            "recorded_sample_count": recorded_samples,
            "expected_duration_s": self.config.expected_duration_seconds,
            "actual_duration_s": actual_duration_s,
            "wall_duration_s": wall_duration_s,
            "sample_count_duration_s": sample_duration_s,
            "sample_count_time_error_ms": (
                sample_duration_s - actual_duration_s
            )
            * 1000.0,
            "coverage": (
                recorded_samples / expected_samples if expected_samples else 0.0
            ),
            "malformed_packets_during_session": malformed_during_session,
            "completed_trials": int(np.count_nonzero(self._trial_rows["completed"])),
            "planned_trials": len(self.plans),
            "scheduled_breaks": self.config.scheduled_break_count,
            "termination_reason": reason,
            "haptic_command_count": len(self._haptic_dataset),
            "feedback_prediction_count": len(self._feedback_dataset) if self._feedback_dataset is not None else 0,
            "haptic_failed_command_count": int(
                np.count_nonzero(~self._haptic_dataset[:]["success"])
            ),
        }

        event_rows = self._events_dataset[:]
        if len(event_rows):
            quality["max_abs_event_timing_error_ms"] = float(
                np.max(np.abs(event_rows["timing_error_ms"]))
            )
            quality["max_abs_sample_clock_error_ms"] = float(
                np.max(np.abs(event_rows["sample_clock_error_ms"]))
            )
        else:
            quality["max_abs_event_timing_error_ms"] = 0.0
            quality["max_abs_sample_clock_error_ms"] = 0.0

        for key, value in quality.items():
            self._quality_group.attrs[key] = value
        ended_at_utc = self.ended_at_utc = utc_now_iso()
        self._file.attrs["status"] = status
        self._file.attrs["ended_at_utc"] = ended_at_utc
        self._write_metadata_json(
            status=status,
            ended_at_utc=ended_at_utc,
            termination_reason=reason,
            quality=quality,
        )
        self._file.flush()
        self._file.close()
        self.closed = True
        return quality

    def discard_before_recording(self) -> None:
        if self.closed:
            return
        self._file.close()
        self.closed = True
        self.path.unlink(missing_ok=True)

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("session file is already closed")


def write_session_readme(
    config: SessionConfig,
    plans: list[TrialPlan],
    *,
    status: str,
    quality: dict[str, Any] | None = None,
    termination_reason: str = "",
    stream: EEGStreamConfig | None = None,
) -> Path:
    readme_path = config.output_dir / SESSION_README_FILENAME
    history = read_interruption_history(readme_path)
    if config.self_learning_enabled:
        from self_learning.documentation import write_readme
        path = write_readme(config, status=status, quality=quality, reason=termination_reason, stream=stream)
        if history:
            path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n\n" + history, encoding="utf-8")
        return path
    template_path = Path(__file__).resolve().parent / "templates" / "session_readme.md"
    template = template_path.read_text(encoding="utf-8")
    sequence = " ".join(
        "L" if plan.class_id == 0 else "R" for plan in plans
    )
    quality = quality or {}
    timing = config.protocol
    stream = stream or EEGStreamConfig()
    content = template.format(
        channel_count=timing.get("channel_count", stream.channels),
        channel_labels=", ".join(timing.get("channel_labels", stream.channel_labels)),
        preparation_s=timing["preparation_seconds"],
        rest_s=timing["rest_seconds"],
        cue_end_s=timing["rest_seconds"] + timing["cue_visible_seconds"],
        trial_s=timing["trial_seconds"],
        task_s=timing["motor_imagery_window_seconds"],
        break_instruction_s=timing["break_instruction_seconds"],
        break_countdown_s=timing["break_countdown_seconds"],
        break_chime_s=timing["break_chime_remaining_seconds"],
        experiment_mode=config.experiment_mode,
        haptic_protocol=(
            f"本次为 tactile_MI。箭头左侧对应 HapticL，右侧对应 HapticR。"
            f"同侧从 cue 开始震动，到 trial 结束停止，计划持续 {timing['motor_imagery_window_seconds']:g} 秒。"
            f"高档 {config.haptic_intensity:.2f}，低档 {config.haptic_low_intensity:.2f}，"
            f"每 {timing['haptic_stimulation']['step_seconds']:g} 秒切换一次。"
            f"准备、trial 休息和组间休息均不施加实验震动。"
            f"设备检查或组间休息点击震动测试，两侧以高档恒定震动 {timing['haptic_stimulation']['test_seconds']:g} 秒后停止；"
            f"组间测试期间暂缓倒计时，结束后从剩余时间继续。"
            if config.tactile_enabled
            else "本次为 pure_MI，无触觉刺激，不连接马达。"
        ),
        subject_id=config.subject_id,
        experiment_date=config.experiment_date,
        description=config.description or "未填写",
        status=status,
        termination_reason=termination_reason or "无",
        left_trials=config.left_trials,
        right_trials=config.right_trials,
        total_trials=config.total_trials,
        break_every_trials=config.break_every_trials,
        scheduled_breaks=config.scheduled_break_count,
        random_seed=config.random_seed,
        sequence=sequence,
        expected_duration_s=config.expected_duration_seconds,
        expected_samples=int(
            round(config.expected_duration_seconds * SAMPLE_RATE_HZ)
        ),
        recorded_samples=quality.get("recorded_sample_count", "尚未结束"),
        actual_duration_s=quality.get("actual_duration_s", "尚未结束"),
        wall_duration_s=quality.get("wall_duration_s", "尚未结束"),
        coverage=(
            f"{float(quality['coverage']) * 100:.3f}%"
            if "coverage" in quality
            else "尚未结束"
        ),
        malformed_packets=quality.get(
            "malformed_packets_during_session", "尚未结束"
        ),
        sample_time_error_ms=quality.get(
            "sample_count_time_error_ms", "尚未结束"
        ),
    )
    if history:
        content = content.rstrip() + "\n\n" + history
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


HISTORY_MARKER = "<!-- interruption-history -->"


def read_interruption_history(path: Path) -> str:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if HISTORY_MARKER in text:
            return text[text.index(HISTORY_MARKER):]
    return ""


def record_interruption(output_dir: Path, *, trial_id: int, completed: int, reason: str, key: str) -> None:
    """日志只存在 README；同一次中断在续采时不重复登记。"""
    path = output_dir / SESSION_README_FILENAME
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    token = f"<!-- interruption:{key} -->"
    if token in text:
        return
    if HISTORY_MARKER not in text:
        text = text.rstrip() + "\n\n" + HISTORY_MARKER + "\n## 中断记录\n\n"
        text += (
            "续采保留完整 trial 并重做中断 trial，采样索引连续追加。\n"
            "H5 的时间轴拼接有效采集段（wall_time_s 仍包含保留的组间休息），"
            "不包含被删除的尝试、离线间隔和重新调试时间，不能视为不中断的物理连续采集。\n\n"
        )
    reason = " ".join(reason.splitlines())
    text += f"- {key}：第 {trial_id} 个 trial 中断/待继续；已保留 {completed} 个完整 trial。原因：{reason}。 {token}\n"
    path.write_text(text, encoding="utf-8")
