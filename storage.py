"""自包含 HDF5 会话文件的流式写入与 README 生成。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from config import (
    ADC_UV_PER_LSB,
    BANDPASS_CASCADE_COUNT,
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    CHANNEL_COUNT,
    CHANNEL_LABELS,
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
from experiment import SessionConfig, TrialPlan, protocol_dict
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
    ) -> None:
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
        self._filtered_pending: list[np.ndarray] = []
        self._pending_sample_count = 0
        self._runtime = {"demo": demo, "time_scale": time_scale}

        self._file = h5py.File(self.path, "x")
        self._create_schema()

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
            **protocol_dict(
                self.config.break_every_trials,
                experiment_mode=self.config.experiment_mode,
                haptic_intensity=self.config.haptic_intensity,
            ),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channel_count": CHANNEL_COUNT,
            "channel_labels": list(CHANNEL_LABELS),
            "signal_unit": "uV",
            "raw_signal": "decoded physical values before software filtering",
            "filtered_signal": {
                "notch_hz": NOTCH_HZ,
                "notch_q": NOTCH_Q,
                "bandpass_hz": [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                "butterworth_order_per_stage": BANDPASS_ORDER,
                "cascade_count": BANDPASS_CASCADE_COUNT,
            },
            "display_limit_uv": DISPLAY_LIMIT_UV,
            "adc_uv_per_lsb": ADC_UV_PER_LSB,
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
            CHANNEL_LABELS,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset_options = {
            "shape": (0, CHANNEL_COUNT),
            "maxshape": (None, CHANNEL_COUNT),
            "chunks": (HDF5_WRITE_BLOCK_SAMPLES, CHANNEL_COUNT),
            "dtype": np.float32,
            "compression": "lzf",
            "shuffle": True,
        }
        self._raw_dataset = eeg_group.create_dataset("raw_uv", **dataset_options)
        self._filtered_dataset = eeg_group.create_dataset(
            "filtered_uv", **dataset_options
        )

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
        self.recording_started_at_utc = utc_now_iso()
        self._file.attrs["status"] = "recording"
        self._file.attrs["recording_started_at_utc"] = self.recording_started_at_utc
        self._write_metadata_json(status="recording")
        self._file.flush()

    def append_samples(
        self,
        raw_uv: np.ndarray,
        filtered_uv: np.ndarray,
    ) -> None:
        self._require_open()
        raw = np.asarray(raw_uv, dtype=np.float32)
        filtered = np.asarray(filtered_uv, dtype=np.float32)
        expected_prefix = (CHANNEL_COUNT,)
        if raw.ndim != 2 or raw.shape[0:1] != expected_prefix:
            raise ValueError(f"raw EEG shape must be ({CHANNEL_COUNT}, samples)")
        if filtered.shape != raw.shape:
            raise ValueError("raw and filtered EEG shapes differ")
        if raw.shape[1] == 0:
            return

        self._raw_pending.append(raw.copy())
        self._filtered_pending.append(filtered.copy())
        self._pending_sample_count += raw.shape[1]
        if self._pending_sample_count >= HDF5_WRITE_BLOCK_SAMPLES:
            self.flush_samples()

    def flush_samples(self) -> None:
        self._require_open()
        if not self._raw_pending:
            return
        raw = np.hstack(self._raw_pending).T
        filtered = np.hstack(self._filtered_pending).T
        start = self._flushed_sample_count
        end = start + raw.shape[0]
        self._raw_dataset.resize((end, CHANNEL_COUNT))
        self._filtered_dataset.resize((end, CHANNEL_COUNT))
        self._raw_dataset[start:end] = raw
        self._filtered_dataset[start:end] = filtered
        self._flushed_sample_count = end
        self._raw_pending.clear()
        self._filtered_pending.clear()
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
        malformed_during_session = max(
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
        ended_at_utc = utc_now_iso()
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
) -> Path:
    template_path = Path(__file__).resolve().parent / "templates" / "session_readme.md"
    template = template_path.read_text(encoding="utf-8")
    sequence = " ".join(
        "L" if plan.class_id == 0 else "R" for plan in plans
    )
    quality = quality or {}
    content = template.format(
        experiment_mode=config.experiment_mode,
        haptic_protocol=(
            f"本次为 tactile_MI。箭头左侧对应 HapticL，右侧对应 HapticR。"
            f"同侧从 cue 开始震动，到 trial 结束停止，计划持续 5.5 秒。"
            f"高档 {config.haptic_intensity:.2f}，低档 {config.haptic_low_intensity:.2f}，"
            f"每 0.25 秒切换一次（完整高低周期 0.5 秒）。"
            f"准备、trial 休息和组间休息均不施加实验震动。"
            f"设备检查或组间休息点击震动测试，两侧以高档恒定震动 2 秒后停止；"
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
    readme_path = config.output_dir / SESSION_README_FILENAME
    readme_path.write_text(content, encoding="utf-8")
    return readme_path
