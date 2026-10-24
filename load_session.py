#!/usr/bin/env python3
"""读取运动想象 HDF5 会话，并可按需导出事件和 trial CSV。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from config import HDF5_SCHEMA_NAME


def _read_json_dataset(dataset: h5py.Dataset) -> dict:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def load_session(path: str | Path) -> dict[str, Any]:
    """一次读取完整会话；EEG 数组的形状为 [samples, channels]。"""

    session_path = Path(path).expanduser().resolve()
    with h5py.File(session_path, "r") as handle:
        schema_name = str(handle.attrs.get("schema_name", ""))
        if schema_name != HDF5_SCHEMA_NAME:
            raise ValueError(
                f"unsupported schema {schema_name!r}; expected {HDF5_SCHEMA_NAME!r}"
            )
        return {
            "path": session_path,
            "status": str(handle.attrs["status"]),
            "schema_version": str(handle.attrs["schema_version"]),
            "metadata": _read_json_dataset(handle["metadata/json"]),
            "protocol": _read_json_dataset(handle["protocol/json"]),
            "raw_uv": handle["eeg/raw_uv"][:],
            "filtered_uv": handle["eeg/filtered_uv"][:] if "eeg/filtered_uv" in handle else None,
            "events": handle["events/table"][:],
            "trials": handle["trials/table"][:],
            "haptic_commands": (
                handle["haptics/commands"][:] if "haptics/commands" in handle
                else np.empty(0, dtype=[])
            ),
            "feedback": (
                handle["self_learning/feedback"][:] if "self_learning/feedback" in handle
                else np.empty(0, dtype=[])
            ),
            "feedback_config": (
                _read_json_dataset(handle["self_learning/config_json"])
                if "self_learning/config_json" in handle else None
            ),
            "quality": {
                key: _python_scalar(value)
                for key, value in handle["quality"].attrs.items()
            },
        }


def extract_trial(
    session: dict[str, Any],
    trial_id: int,
    *,
    signal: str = "raw_uv",
    window: str = "whole_trial",
) -> np.ndarray:
    """按 HDF5 中的实际边界提取一个 trial，返回 [samples, channels]。"""

    if signal not in {"raw_uv", "filtered_uv"}:
        raise ValueError("signal must be raw_uv or filtered_uv")
    if session.get(signal) is None:
        raise ValueError(f"本文件未保存 {signal}，请使用 raw_uv")
    boundary_fields = {
        "whole_trial": ("rest_start_sample", "trial_end_sample"),
        "motor_imagery": ("cue_start_sample", "trial_end_sample"),
        "arrow_visible": ("cue_start_sample", "cue_end_sample"),
        "imagery_black": ("cue_end_sample", "trial_end_sample"),
    }
    if window not in boundary_fields:
        raise ValueError(f"unsupported trial window: {window}")

    trials = session["trials"]
    matches = trials[trials["trial_id"] == trial_id]
    if len(matches) != 1:
        raise ValueError(f"trial {trial_id} does not exist")
    row = matches[0]
    start_field, end_field = boundary_fields[window]
    start = int(row[start_field])
    end = int(row[end_field])
    if start < 0 or end < 0:
        raise ValueError(f"trial {trial_id} is incomplete for window {window}")
    if end > len(session[signal]):
        raise ValueError(f"{signal} 未覆盖此 trial；旧滤波数据只保留历史前缀，请使用 raw_uv")
    return session[signal][start:end]


def export_tables_to_csv(
    session: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """把 HDF5 内的两张索引表导出为便于人工检查的 CSV。"""

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    events_path = target / "events.csv"
    trials_path = target / "trials.csv"
    _write_structured_csv(events_path, session["events"])
    _write_structured_csv(trials_path, session["trials"])
    return events_path, trials_path


def _write_structured_csv(path: Path, table: np.ndarray) -> None:
    field_names = list(table.dtype.names or ())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(field_names)
        for row in table:
            writer.writerow([_python_scalar(row[field]) for field in field_names])


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def print_summary(session: dict[str, Any]) -> None:
    metadata = session["metadata"]
    quality = session["quality"]
    print(f"File: {session['path']}")
    print(f"Status: {session['status']}")
    print(f"Subject: {metadata['subject_id']}")
    print(f"Date: {metadata['experiment_date']}")
    print(f"Mode: {metadata.get('experiment_mode', 'pure_MI')}")
    print(f"EEG shape: {session['raw_uv'].shape}")
    print(f"Events: {len(session['events'])}")
    print(f"Trials: {len(session['trials'])}")
    if "coverage" in quality:
        print(f"Coverage: {float(quality['coverage']) * 100:.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_file", type=Path)
    parser.add_argument(
        "--export-csv",
        type=Path,
        metavar="DIRECTORY",
        help="可选：将 events 和 trials 表导出到指定目录",
    )
    args = parser.parse_args()

    session = load_session(args.session_file)
    print_summary(session)
    if args.export_csv is not None:
        events_path, trials_path = export_tables_to_csv(
            session,
            args.export_csv,
        )
        print(f"Exported: {events_path}")
        print(f"Exported: {trials_path}")


if __name__ == "__main__":
    main()
