"""只读解析续采文件，并计算可保留的完整 trial 前缀。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import h5py
import numpy as np

import config as constants
from experiment import SessionConfig, TrialPlan
from storage import EVENT_DTYPE, TRIAL_DTYPE, HAPTIC_COMMAND_DTYPE, FEEDBACK_DTYPE


def file_stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class RecoveryCut:
    completed: int
    samples: int
    events: int
    active_seconds: float
    wall_seconds: float
    break_completed: bool


def recovery_cut(trials: np.ndarray, events: np.ndarray) -> RecoveryCut:
    """仅保留完整 trial；若其后的组间休息已结束，也保留这一整段休息。"""
    completed = int(np.count_nonzero(trials["completed"]))
    if not np.all(trials["completed"][:completed]) or np.any(trials["completed"][completed:]):
        raise ValueError("已完成 trial 不是连续前缀，无法安全续采")
    if not completed:
        return RecoveryCut(0, 0, 0, 0.0, 0.0, False)
    ends = np.flatnonzero(
        (events["trial_id"] == completed)
        & (events["event_code"] == constants.EVENT_TRIAL_END)
    )
    if len(ends) != 1:
        raise ValueError("完整 trial 缺少唯一结束事件")
    end = int(ends[0])
    row = trials[completed - 1]
    if events[end]["sample_index"] != row["trial_end_sample"]:
        raise ValueError("trial 结束采样索引与事件不一致")
    break_completed = False
    # 下一个 trial 开始前，完整的休息可以保留；中断的休息在续采时重做。
    for index in range(end + 1, len(events)):
        if events[index]["event_code"] == constants.EVENT_TRIAL_START:
            break
        if (events[index]["event_code"] == constants.EVENT_BREAK_END
                and events[index]["trial_id"] == completed):
            end = index
            break_completed = True
    return RecoveryCut(
        completed, int(row["trial_end_sample"]), end + 1,
        float(row["actual_trial_end_s"]), float(events[end]["wall_time_s"]),
        break_completed,
    )


@dataclass
class ResumeInfo:
    path: Path
    stamp: tuple
    config: SessionConfig
    plans: list[TrialPlan]
    metadata: dict
    protocol: dict
    feedback: dict | None
    cut: RecoveryCut

    def check_unchanged(self) -> None:
        if file_stamp(self.path) != self.stamp:
            raise ValueError("文件在解析后发生了变化，请返回 restart_test 重新解析")

    def check_runtime(self, *, demo: bool, time_scale: float, mock_teacher: bool) -> None:
        runtime = self.metadata["runtime"]
        if bool(runtime["demo"]) != demo or not math.isclose(float(runtime["time_scale"]), time_scale):
            raise ValueError("原文件的真实/演示模式或 time_scale 与当前服务不一致")
        if self.feedback and bool(self.feedback["load_info"]["mock"]) != mock_teacher:
            raise ValueError("原实验 Teacher 的真实/模拟模式与当前服务不一致")

    def payload(self) -> dict:
        c, total, n = self.cut.completed, self.config.total_trials, self.config.break_every_trials
        remaining = total - c
        breaks = sum(1 for i in range(c, total) if i > 0 and i % n == 0)
        if c and c % n == 0 and self.cut.break_completed:
            breaks -= 1
        before_break = min(remaining, n - c % n)
        if c and c % n == 0 and not self.cut.break_completed:
            before_break = 0
        seconds = remaining * self.protocol["trial_seconds"] + breaks * (
            self.protocol["break_instruction_seconds"] + self.protocol["break_countdown_seconds"]
        )
        return {
            "path": str(self.path), "subject_id": self.config.subject_id,
            "experiment_date": self.config.experiment_date,
            "description": self.config.description, "experiment_mode": self.config.experiment_mode,
            "total_trials": total, "completed_trials": c, "remaining_trials": remaining,
            "next_trial_id": c + 1, "next_block_id": self.plans[c].block_id,
            "break_every_trials": n, "trials_before_break": before_break,
            "remaining_breaks": breaks, "remaining_seconds": seconds,
            "trial_seconds": self.protocol["trial_seconds"],
            "rest_seconds": self.protocol["rest_seconds"],
            "task_seconds": self.protocol["motor_imagery_window_seconds"],
            "left_trials": self.config.left_trials, "right_trials": self.config.right_trials,
            "remaining_left": sum(p.class_id == 0 for p in self.plans[c:]),
            "remaining_right": sum(p.class_id == 1 for p in self.plans[c:]),
            "teacher_model_path": self.config.teacher_model_path,
            "haptic_intensity": self.config.haptic_intensity,
        }


def parse_resume_path(value: str) -> ResumeInfo:
    """不改动 H5 或 README；所有校验都在允许 r+ 打开之前完成。"""
    text = value.strip().strip('"')
    if not text:
        raise ValueError("请输入实验文件或文件夹的绝对路径")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError("请使用绝对路径")
    if path.is_dir():
        path = path / constants.DATA_FILENAME
    path = path.resolve()
    try:
        stamp = file_stamp(path)
        with h5py.File(path, "r") as root:
            if root.attrs.get("schema_name") != constants.HDF5_SCHEMA_NAME:
                raise ValueError("不是本系统的实验 H5 文件")
            if str(root.attrs.get("schema_version")) not in {"1.1", "1.2", "1.3"}:
                raise ValueError("暂不支持此 H5 版本")
            if root.attrs.get("status") == "complete":
                raise ValueError("该实验已经 complete，无需继续")
            metadata = json.loads(root["metadata/json"][()])
            protocol = json.loads(root["protocol/json"][()])
            # 老文件未记录准备时长，其版本固定为 5 秒。
            protocol.setdefault("preparation_seconds", 5.0)
            names = {f.name for f in fields(SessionConfig)} - {"output_dir", "saved_protocol"}
            config = SessionConfig(
                **{k: v for k, v in metadata.items() if k in names},
                output_dir=path.parent, saved_protocol=protocol,
            )
            if config.experiment_mode not in {"pure_MI", "tactile_MI", "self_learning"}:
                raise ValueError("未知实验模式")
            if not 1 <= config.total_trials <= 10_000 or not 1 <= config.break_every_trials <= 10_000:
                raise ValueError("trial 数量或休息间隔无效")
            if protocol["experiment_mode"] != config.experiment_mode or protocol["break_every_trials"] != config.break_every_trials:
                raise ValueError("实验配置与协议不一致")
            for key in ("rest_seconds", "cue_visible_seconds", "imagery_black_seconds",
                        "motor_imagery_window_seconds", "trial_seconds", "preparation_seconds",
                        "break_instruction_seconds", "break_countdown_seconds", "break_chime_remaining_seconds"):
                if not math.isfinite(protocol[key]) or protocol[key] < 0:
                    raise ValueError(f"无效的实验时长：{key}")
            if protocol["trial_seconds"] <= 0 or protocol["motor_imagery_window_seconds"] <= 0:
                raise ValueError("trial 或任务时长必须大于零")
            if not math.isclose(protocol["rest_seconds"] + protocol["motor_imagery_window_seconds"], protocol["trial_seconds"]):
                raise ValueError("trial 时长与休息/任务时长不一致")
            if config.self_learning_enabled:
                if protocol["cue_visible_seconds"] != 0 or protocol["imagery_black_seconds"] != 0:
                    raise ValueError("自主学习不应包含指令阶段")
            elif not math.isclose(
                protocol["cue_visible_seconds"] + protocol["imagery_black_seconds"],
                protocol["motor_imagery_window_seconds"],
            ):
                raise ValueError("指令和黑屏时长与任务窗口不一致")
            expected_signal = {
                "sample_rate_hz": constants.SAMPLE_RATE_HZ, "channel_count": constants.CHANNEL_COUNT,
                "channel_labels": list(constants.CHANNEL_LABELS), "signal_unit": "uV",
                "adc_uv_per_lsb": constants.ADC_UV_PER_LSB,
                "filtered_signal": {
                    "notch_hz": constants.NOTCH_HZ, "notch_q": constants.NOTCH_Q,
                    "bandpass_hz": [constants.BANDPASS_LOW_HZ, constants.BANDPASS_HIGH_HZ],
                    "butterworth_order_per_stage": constants.BANDPASS_ORDER,
                    "cascade_count": constants.BANDPASS_CASCADE_COUNT,
                },
            }
            if any(protocol[k] != v for k, v in expected_signal.items()):
                raise ValueError("文件的采样率、通道、单位或显示滤波设置与当前采集程序不兼容")
            for key, dtype in (("trials/table", TRIAL_DTYPE), ("events/table", EVENT_DTYPE),
                               ("haptics/commands", HAPTIC_COMMAND_DTYPE)):
                dataset = root[key]
                if dataset.ndim != 1 or dataset.dtype != dtype:
                    raise ValueError(f"数据表格式不兼容：{key}")
                if key != "trials/table" and dataset.maxshape != (None,):
                    raise ValueError(f"数据表不支持追加：{key}")
            raw, filtered = root["eeg/raw_uv"], root["eeg/filtered_uv"]
            if raw.shape != filtered.shape or raw.ndim != 2 or raw.shape[1] != constants.CHANNEL_COUNT:
                raise ValueError("原始与滤波 EEG 维度不一致")
            for dataset in (raw, filtered):
                if dataset.dtype != np.dtype("float32") or dataset.maxshape != (None, constants.CHANNEL_COUNT):
                    raise ValueError("EEG 格式不支持追加")
            trials, events = root["trials/table"][:], root["events/table"][:]
            if len(trials) != config.total_trials:
                raise ValueError("trial 计划不完整")
            plans = [TrialPlan(**{f.name: row[f.name].item() for f in fields(TrialPlan)}) for row in trials]
            for i, p in enumerate(plans):
                expected = [
                    i * protocol["trial_seconds"],
                    i * protocol["trial_seconds"] + protocol["rest_seconds"],
                    i * protocol["trial_seconds"] + protocol["rest_seconds"] + protocol["cue_visible_seconds"],
                    (i + 1) * protocol["trial_seconds"],
                ]
                if (p.trial_id != i + 1 or p.block_id != i // config.break_every_trials + 1
                        or not np.allclose([p.planned_rest_start_s, p.planned_cue_start_s,
                                            p.planned_cue_end_s, p.planned_trial_end_s], expected, rtol=0, atol=1e-7)):
                    raise ValueError("trial 编号、分组或时序计划不一致")
            classes = trials["class_id"]
            if config.self_learning_enabled:
                if np.any(classes != -1):
                    raise ValueError("自主学习 trial 标签无效")
            elif (np.count_nonzero(classes == 0) != config.left_trials
                  or np.count_nonzero(classes == 1) != config.right_trials):
                raise ValueError("左右手计划与实验配置不一致")
            cut = recovery_cut(trials, events)
            if cut.completed == len(trials):
                raise ValueError("所有 trial 已完成，没有剩余实验")
            previous_end = 0
            for row in trials[:cut.completed]:
                samples = [int(row[k]) for k in ("rest_start_sample", "cue_start_sample", "cue_end_sample", "trial_end_sample")]
                times = [float(row[k]) for k in ("actual_rest_start_s", "actual_cue_start_s", "actual_cue_end_s", "actual_trial_end_s")]
                if (samples != sorted(samples) or samples[0] < previous_end
                        or samples[-1] > len(raw) or samples[-1] <= samples[0]
                        or not np.isfinite(times).all() or times != sorted(times)):
                    raise ValueError("完整 trial 的采样边界或时间损坏")
                previous_end = samples[-1]
            kept = events[:cut.events]
            if (not np.array_equal(events["event_id"], np.arange(len(events)))
                    or np.any(kept["sample_index"] < 0) or np.any(kept["sample_index"] > cut.samples)
                    or not np.isfinite(kept["wall_time_s"]).all()):
                raise ValueError("事件编号或索引损坏")
            haptic = protocol["haptic_stimulation"]
            if config.tactile_enabled:
                if (haptic["motor_names"] != {str(k): v for k, v in constants.HAPTIC_DEVICE_NAMES.items()}
                        or haptic["level_uuid"] != constants.HAPTIC_LEVEL_UUID
                        or haptic["high_intensity"] != config.haptic_intensity
                        or not 0 <= haptic["low_intensity"] <= config.haptic_intensity <= 1
                        or not all(math.isfinite(haptic[k]) and haptic[k] > 0 for k in ("step_seconds", "test_seconds"))):
                    raise ValueError("触觉协议无效或与当前硬件不兼容")
            feedback = None
            if config.self_learning_enabled:
                from self_learning.control import ControlConfig
                feedback = json.loads(root["self_learning/config_json"][()])
                ControlConfig(**feedback["config"])
                load = feedback["load_info"]
                if not load["mock"] and (not load["sha256"] or not Path(config.teacher_model_path).is_file()):
                    raise ValueError("原 Teacher 文件不存在，请先恢复原路径的模型文件")
                if root["self_learning/feedback"].dtype != FEEDBACK_DTYPE or root["self_learning/feedback"].maxshape != (None,):
                    raise ValueError("反馈记录格式不支持追加")
            info = ResumeInfo(path, stamp, config, plans, metadata, protocol, feedback, cut)
            # 必须包含原采集运行方式，不能把演示数据接到真实数据上。
            bool(metadata["runtime"]["demo"])
            float(metadata["runtime"]["time_scale"])
            root["quality"]
        info.check_unchanged()
        return info
    except ValueError:
        raise
    except (OSError, KeyError, TypeError, OverflowError, AttributeError, IndexError) as exc:
        raise ValueError(f"无法解析实验文件，请重新选择有效的 H5：{exc}") from exc
