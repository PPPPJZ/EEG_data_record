"""运动想象实验的配置、随机序列和确定性时间计划。"""

from __future__ import annotations

import math
import random
import secrets
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from config import (
    CLASS_LEFT,
    CLASS_NAMES,
    CLASS_RIGHT,
    BREAK_CHIME_REMAINING_SECONDS,
    BREAK_COUNTDOWN_SECONDS,
    BREAK_INSTRUCTION_SECONDS,
    DEFAULT_BREAK_EVERY_TRIALS,
    CUE_SECONDS,
    IMAGERY_BLACK_SECONDS,
    REST_SECONDS,
    TASK_SECONDS,
    TRIAL_SECONDS,
    DEFAULT_HAPTIC_INTENSITY,
    HAPTIC_DEVICE_NAMES,
    HAPTIC_INTENSITY_DIFFERENCE,
    HAPTIC_LEVEL_UUID,
    HAPTIC_STEP_SECONDS,
    HAPTIC_TEST_SECONDS,
    MODE_PURE_MI,
    MODE_TACTILE_MI,
    MODE_SELF_LEARNING,
    CLASS_NONE,
    DEFAULT_LEARNING_TRIALS,
    SELF_LEARNING_REST_SECONDS,
    SELF_LEARNING_TASK_SECONDS,
    SELF_LEARNING_TRIAL_SECONDS,
    PREPARATION_SECONDS,
)


@dataclass(frozen=True)
class SessionConfig:
    subject_id: str
    experiment_date: str
    description: str
    left_trials: int
    right_trials: int
    break_every_trials: int
    output_dir: Path
    random_seed: int
    experiment_mode: str = MODE_PURE_MI
    haptic_intensity: float = DEFAULT_HAPTIC_INTENSITY
    learning_trials: int = DEFAULT_LEARNING_TRIALS
    teacher_model_path: str = ""
    saved_protocol: dict | None = None

    @property
    def protocol(self) -> dict:
        return self.saved_protocol if self.saved_protocol is not None else protocol_dict(
            self.break_every_trials, experiment_mode=self.experiment_mode,
            haptic_intensity=self.haptic_intensity,
        )

    @property
    def self_learning_enabled(self) -> bool:
        return self.experiment_mode == MODE_SELF_LEARNING

    @property
    def tactile_enabled(self) -> bool:
        return self.experiment_mode == MODE_TACTILE_MI

    @property
    def haptic_low_intensity(self) -> float:
        if self.saved_protocol is not None and self.tactile_enabled:
            return float(self.protocol["haptic_stimulation"]["low_intensity"])
        return round(max(0.0, self.haptic_intensity - HAPTIC_INTENSITY_DIFFERENCE), 2)

    @property
    def total_trials(self) -> int:
        if self.self_learning_enabled:
            return self.learning_trials
        return self.left_trials + self.right_trials

    @property
    def expected_duration_seconds(self) -> float:
        return self.total_trials * self.protocol["trial_seconds"]

    @property
    def scheduled_break_count(self) -> int:
        return max(0, (self.total_trials - 1) // self.break_every_trials)

    @property
    def minimum_session_duration_seconds(self) -> float:
        break_seconds = self.protocol["break_instruction_seconds"] + self.protocol["break_countdown_seconds"]
        return self.expected_duration_seconds + self.scheduled_break_count * break_seconds

    def as_serializable_dict(self) -> dict:
        result = asdict(self)
        result.pop("saved_protocol", None)  # 协议只存于 protocol/json。
        result["output_dir"] = str(self.output_dir)
        result["total_trials"] = self.total_trials
        result["expected_duration_seconds"] = self.expected_duration_seconds
        result["scheduled_break_count"] = self.scheduled_break_count
        result["minimum_session_duration_seconds"] = (
            self.minimum_session_duration_seconds
        )
        return result


@dataclass(frozen=True)
class TrialPlan:
    trial_id: int
    block_id: int
    class_id: int
    planned_rest_start_s: float
    planned_cue_start_s: float
    planned_cue_end_s: float
    planned_trial_end_s: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, "unknown")


def validate_form_values(
    *,
    subject_id: str,
    experiment_date: str,
    description: str,
    left_trials: int,
    right_trials: int,
    break_every_trials: int,
    output_dir: str,
) -> tuple[str, str, str, int, int, int, Path]:
    clean_subject_id = subject_id.strip()
    if not clean_subject_id:
        raise ValueError("受试者编号不能为空")
    if len(clean_subject_id) > 80:
        raise ValueError("受试者编号不能超过 80 个字符")

    try:
        date.fromisoformat(experiment_date)
    except ValueError as exc:
        raise ValueError("实验日期必须是 YYYY-MM-DD 格式") from exc

    if left_trials < 0 or right_trials < 0:
        raise ValueError("左右手 trial 数不能为负数")
    if left_trials + right_trials <= 0:
        raise ValueError("至少需要设置一个 trial")
    if left_trials + right_trials > 10_000:
        raise ValueError("单次实验的 trial 总数不能超过 10000")
    if break_every_trials <= 0:
        raise ValueError("组间休息间隔 N 必须是正整数")
    if break_every_trials > 10_000:
        raise ValueError("组间休息间隔 N 不能超过 10000")

    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        raise ValueError("实验数据记录文件夹必须使用绝对路径")

    return (
        clean_subject_id,
        experiment_date,
        description.strip(),
        int(left_trials),
        int(right_trials),
        int(break_every_trials),
        output_path.resolve(),
    )


def create_session_config(
    *,
    subject_id: str,
    experiment_date: str,
    description: str,
    left_trials: int,
    right_trials: int,
    break_every_trials: int = DEFAULT_BREAK_EVERY_TRIALS,
    output_dir: str,
    random_seed: int | None = None,
    experiment_mode: str = MODE_PURE_MI,
    haptic_intensity: float = DEFAULT_HAPTIC_INTENSITY,
    learning_trials: int = DEFAULT_LEARNING_TRIALS,
    teacher_model_path: str = "",
    mock_teacher: bool = False,
) -> SessionConfig:
    if experiment_mode not in {MODE_PURE_MI, MODE_TACTILE_MI, MODE_SELF_LEARNING}:
        raise ValueError("不支持的实验模式")
    learning = experiment_mode == MODE_SELF_LEARNING
    if learning:
        if not 1 <= learning_trials <= 10_000:
            raise ValueError("自主学习 trial 数必须在 1–10000")
        if not mock_teacher:
            teacher = Path(teacher_model_path.strip()).expanduser()
            if not teacher_model_path.strip() or not teacher.is_absolute() or not teacher.is_file():
                raise ValueError("请输入存在的 teacher 模型文件绝对路径")
            teacher_model_path = str(teacher.resolve())
    if not math.isfinite(haptic_intensity) or not 0.0 <= haptic_intensity <= 1.0:
        raise ValueError("震动强度必须是 0 到 1 之间的有限数值")
    (
        clean_subject_id,
        clean_date,
        clean_description,
        clean_left,
        clean_right,
        clean_break_every,
        clean_output,
    ) = validate_form_values(
        subject_id=subject_id,
        experiment_date=experiment_date,
        description=description,
        left_trials=learning_trials if learning else left_trials,
        right_trials=0 if learning else right_trials,
        break_every_trials=break_every_trials,
        output_dir=output_dir,
    )
    return SessionConfig(
        subject_id=clean_subject_id,
        experiment_date=clean_date,
        description=clean_description,
        left_trials=0 if learning else clean_left,
        right_trials=0 if learning else clean_right,
        break_every_trials=clean_break_every,
        output_dir=clean_output,
        random_seed=secrets.randbits(63) if random_seed is None else random_seed,
        experiment_mode=experiment_mode,
        haptic_intensity=round(haptic_intensity * 100) / 100,
        learning_trials=learning_trials,
        teacher_model_path=teacher_model_path if learning else "",
    )


def build_trial_sequence(config: SessionConfig) -> list[int]:
    if config.self_learning_enabled:
        return [CLASS_NONE] * config.total_trials
    sequence = [CLASS_LEFT] * config.left_trials
    sequence.extend([CLASS_RIGHT] * config.right_trials)
    random.Random(config.random_seed).shuffle(sequence)
    return sequence


def build_trial_plans(
    sequence: list[int],
    break_every_trials: int = DEFAULT_BREAK_EVERY_TRIALS,
    *,
    experiment_mode: str = MODE_PURE_MI,
) -> list[TrialPlan]:
    if break_every_trials <= 0:
        raise ValueError("break_every_trials must be positive")
    plans: list[TrialPlan] = []
    learning = experiment_mode == MODE_SELF_LEARNING
    for zero_based_index, class_id in enumerate(sequence):
        if class_id not in ({CLASS_NONE} if learning else {CLASS_LEFT, CLASS_RIGHT}):
            raise ValueError(f"unsupported class id: {class_id}")
        rest_start = zero_based_index * (SELF_LEARNING_TRIAL_SECONDS if learning else TRIAL_SECONDS)
        cue_start = rest_start + (SELF_LEARNING_REST_SECONDS if learning else REST_SECONDS)
        cue_end = cue_start if learning else cue_start + CUE_SECONDS
        plans.append(
            TrialPlan(
                trial_id=zero_based_index + 1,
                block_id=zero_based_index // break_every_trials + 1,
                class_id=class_id,
                planned_rest_start_s=rest_start,
                planned_cue_start_s=cue_start,
                planned_cue_end_s=cue_end,
                planned_trial_end_s=(
                    cue_start + SELF_LEARNING_TASK_SECONDS if learning
                    else cue_end + IMAGERY_BLACK_SECONDS
                ),
            )
        )
    return plans


def protocol_dict(
    break_every_trials: int = DEFAULT_BREAK_EVERY_TRIALS,
    *,
    experiment_mode: str = MODE_PURE_MI,
    haptic_intensity: float = DEFAULT_HAPTIC_INTENSITY,
) -> dict:
    tactile = experiment_mode == MODE_TACTILE_MI
    learning = experiment_mode == MODE_SELF_LEARNING
    return {
        "experiment_mode": experiment_mode,
        "instruction": "none; subject chooses privately" if learning else "left/right visual cue",
        "trial_class_id": "-1 unknown; teacher predictions are not labels" if learning else "instruction",
        "haptic_stimulation": {
            "enabled": tactile,
            "motor_names": {str(key): value for key, value in HAPTIC_DEVICE_NAMES.items()},
            "level_uuid": HAPTIC_LEVEL_UUID,
            "high_intensity": haptic_intensity if tactile else None,
            "low_intensity": (
                round(max(0.0, haptic_intensity - HAPTIC_INTENSITY_DIFFERENCE), 2)
                if tactile else None
            ),
            "step_seconds": HAPTIC_STEP_SECONDS,
            "full_cycle_seconds": 2 * HAPTIC_STEP_SECONDS,
            "window": "cue_start inclusive to trial_end exclusive",
            "side": "ipsilateral to visual instruction",
            "test_seconds": HAPTIC_TEST_SECONDS,
            "test_pattern": "both motors at constant high intensity",
            "break_test_behavior": "freeze countdown during test; then resume",
            "synchronization": "software scheduled; BLE acknowledgement is not physical onset",
        },
        "rest_seconds": SELF_LEARNING_REST_SECONDS if learning else REST_SECONDS,
        "cue_visible_seconds": 0 if learning else CUE_SECONDS,
        "imagery_black_seconds": 0 if learning else IMAGERY_BLACK_SECONDS,
        "motor_imagery_window_seconds": SELF_LEARNING_TASK_SECONDS if learning else TASK_SECONDS,
        "trial_seconds": SELF_LEARNING_TRIAL_SECONDS if learning else TRIAL_SECONDS,
        "preparation_seconds": PREPARATION_SECONDS,
        "break_every_trials": break_every_trials,
        "break_instruction_seconds": BREAK_INSTRUCTION_SECONDS,
        "break_countdown_seconds": BREAK_COUNTDOWN_SECONDS,
        "break_chime_remaining_seconds": BREAK_CHIME_REMAINING_SECONDS,
        "break_eeg_recording": False,
        "break_pause_behavior": (
            "pause freezes countdown; continue restarts a full countdown"
        ),
        "motor_imagery_window": "cue_start inclusive to trial_end exclusive",
        "sample_slices": {
            "whole_trial": "[rest_start_sample:trial_end_sample]",
            "motor_imagery": "[cue_start_sample:trial_end_sample]",
            "arrow_visible": "[cue_start_sample:cue_end_sample]",
            "imagery_black": "[cue_end_sample:trial_end_sample]",
        },
    }
