#!/usr/bin/env python3
"""Motren_E 左右手运动想象实验采集服务。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    BREAK_CHIME_REMAINING_SECONDS,
    BREAK_COUNTDOWN_SECONDS,
    BREAK_INSTRUCTION_SECONDS,
    CHANNEL_COUNT,
    CLASS_NONE,
    CUE_SECONDS,
    DEFAULT_BREAK_EVERY_TRIALS,
    EVENT_BREAK_CHIME,
    EVENT_BREAK_COUNTDOWN_RESTARTED,
    EVENT_BREAK_DEVICE_DISCONNECTED,
    EVENT_BREAK_END,
    EVENT_BREAK_INSTRUCTION_START,
    EVENT_BREAK_MONITOR_START,
    EVENT_BREAK_PAUSED,
    EVENT_BREAK_SIGNAL_RESTORED,
    EVENT_CUE_END,
    EVENT_CUE_LEFT,
    EVENT_CUE_RIGHT,
    EVENT_RECORDING_ABORTED,
    EVENT_RECORDING_COMPLETE,
    EVENT_RECORDING_START,
    EVENT_TRIAL_END,
    EVENT_TRIAL_START,
    IMAGERY_BLACK_SECONDS,
    PREPARATION_SECONDS,
    REST_SECONDS,
    SAMPLE_BROADCAST_HZ,
    STATE_BROADCAST_HZ,
    TASK_SECONDS,
    DEFAULT_HAPTIC_INTENSITY,
    MODE_PURE_MI,
)
from eeg import BLEEEGSource, SimulatedEEGSource
from haptics import HapticCommand, HapticController
from experiment import (
    SessionConfig,
    TrialPlan,
    build_trial_plans,
    build_trial_sequence,
    create_session_config,
)
from storage import (
    SessionWriter,
    ensure_empty_output_dir,
    write_session_readme,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
TRIAL_STAGES = {"preparing", "rest", "cue", "imagery"}
BREAK_STAGES = {"break_instruction", "break_monitor"}
ACTIVE_STAGES = TRIAL_STAGES | BREAK_STAGES


class SessionForm(BaseModel):
    experiment_mode: Literal["pure_MI", "tactile_MI"] = MODE_PURE_MI
    haptic_intensity: float = Field(
        default=DEFAULT_HAPTIC_INTENSITY, ge=0.0, le=1.0, allow_inf_nan=False
    )
    subject_id: str = Field(max_length=80)
    experiment_date: str
    description: str = Field(default="", max_length=4000)
    left_trials: int = Field(default=50, ge=0, le=10_000)
    right_trials: int = Field(default=50, ge=0, le=10_000)
    break_every_trials: int = Field(
        default=DEFAULT_BREAK_EVERY_TRIALS,
        ge=1,
        le=10_000,
    )
    output_dir: str


class ApplicationService:
    def __init__(self, *, demo: bool = False, time_scale: float = 1.0):
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        if not demo and time_scale != 1.0:
            raise ValueError("真实采集必须使用 time_scale=1")
        self.demo = demo
        self.time_scale = time_scale

        self.stage = "information"
        self.config: SessionConfig | None = None
        self.sequence: list[int] = []
        self.plans: list[TrialPlan] = []
        self.session_locked = False
        self.last_outcome = ""
        self.last_message = ""
        self.result_summary: dict | None = None

        self.device_status = "等待受试者信息"
        self.device_connected = False
        self.source: BLEEEGSource | SimulatedEEGSource | None = None
        self._source_task: asyncio.Task | None = None
        self.haptics: HapticController | None = None
        self._scan_lock = asyncio.Lock()

        self.clients: set[WebSocket] = set()
        self.sample_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
        self._background_tasks: list[asyncio.Task] = []
        self._broadcast_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()

        self.writer: SessionWriter | None = None
        self._experiment_task: asyncio.Task | None = None
        self._recording_active = False
        self._recording_t0_wall: float | None = None
        self._active_clock_origin_wall: float | None = None
        self._active_clock_offset_s = 0.0
        self._schedule_origin_wall: float | None = None
        self._trial_index = -1
        self._phase_started_wall: float | None = None
        self._phase_duration_protocol_s = 0.0
        self._task_started_wall: float | None = None
        self._abort_scheduled = False
        self._has_valid_signal = False
        self._break_remaining_s = BREAK_COUNTDOWN_SECONDS
        self._break_paused = False
        self._break_chime_emitted = False
        self._break_chime_serial = 0
        self._break_after_trial_id = -1
        self._break_signal_lost = False
        self._break_requires_manual_continue = False

    async def start(self) -> None:
        self._background_tasks = [
            asyncio.create_task(self._sample_broadcast_loop(), name="sample-broadcast"),
            asyncio.create_task(self._state_broadcast_loop(), name="state-broadcast"),
        ]

    async def stop(self) -> None:
        if self.stage in ACTIVE_STAGES:
            await self.abort_experiment("服务器停止")
        await self._stop_haptics()
        await self._stop_source()
        for task in self._background_tasks:
            task.cancel()
        for task in self._background_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()

    def _malformed_packets(self) -> int:
        return int(getattr(self.source, "malformed_packets", 0))

    async def _start_source(self) -> None:
        if self._source_task is not None and not self._source_task.done():
            return
        if self.demo:
            self.source = SimulatedEEGSource(
                self._on_samples,
                self._on_source_status,
                time_scale=self.time_scale,
            )
        else:
            self.source = BLEEEGSource(
                self._on_samples,
                self._on_source_status,
                scan_lock=self._scan_lock,
            )
        self._source_task = asyncio.create_task(
            self.source.run(),
            name="eeg-source",
        )

    async def _stop_source(self) -> None:
        if self.source is not None:
            self.source.stop()
        if self._source_task is not None:
            self._source_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._source_task
        self.source = None
        self._source_task = None
        self.device_connected = False
        self._has_valid_signal = False

    async def _start_haptics(self) -> None:
        if self.config is not None and self.config.tactile_enabled:
            self.haptics = HapticController(
                self._on_haptic_fault,
                self._on_haptic_command,
                demo=self.demo,
                time_scale=self.time_scale,
                scan_lock=self._scan_lock,
            )
            await self.haptics.start()

    async def _stop_haptics(self) -> None:
        if self.haptics is not None:
            await self.haptics.stop()
            self.haptics = None

    def _haptics_ready(self) -> bool:
        return (
            self.config is None
            or not self.config.tactile_enabled
            or (self.haptics is not None and self.haptics.ready)
        )

    def _on_haptic_fault(self, reason: str) -> None:
        if self.stage in BREAK_STAGES:
            self._break_paused = True
            self._break_requires_manual_continue = True
        elif self.stage in TRIAL_STAGES and not self._abort_scheduled:
            self._recording_active = False
            self._abort_scheduled = True
            asyncio.create_task(
                self.abort_experiment(f"触觉设备异常：{reason}"),
                name="abort-on-haptic-fault",
            )

    def _on_haptic_command(self, command: HapticCommand) -> None:
        if self.writer is None or self._recording_t0_wall is None:
            return
        try:
            self.writer.append_haptic_command(
                command,
                recording_origin=self._recording_t0_wall,
                time_scale=self.time_scale,
            )
        except Exception as exc:
            if not self._abort_scheduled:
                self._abort_scheduled = True
                self._recording_active = False
                asyncio.create_task(
                    self.abort_experiment(f"震动命令记录失败：{exc}"),
                    name="abort-on-haptic-storage-error",
                )

    async def test_haptics(self) -> None:
        async with self._session_lock:
            if self.stage not in {"device_check", "break_monitor"}:
                raise ValueError("只能在设备检查或组间休息时测试震动")
            if self.haptics is None or self.config is None:
                raise ValueError("震动测试仅适用于 tactile_MI")
            if self._abort_scheduled:
                raise ValueError("实验正在停止")
            self.haptics.start_test(self.config.haptic_intensity)
        await self.broadcast({"type": "state", **self.state_payload()})

    async def stop_haptic_test(self) -> None:
        async with self._session_lock:
            if self.haptics is not None:
                await self.haptics.stop_stimulation()
        await self.broadcast({"type": "state", **self.state_payload()})

    def _on_source_status(self, status: str, connected: bool) -> None:
        connection_lost = self.device_connected and not connected
        self.device_status = status
        self.device_connected = connected
        if not connected:
            self._has_valid_signal = False

        if connection_lost and self.stage in BREAK_STAGES:
            self._break_paused = True
            self._break_signal_lost = True
            self._break_requires_manual_continue = True
            self._record_break_event(EVENT_BREAK_DEVICE_DISCONNECTED)
        elif (
            connection_lost
            and self.stage in TRIAL_STAGES
            and not self._abort_scheduled
        ):
            self._recording_active = False
            self._abort_scheduled = True
            asyncio.get_running_loop().create_task(
                self.abort_experiment("正式采集期间蓝牙连接断开"),
                name="abort-on-bluetooth-loss",
            )

    def _on_samples(
        self,
        raw_uv: np.ndarray,
        filtered_uv: np.ndarray,
    ) -> None:
        signal_was_unavailable = not self._has_valid_signal
        if self.device_connected:
            self._has_valid_signal = True
        if (
            signal_was_unavailable
            and self._has_valid_signal
            and self._break_signal_lost
            and self.stage in BREAK_STAGES
            and self.device_connected
        ):
            self._break_signal_lost = False
            self._record_break_event(EVENT_BREAK_SIGNAL_RESTORED)

        if self._recording_active and self.writer is not None:
            try:
                self.writer.append_samples(raw_uv, filtered_uv)
            except Exception as exc:  # noqa: BLE001 - 落盘失败必须终止实验
                if not self._abort_scheduled:
                    self._abort_scheduled = True
                    asyncio.get_running_loop().create_task(
                        self.abort_experiment(f"数据写入失败：{exc}"),
                        name="abort-on-storage-error",
                    )

        if self.sample_queue.full():
            with suppress(asyncio.QueueEmpty):
                self.sample_queue.get_nowait()
        with suppress(asyncio.QueueFull):
            self.sample_queue.put_nowait(filtered_uv)

    async def configure_session(self, form: SessionForm) -> dict:
        async with self._session_lock:
            if self.stage in ACTIVE_STAGES:
                raise ValueError("实验正在运行，不能修改受试者信息")
            form_values = (
                form.model_dump()
                if hasattr(form, "model_dump")
                else form.dict()
            )
            config = create_session_config(**form_values)
            ensure_empty_output_dir(config.output_dir)
            await self._stop_haptics()
            self.config = config
            self.sequence = build_trial_sequence(config)
            self.plans = build_trial_plans(
                self.sequence,
                config.break_every_trials,
            )
            self.session_locked = False
            self.last_outcome = ""
            self.last_message = ""
            self.result_summary = None
            self.stage = "device_check"
            self.device_status = "正在启动采集服务"
            await self._start_source()
            await self._start_haptics()
        await self.broadcast({"type": "state", **self.state_payload()})
        return {
            "ok": True,
            "total_trials": config.total_trials,
            "random_seed": config.random_seed,
        }

    async def new_session(self) -> None:
        async with self._session_lock:
            if self.stage in ACTIVE_STAGES:
                raise ValueError("请先终止当前实验")
            await self._stop_haptics()
            await self._stop_source()
            self.config = None
            self.sequence = []
            self.plans = []
            self.session_locked = False
            self.last_outcome = ""
            self.last_message = ""
            self.result_summary = None
            self.stage = "information"
            self.device_status = "等待受试者信息"
            self._clear_sample_queue()
        await self.broadcast({"type": "state", **self.state_payload()})

    async def start_experiment(self) -> None:
        async with self._session_lock:
            if self.stage != "device_check":
                raise ValueError("当前状态不能开始采集")
            if self.config is None or not self.plans:
                raise ValueError("缺少有效的实验配置")
            if self.session_locked:
                raise ValueError("当前文件夹已有实验数据，请新建实验")
            if not self.device_connected:
                raise ValueError("脑电设备尚未连接")
            if not self._haptics_ready():
                raise ValueError("两侧马达尚未全部连接")
            if self.haptics is not None and self.haptics.busy:
                raise ValueError("请等待震动测试结束后开始采集")

            ensure_empty_output_dir(self.config.output_dir)
            try:
                self.writer = SessionWriter(
                    self.config,
                    self.plans,
                    malformed_packets_at_start=self._malformed_packets(),
                    demo=self.demo,
                    time_scale=self.time_scale,
                )
                write_session_readme(
                    self.config,
                    self.plans,
                    status="preparing",
                )
            except Exception:
                if self.writer is not None:
                    self.writer.discard_before_recording()
                    self.writer = None
                (self.config.output_dir / "README.md").unlink(missing_ok=True)
                raise
            self.last_outcome = ""
            self.last_message = ""
            self.result_summary = None
            self._trial_index = -1
            self._recording_active = False
            self._recording_t0_wall = None
            self._active_clock_origin_wall = None
            self._active_clock_offset_s = 0.0
            self._schedule_origin_wall = None
            self._break_remaining_s = BREAK_COUNTDOWN_SECONDS
            self._break_paused = False
            self._break_chime_emitted = False
            self._break_chime_serial = 0
            self._break_after_trial_id = -1
            self._break_signal_lost = False
            self._break_requires_manual_continue = False
            self._set_phase("preparing", PREPARATION_SECONDS)
            self._experiment_task = asyncio.create_task(
                self._run_experiment(),
                name="motor-imagery-experiment",
            )
        await self.broadcast({"type": "state", **self.state_payload()})

    def _set_phase(self, stage: str, duration_protocol_s: float) -> None:
        self.stage = stage
        self._phase_started_wall = asyncio.get_running_loop().time()
        self._phase_duration_protocol_s = duration_protocol_s

    async def _sleep_until(self, deadline_wall: float) -> None:
        while True:
            remaining = deadline_wall - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.25))

    def _active_elapsed(self) -> float:
        elapsed = self._active_clock_offset_s
        if self._active_clock_origin_wall is not None:
            elapsed += (
                asyncio.get_running_loop().time() - self._active_clock_origin_wall
            ) / self.time_scale
        return max(0.0, elapsed)

    def _wall_elapsed(self) -> float:
        if self._recording_t0_wall is None:
            return 0.0
        return max(
            0.0,
            (
                asyncio.get_running_loop().time() - self._recording_t0_wall
            )
            / self.time_scale,
        )

    def _pause_active_clock(self) -> None:
        if self._active_clock_origin_wall is None:
            return
        self._active_clock_offset_s = self._active_elapsed()
        self._active_clock_origin_wall = None

    def _resume_active_clock(self) -> None:
        if self._active_clock_origin_wall is None:
            self._active_clock_origin_wall = asyncio.get_running_loop().time()

    def _record_event(
        self,
        *,
        event_code: int,
        trial_id: int,
        class_id: int,
        planned_time_s: float,
        boundary: str | None = None,
    ) -> float:
        if self.writer is None or self._recording_t0_wall is None:
            raise RuntimeError("recording writer is unavailable")
        actual_time_s = self._active_elapsed()
        sample_index = self.writer.append_event(
            event_code=event_code,
            trial_id=trial_id,
            class_id=class_id,
            planned_time_s=planned_time_s,
            actual_time_s=actual_time_s,
            wall_time_s=self._wall_elapsed(),
        )
        if boundary is not None:
            self.writer.mark_trial_boundary(
                trial_id=trial_id,
                boundary=boundary,
                actual_time_s=actual_time_s,
                sample_index=sample_index,
            )
        return actual_time_s

    def _record_break_event(self, event_code: int) -> None:
        if (
            self.writer is None
            or self._recording_t0_wall is None
            or self._break_after_trial_id < 1
        ):
            return
        boundary_time = self.plans[
            self._break_after_trial_id - 1
        ].planned_trial_end_s
        self._record_event(
            event_code=event_code,
            trial_id=self._break_after_trial_id,
            class_id=CLASS_NONE,
            planned_time_s=boundary_time,
        )

    async def toggle_break_countdown(self) -> None:
        async with self._session_lock:
            if self.stage != "break_monitor":
                raise ValueError("当前不在组间休息监控阶段")
            if self._break_paused:
                if (
                    not self.device_connected or not self._has_valid_signal
                    or not self._haptics_ready()
                ):
                    raise ValueError("EEG 信号或触觉设备尚未就绪，不能继续")
                if self.haptics is not None and self.haptics.busy:
                    raise ValueError("请等待震动测试结束后继续")
                self._break_remaining_s = BREAK_COUNTDOWN_SECONDS
                self._break_chime_emitted = False
                self._break_paused = False
                self._break_requires_manual_continue = False
                self._record_break_event(EVENT_BREAK_COUNTDOWN_RESTARTED)
            else:
                self._break_paused = True
                self._record_break_event(EVENT_BREAK_PAUSED)
        await self.broadcast({"type": "state", **self.state_payload()})

    async def _run_interblock_break(
        self,
        completed_plan: TrialPlan,
        next_plan: TrialPlan,
    ) -> None:
        loop = asyncio.get_running_loop()
        self._recording_active = False
        self._pause_active_clock()
        if self.writer is not None:
            self.writer.flush_samples()

        self._break_after_trial_id = completed_plan.trial_id
        self._set_phase("break_instruction", BREAK_INSTRUCTION_SECONDS)
        self._record_break_event(EVENT_BREAK_INSTRUCTION_START)
        await self._sleep_until(
            loop.time() + BREAK_INSTRUCTION_SECONDS * self.time_scale
        )

        self._set_phase("break_monitor", BREAK_COUNTDOWN_SECONDS)
        self._break_remaining_s = BREAK_COUNTDOWN_SECONDS
        self._break_chime_emitted = False
        self._break_paused = (
            self._break_requires_manual_continue
            or not (
                self.device_connected and self._has_valid_signal
                and self._haptics_ready()
            )
        )
        self._record_break_event(EVENT_BREAK_MONITOR_START)

        last_update = loop.time()
        while self._break_remaining_s > 0:
            await asyncio.sleep(
                min(0.05, max(0.002, 0.05 * self.time_scale))
            )
            now = loop.time()
            if not self._break_paused and not (
                self.haptics is not None and self.haptics.busy
            ):
                elapsed = (now - last_update) / self.time_scale
                previous_remaining = self._break_remaining_s
                self._break_remaining_s = max(
                    0.0,
                    self._break_remaining_s - elapsed,
                )
                if (
                    not self._break_chime_emitted
                    and previous_remaining > BREAK_CHIME_REMAINING_SECONDS
                    and self._break_remaining_s
                    <= BREAK_CHIME_REMAINING_SECONDS
                ):
                    self._break_chime_emitted = True
                    self._break_chime_serial += 1
                    self._record_break_event(EVENT_BREAK_CHIME)
            last_update = now

        self._record_break_event(EVENT_BREAK_END)
        self._break_paused = False
        self._break_after_trial_id = -1
        self._break_requires_manual_continue = False
        self._resume_active_clock()
        self._schedule_origin_wall = (
            loop.time()
            - next_plan.planned_rest_start_s * self.time_scale
        )

    async def _run_experiment(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            prep_deadline = loop.time() + PREPARATION_SECONDS * self.time_scale
            await self._sleep_until(prep_deadline)
            if (
                not self.device_connected or not self._has_valid_signal
                or not self._haptics_ready()
            ):
                await self._abort_from_experiment_task(
                    "准备阶段结束时 EEG 信号或触觉设备未就绪"
                )
                return
            if self.writer is None:
                raise RuntimeError("session writer was not created")

            self._recording_t0_wall = loop.time()
            self._active_clock_origin_wall = self._recording_t0_wall
            self._active_clock_offset_s = 0.0
            self._schedule_origin_wall = self._recording_t0_wall
            self.writer.begin_recording()
            self._recording_active = False
            self._record_event(
                event_code=EVENT_RECORDING_START,
                trial_id=-1,
                class_id=CLASS_NONE,
                planned_time_s=0.0,
            )

            for index, plan in enumerate(self.plans):
                if (
                    index > 0
                    and index % self.config.break_every_trials == 0
                ):
                    await self._run_interblock_break(
                        self.plans[index - 1],
                        plan,
                    )

                self._trial_index = index
                if self._schedule_origin_wall is None:
                    raise RuntimeError("experiment schedule clock is unavailable")
                trial_start_deadline = (
                    self._schedule_origin_wall
                    + plan.planned_rest_start_s * self.time_scale
                )
                await self._sleep_until(trial_start_deadline)
                self._set_phase("rest", REST_SECONDS)
                self._record_event(
                    event_code=EVENT_TRIAL_START,
                    trial_id=plan.trial_id,
                    class_id=plan.class_id,
                    planned_time_s=plan.planned_rest_start_s,
                    boundary="rest_start",
                )
                self._recording_active = True

                cue_deadline = (
                    self._schedule_origin_wall
                    + plan.planned_cue_start_s * self.time_scale
                )
                await self._sleep_until(cue_deadline)
                self._set_phase("cue", CUE_SECONDS)
                self._task_started_wall = loop.time()
                trial_end_deadline = (
                    self._schedule_origin_wall
                    + plan.planned_trial_end_s * self.time_scale
                )
                if self.haptics is not None:
                    self.haptics.start_trial(
                        plan.class_id,
                        self.config.haptic_intensity,
                        trial_id=plan.trial_id,
                        beginning=self._task_started_wall,
                        deadline=trial_end_deadline,
                    )
                self._record_event(
                    event_code=(
                        EVENT_CUE_LEFT
                        if plan.class_id == 0
                        else EVENT_CUE_RIGHT
                    ),
                    trial_id=plan.trial_id,
                    class_id=plan.class_id,
                    planned_time_s=plan.planned_cue_start_s,
                    boundary="cue_start",
                )
                if self.haptics is not None:
                    await self.broadcast({"type": "state", **self.state_payload()})

                cue_end_deadline = (
                    self._schedule_origin_wall
                    + plan.planned_cue_end_s * self.time_scale
                )
                await self._sleep_until(cue_end_deadline)
                self._set_phase("imagery", IMAGERY_BLACK_SECONDS)
                self._record_event(
                    event_code=EVENT_CUE_END,
                    trial_id=plan.trial_id,
                    class_id=plan.class_id,
                    planned_time_s=plan.planned_cue_end_s,
                    boundary="cue_end",
                )

                trial_end_deadline = (
                    self._schedule_origin_wall
                    + plan.planned_trial_end_s * self.time_scale
                )
                await self._sleep_until(trial_end_deadline)
                self._record_event(
                    event_code=EVENT_TRIAL_END,
                    trial_id=plan.trial_id,
                    class_id=plan.class_id,
                    planned_time_s=plan.planned_trial_end_s,
                    boundary="trial_end",
                )
                if (
                    (index + 1) % self.config.break_every_trials == 0
                    and index + 1 < len(self.plans)
                ):
                    self._recording_active = False
                if self.haptics is not None:
                    await self.haptics.stop_stimulation()

            async with self._session_lock:
                if self.writer is None:
                    return
                actual_duration = self._record_event(
                    event_code=EVENT_RECORDING_COMPLETE,
                    trial_id=-1,
                    class_id=CLASS_NONE,
                    planned_time_s=self.config.expected_duration_seconds,
                )
                self._recording_active = False
                self._pause_active_clock()
                quality = self.writer.finalize(
                    status="complete",
                    actual_duration_s=actual_duration,
                    wall_duration_s=self._wall_elapsed(),
                    malformed_packets_at_end=self._malformed_packets(),
                )
                write_session_readme(
                    self.config,
                    self.plans,
                    status="complete",
                    quality=quality,
                )
                self.writer = None
                self.session_locked = True
                self.stage = "complete"
                self.last_outcome = "complete"
                self.last_message = "实验已完成，数据已安全保存"
                self.result_summary = quality
                self._experiment_task = None
                self._recording_t0_wall = None
                self._active_clock_origin_wall = None
                self._active_clock_offset_s = 0.0
                self._schedule_origin_wall = None
                self._task_started_wall = None
                self._abort_scheduled = False
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - 任意运行错误都要安全结束文件
            await self._abort_from_experiment_task(f"实验运行错误：{exc}")

    async def _abort_from_experiment_task(self, reason: str) -> None:
        await self._finalize_abort(reason, cancel_task=False)

    async def abort_experiment(self, reason: str = "操作员按下 Esc") -> None:
        await self._finalize_abort(reason, cancel_task=True)

    async def _finalize_abort(self, reason: str, *, cancel_task: bool) -> None:
        async with self._session_lock:
            if self.stage not in ACTIVE_STAGES and self.writer is None:
                self._abort_scheduled = False
                return
            task = self._experiment_task
            if cancel_task and task is not None and task is not asyncio.current_task():
                task.cancel()

            was_recording = self._recording_t0_wall is not None
            self._recording_active = False
            if self.haptics is not None:
                await self.haptics.stop_stimulation()
            if was_recording:
                self._pause_active_clock()
            quality: dict | None = None
            if self.writer is not None:
                if was_recording:
                    actual_duration = self._active_elapsed()
                    wall_duration = self._wall_elapsed()
                    self._record_event(
                        event_code=EVENT_RECORDING_ABORTED,
                        trial_id=(
                            self.plans[self._trial_index].trial_id
                            if 0 <= self._trial_index < len(self.plans)
                            else -1
                        ),
                        class_id=(
                            self.plans[self._trial_index].class_id
                            if 0 <= self._trial_index < len(self.plans)
                            else CLASS_NONE
                        ),
                        planned_time_s=actual_duration,
                    )
                    quality = self.writer.finalize(
                        status="incomplete",
                        actual_duration_s=actual_duration,
                        wall_duration_s=wall_duration,
                        malformed_packets_at_end=self._malformed_packets(),
                        reason=reason,
                    )
                    write_session_readme(
                        self.config,
                        self.plans,
                        status="incomplete",
                        quality=quality,
                        termination_reason=reason,
                    )
                else:
                    self.writer.discard_before_recording()
                    if self.config is not None:
                        (
                            self.config.output_dir / "README.md"
                        ).unlink(missing_ok=True)
                self.writer = None

            self.session_locked = was_recording
            self.stage = "device_check"
            self.last_outcome = "incomplete" if was_recording else "cancelled"
            self.last_message = (
                f"实验已中断并保留不完整数据：{reason}"
                if was_recording
                else f"准备已取消，未保存 EEG 数据：{reason}"
            )
            self.result_summary = quality
            self._experiment_task = None
            self._recording_t0_wall = None
            self._active_clock_origin_wall = None
            self._active_clock_offset_s = 0.0
            self._schedule_origin_wall = None
            self._task_started_wall = None
            self._break_paused = False
            self._break_after_trial_id = -1
            self._break_signal_lost = False
            self._break_requires_manual_continue = False
            self._abort_scheduled = False
        await self.broadcast({"type": "state", **self.state_payload()})

    def _clear_sample_queue(self) -> None:
        while not self.sample_queue.empty():
            with suppress(asyncio.QueueEmpty):
                self.sample_queue.get_nowait()

    def _phase_progress(self) -> float:
        if self._phase_started_wall is None or self._phase_duration_protocol_s <= 0:
            return 0.0
        elapsed_protocol = (
            asyncio.get_running_loop().time() - self._phase_started_wall
        ) / self.time_scale
        return min(1.0, max(0.0, elapsed_protocol / self._phase_duration_protocol_s))

    def _task_progress(self) -> float:
        if self._task_started_wall is None:
            return 0.0

        elapsed_protocol = (
            asyncio.get_running_loop().time() - self._task_started_wall
        ) / self.time_scale
        return min(1.0, max(0.0, elapsed_protocol / TASK_SECONDS))

    def state_payload(self) -> dict:
        config = self.config
        payload = {
            "stage": self.stage,
            "configured": config is not None,
            "device_status": self.device_status,
            "device_connected": self.device_connected,
            "malformed_packets": self._malformed_packets(),
            "session_locked": self.session_locked,
            "can_start": (
                self.stage == "device_check"
                and config is not None
                and self.device_connected
                and self._has_valid_signal
                and self._haptics_ready()
                and not (self.haptics is not None and self.haptics.busy)
                and not self.session_locked
            ),
            "last_outcome": self.last_outcome,
            "last_message": self.last_message,
            "demo": self.demo,
            "result_summary": self.result_summary,
            "output_dir": str(config.output_dir) if config else "",
            "subject_id": config.subject_id if config else "",
            "experiment_mode": config.experiment_mode if config else None,
            "haptic_intensity": config.haptic_intensity if config else DEFAULT_HAPTIC_INTENSITY,
            "haptics": self.haptics.state_payload() if self.haptics else None,
            "can_test_haptics": (
                self.stage in {"device_check", "break_monitor"}
                and self.haptics is not None
                and self.haptics.ready
                and not self.haptics.busy
                and not self._abort_scheduled
            ),
            "total_trials": config.total_trials if config else 0,
            "trial_id": (
                self.plans[self._trial_index].trial_id
                if 0 <= self._trial_index < len(self.plans)
                else 0
            ),
            "remaining_trials": (
                0
                if self.stage == "complete"
                else (
                    len(self.plans) - self._trial_index
                    if 0 <= self._trial_index < len(self.plans)
                    else len(self.plans)
                )
            ),
            "cue_direction": (
                self.plans[self._trial_index].class_name
                if 0 <= self._trial_index < len(self.plans)
                else ""
            ),
            "progress": 0.0,
            "preparation_remaining": None,
            "break_every_trials": (
                config.break_every_trials
                if config
                else DEFAULT_BREAK_EVERY_TRIALS
            ),
            "scheduled_breaks": config.scheduled_break_count if config else 0,
            "break_remaining": (
                max(0, math.ceil(self._break_remaining_s))
                if self.stage == "break_monitor"
                else None
            ),
            "break_paused": self._break_paused,
            "break_can_continue": (
                self.stage == "break_monitor"
                and self._break_paused
                and self.device_connected
                and self._has_valid_signal
                and self._haptics_ready()
                and not (self.haptics is not None and self.haptics.busy)
            ),
            "break_chime_serial": self._break_chime_serial,
            "completed_trials": max(0, self._trial_index + 1),
        }

        if self.stage == "preparing":
            payload["progress"] = self._phase_progress()
            elapsed = (
                (
                    asyncio.get_running_loop().time()
                    - (self._phase_started_wall or asyncio.get_running_loop().time())
                )
                / self.time_scale
            )
            payload["preparation_remaining"] = max(
                0, math.ceil(PREPARATION_SECONDS - elapsed)
            )
        elif self.stage == "rest":
            payload["progress"] = self._phase_progress()
        elif self.stage in {"cue", "imagery"}:
            payload["progress"] = self._task_progress()
        return payload

    async def add_client(self, websocket: WebSocket) -> None:
        self.clients.add(websocket)
        await websocket.send_json({"type": "state", **self.state_payload()})

    def remove_client(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        failed: list[WebSocket] = []
        async with self._broadcast_lock:
            for websocket in tuple(self.clients):
                try:
                    await websocket.send_json(message)
                except Exception:  # noqa: BLE001 - 清理失效浏览器连接
                    failed.append(websocket)
        for websocket in failed:
            self.remove_client(websocket)

    async def _sample_broadcast_loop(self) -> None:
        interval = 1.0 / SAMPLE_BROADCAST_HZ
        while True:
            await asyncio.sleep(interval)
            chunks: list[np.ndarray] = []
            while not self.sample_queue.empty():
                with suppress(asyncio.QueueEmpty):
                    chunks.append(self.sample_queue.get_nowait())
            if not chunks or self.stage not in {"device_check", "break_monitor"}:
                continue
            samples = np.hstack(chunks)
            await self.broadcast(
                {
                    "type": "samples",
                    "channels": CHANNEL_COUNT,
                    "samples": np.round(samples, 3).tolist(),
                }
            )

    async def _state_broadcast_loop(self) -> None:
        interval = 1.0 / STATE_BROADCAST_HZ
        while True:
            await self.broadcast({"type": "state", **self.state_payload()})
            await asyncio.sleep(interval)


def create_app(*, demo: bool = False, time_scale: float = 1.0) -> FastAPI:
    service = ApplicationService(demo=demo, time_scale=time_scale)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    web_app = FastAPI(
        title="Motor Imagery EEG Acquisition",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    web_app.state.service = service
    web_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @web_app.middleware("http")
    async def disable_ui_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @web_app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @web_app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @web_app.get("/api/status")
    async def status():
        return service.state_payload()

    @web_app.post("/api/session/configure")
    async def configure(form: SessionForm):
        try:
            return await service.configure_session(form)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @web_app.post("/api/session/start")
    async def start_experiment():
        try:
            await service.start_experiment()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @web_app.post("/api/session/abort")
    async def abort_experiment():
        if service.stage in ACTIVE_STAGES:
            await service.abort_experiment()
        elif service.haptics is not None and service.haptics.busy:
            await service.stop_haptic_test()
        else:
            raise HTTPException(status_code=409, detail="当前没有正在运行的实验")
        return {"ok": True}

    @web_app.post("/api/session/break/toggle")
    async def toggle_break_countdown():
        try:
            await service.toggle_break_countdown()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @web_app.post("/api/session/new")
    async def new_session():
        try:
            await service.new_session()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @web_app.post("/api/haptics/test")
    async def test_haptics():
        try:
            await service.test_haptics()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @web_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await service.add_client(websocket)
        try:
            while True:
                message = await websocket.receive_json()
                if message.get("action") == "abort" and service.stage in ACTIVE_STAGES:
                    await service.abort_experiment()
        except WebSocketDisconnect:
            pass
        finally:
            service.remove_client(websocket)

    return web_app


app = create_app(
    demo=os.getenv("MOTOR_IMAGERY_DEMO", "").lower() in {"1", "true", "yes"},
    time_scale=float(os.getenv("MOTOR_IMAGERY_TIME_SCALE", "1")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用模拟 EEG 和模拟马达，不连接蓝牙设备",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="仅演示模式使用；0.05 表示按 20 倍速度运行",
    )
    args = parser.parse_args()
    if not args.demo and args.time_scale != 1.0:
        parser.error("--time-scale 只能与 --demo 一起使用")

    logging.getLogger("bleak").setLevel(logging.WARNING)
    uvicorn.run(
        create_app(demo=args.demo, time_scale=args.time_scale),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
