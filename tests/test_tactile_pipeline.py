"""触觉刺激与采集流程的回归验证；不会连接或驱动真实设备。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import ApplicationService, SessionForm
from config import (
    CLASS_LEFT, CLASS_RIGHT, EVENT_CUE_LEFT, EVENT_CUE_RIGHT, EVENT_TRIAL_END,
    HAPTIC_DEVICE_NAMES,
)
from haptics import HapticController
from load_session import load_session


async def wait_until(predicate, timeout=12.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("未到达预期状态")
        await asyncio.sleep(0.002)


class FakeMotor:
    def __init__(self, *, fail_positive=False, delay_positive=0.0):
        self.is_connected = True
        self.fail_positive = fail_positive
        self.delay_positive = delay_positive
        self.levels = []

    async def write_gatt_char(self, uuid, payload, *, response):
        assert response is True
        level = payload[0]
        if level and self.fail_positive:
            raise OSError("模拟 GATT 写入失败")
        if level:
            await asyncio.sleep(self.delay_positive)
        self.levels.append(level)

    async def disconnect(self):
        self.is_connected = False


def fake_controller(*, left=None, right=None):
    commands, faults = [], []
    controller = HapticController(faults.append, commands.append)
    controller.running = True
    controller._clients = {
        CLASS_LEFT: left if left is not None else FakeMotor(),
        CLASS_RIGHT: right if right is not None else FakeMotor(),
    }
    controller._connected = {CLASS_LEFT: True, CLASS_RIGHT: True}
    return controller, commands, faults


class MotorSchedulingTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_waits_for_zero_already_in_progress(self):
        stopping = asyncio.Event()

        class SlowStopMotor(FakeMotor):
            async def write_gatt_char(self, uuid, payload, *, response):
                if payload[0] == 0:
                    stopping.set()
                    await asyncio.sleep(0.03)
                await super().write_gatt_char(uuid, payload, response=response)

        controller, commands, faults = fake_controller(
            left=SlowStopMotor(), right=SlowStopMotor()
        )
        controller.time_scale = 0.01
        try:
            controller.start_test(0.8)
            await stopping.wait()
            await controller.stop_stimulation()
            self.assertFalse(controller.busy)
            self.assertFalse(faults)
            self.assertTrue(all(item.success for item in commands))
            for side in (CLASS_LEFT, CLASS_RIGHT):
                self.assertEqual(controller._clients[side].levels[-1], 0)
        finally:
            await controller.stop()

    async def test_connection_initialization_reconnect_and_old_callback(self):
        created = []
        scan_active = 0
        scan_peak = 0

        class LifecycleMotor(FakeMotor):
            def __init__(self, device, *, disconnected_callback, timeout):
                super().__init__()
                self.is_connected = False
                self.name = device.name
                self.callback = disconnected_callback
                self.services = SimpleNamespace(get_characteristic=lambda uuid: object())
                created.append(self)

            async def connect(self):
                self.is_connected = True

            def lose_connection(self):
                self.is_connected = False
                self.callback(self)

            async def disconnect(self):
                self.lose_connection()

        async def scan(matches, *, timeout):
            nonlocal scan_active, scan_peak
            scan_active += 1
            scan_peak = max(scan_peak, scan_active)
            try:
                await asyncio.sleep(0.003)
                for name in HAPTIC_DEVICE_NAMES.values():
                    device = SimpleNamespace(name=name)
                    if matches(device, SimpleNamespace(local_name=name)):
                        return device
                return None
            finally:
                scan_active -= 1

        commands, faults = [], []
        controller = HapticController(faults.append, commands.append, time_scale=0.01)
        with patch("haptics.BleakClient", LifecycleMotor), patch(
            "haptics.BleakScanner.find_device_by_filter", side_effect=scan
        ), patch("haptics.RECONNECT_DELAY_SECONDS", 0.001):
            try:
                await controller.start()
                await wait_until(lambda: controller.ready)
                self.assertEqual(scan_peak, 1)
                self.assertTrue(all(client.levels == [0] for client in created))
                old_left = controller._clients[CLASS_LEFT]
                old_left.lose_connection()
                await wait_until(lambda: (
                    controller.ready
                    and controller._clients[CLASS_LEFT] is not old_left
                ))
                self.assertTrue(faults)
                self.assertEqual(len(created), 3)
                old_left.callback(old_left)
                await asyncio.sleep(0)
                self.assertTrue(controller.ready)
                controller.start_test(0.8)
                await controller._stimulus_task
                for side in (CLASS_LEFT, CLASS_RIGHT):
                    self.assertEqual(controller._clients[side].levels[-2:], [80, 0])
            finally:
                await controller.stop()
        self.assertTrue(all(not client.is_connected for client in created))
        self.assertFalse(controller._clients)
        self.assertTrue(all(item.success for item in commands))

    async def test_real_time_quarter_second_steps_and_ipsilateral_only(self):
        controller, commands, faults = fake_controller()
        beginning = asyncio.get_running_loop().time()
        try:
            controller.start_trial(
                CLASS_RIGHT, 0.8, trial_id=1,
                beginning=beginning, deadline=beginning + 0.82,
            )
            await controller._stimulus_task
            positive = [item for item in commands if item.percentage > 0]
            self.assertEqual([item.motor_id for item in positive], [CLASS_RIGHT] * 4)
            self.assertEqual([item.percentage for item in positive], [80, 65, 80, 65])
            deltas = np.diff([item.requested_at for item in positive])
            self.assertTrue(np.all(np.abs(deltas - 0.25) < 0.06), deltas)
            for side in (CLASS_LEFT, CLASS_RIGHT):
                self.assertEqual(controller._clients[side].levels[-1], 0)
            self.assertFalse(faults)
        finally:
            await controller.stop()

    async def test_low_intensity_clamps_and_immediate_cancel_releases_busy(self):
        controller, commands, faults = fake_controller()
        beginning = asyncio.get_running_loop().time()
        try:
            controller.start_trial(
                CLASS_LEFT, 0.1, trial_id=1,
                beginning=beginning, deadline=beginning + 0.3,
            )
            await controller._stimulus_task
            left = [item.percentage for item in commands if item.motor_id == CLASS_LEFT]
            self.assertEqual(left[:2], [10, 0])
            controller.start_test(0.8)
            await controller.stop_stimulation()
            self.assertFalse(controller.busy)
            self.assertTrue(controller.ready)
            self.assertFalse(faults)
        finally:
            await controller.stop()

    async def test_partial_bilateral_test_failure_stops_successful_motor(self):
        controller, commands, faults = fake_controller(
            left=FakeMotor(fail_positive=True)
        )
        try:
            controller.start_test(0.8)
            await controller._stimulus_task
            self.assertTrue(faults)
            self.assertTrue(any(not item.success for item in commands))
            self.assertIn(80, controller._clients[CLASS_RIGHT].levels)
            self.assertEqual(controller._clients[CLASS_RIGHT].levels[-1], 0)
            self.assertFalse(controller.busy)
        finally:
            await controller.stop()

    async def test_write_timeout_stops_other_motor_and_records_failure(self):
        controller, commands, faults = fake_controller(
            left=FakeMotor(delay_positive=1.0)
        )
        try:
            with patch("haptics.HAPTIC_WRITE_TIMEOUT_SECONDS", 0.02):
                controller.start_test(0.8)
                await controller._stimulus_task
            self.assertTrue(faults)
            self.assertTrue(any("TimeoutError" in item.error for item in commands))
            self.assertEqual(controller._clients[CLASS_RIGHT].levels[-1], 0)
        finally:
            await controller.stop()


class TactileAcquisitionTest(unittest.IsolatedAsyncioTestCase):
    async def configure(self, service, output, *, mode="tactile_MI"):
        await service.start()
        await service.configure_session(SessionForm(
            subject_id="S-TACTILE",
            experiment_date="2026-10-16",
            description="触觉模式回归验证",
            left_trials=1, right_trials=1, break_every_trials=1,
            experiment_mode=mode, haptic_intensity=0.8,
            output_dir=str(output),
        ))
        await wait_until(lambda: service.state_payload()["can_start"])

    async def test_two_sides_cue_to_end_and_break_test_does_not_cross_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tactile"
            # 将 0.25 s 触觉步长保留为 62.5 ms，超过 Windows 默认定时粒度。
            service = ApplicationService(demo=True, time_scale=0.25)
            trace = []
            original = service._on_haptic_command

            def observe(command):
                trace.append((command, service.stage))
                original(command)

            service._on_haptic_command = observe
            try:
                await self.configure(service, output)
                await service.test_haptics()
                self.assertFalse(service.state_payload()["can_start"])
                with self.assertRaisesRegex(ValueError, "震动测试"):
                    await service.start_experiment()
                await wait_until(lambda: not service.haptics.busy)
                device_test = [item for item, _ in trace if item.percentage > 0]
                self.assertEqual({item.motor_id for item in device_test}, {0, 1})
                self.assertEqual({item.percentage for item in device_test}, {80})
                self.assertEqual(list(output.iterdir()), [])

                await service.start_experiment()
                await wait_until(lambda: service.stage == "break_monitor")
                samples_at_break = service.writer.sample_count
                # 倒计时只剩不到两秒时测试，必须等到测试停机才进入下一 trial。
                service._break_remaining_s = 0.4
                await service.test_haptics()
                with self.assertRaisesRegex(ValueError, "震动正在执行"):
                    await service.test_haptics()
                await asyncio.sleep(0.04)
                self.assertEqual(service.stage, "break_monitor")
                self.assertEqual(service.writer.sample_count, samples_at_break)
                self.assertAlmostEqual(service._break_remaining_s, 0.4)
                await wait_until(lambda: service.stage == "complete")
                self.assertFalse(service.haptics.busy)
            finally:
                await service.stop()

            session = load_session(output / "motor_imagery.h5")
            self.assertEqual(session["status"], "complete")
            self.assertEqual(session["metadata"]["experiment_mode"], "tactile_MI")
            self.assertTrue(session["metadata"]["runtime"]["demo"])
            self.assertEqual(
                session["protocol"]["haptic_stimulation"]["low_intensity"], 0.65
            )
            commands = session["haptic_commands"]
            self.assertEqual(session["quality"]["haptic_failed_command_count"], 0)
            for trial in session["trials"]:
                rows = commands[
                    (commands["trial_id"] == trial["trial_id"])
                    & (commands["percentage"] > 0)
                ]
                self.assertGreater(len(rows), 15)
                self.assertEqual(set(rows["motor_id"]), {trial["class_id"]})
                self.assertEqual(set(rows["percentage"]), {65, 80})
                events = session["events"]
                cue = events[
                    (events["trial_id"] == trial["trial_id"])
                    & np.isin(events["event_code"], [EVENT_CUE_LEFT, EVENT_CUE_RIGHT])
                ][0]
                end = events[
                    (events["trial_id"] == trial["trial_id"])
                    & (events["event_code"] == EVENT_TRIAL_END)
                ][0]
                self.assertTrue(np.all(rows["requested_wall_time_s"] >= cue["wall_time_s"]))
                self.assertTrue(np.all(rows["requested_wall_time_s"] < end["wall_time_s"]))

            break_tests = commands[commands["context"] == b"test"]
            self.assertEqual(set(break_tests["trial_id"]), {-1})
            self.assertEqual(set(break_tests["sample_index"]), {samples_at_break})
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"motor_imagery.h5", "README.md"}
            )
            forbidden = {"information", "preparing", "rest", "break_instruction", "complete"}
            self.assertFalse(any(
                item.percentage > 0 and stage in forbidden for item, stage in trace
            ))

    async def test_pure_mode_never_creates_or_scans_motor_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pure"
            service = ApplicationService(demo=True, time_scale=0.01)
            try:
                with patch("app.HapticController") as constructor:
                    await self.configure(service, output, mode="pure_MI")
                    constructor.assert_not_called()
                    self.assertIsNone(service.haptics)
                    with self.assertRaisesRegex(ValueError, "tactile_MI"):
                        await service.test_haptics()
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == "complete")
            finally:
                await service.stop()
            session = load_session(output / "motor_imagery.h5")
            self.assertEqual(len(session["haptic_commands"]), 0)
            self.assertFalse(session["protocol"]["haptic_stimulation"]["enabled"])

    async def test_abort_stops_both_and_retains_incomplete_session(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "abort"
            service = ApplicationService(demo=True, time_scale=0.05)
            trace = []
            original = service._on_haptic_command

            def observe(command):
                trace.append(command)
                original(command)

            service._on_haptic_command = observe
            try:
                await self.configure(service, output)
                await service.start_experiment()
                await wait_until(lambda: any(item.percentage > 0 for item in trace))
                await service.abort_experiment("测试中断")
                self.assertFalse(service.haptics.busy)
                for side in (0, 1):
                    self.assertEqual(
                        [item.percentage for item in trace if item.motor_id == side][-1], 0
                    )
            finally:
                await service.stop()
            session = load_session(output / "motor_imagery.h5")
            self.assertEqual(session["status"], "incomplete")

    async def test_motor_disconnect_during_trial_aborts_and_in_break_pauses(self):
        for stage in ("cue", "break_monitor"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                service = ApplicationService(demo=True, time_scale=0.05)
                try:
                    await self.configure(service, Path(directory) / "disconnect")
                    await service.start_experiment()
                    await wait_until(lambda: service.stage == stage)
                    service.haptics._disconnected(CLASS_RIGHT)
                    if stage == "cue":
                        await wait_until(lambda: service.last_outcome == "incomplete")
                        self.assertFalse(service.haptics.busy)
                    else:
                        self.assertTrue(service._break_paused)
                        self.assertIsNotNone(service.writer)
                        self.assertFalse(service.state_payload()["break_can_continue"])
                        with self.assertRaisesRegex(ValueError, "尚未就绪"):
                            await service.toggle_break_countdown()
                        service.haptics._connected[CLASS_RIGHT] = True
                        self.assertTrue(service.state_payload()["break_can_continue"])
                        await service.toggle_break_countdown()
                        self.assertGreater(service._break_remaining_s, 29)
                        await service.abort_experiment("测试完毕")
                finally:
                    await service.stop()


if __name__ == "__main__":
    unittest.main()
