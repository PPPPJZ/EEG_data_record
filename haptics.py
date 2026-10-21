"""双侧 BLE 马达连接、定时刺激与停止保护。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from bleak import BleakClient, BleakScanner

from config import (
    HAPTIC_CONNECT_TIMEOUT_SECONDS,
    HAPTIC_DEVICE_NAMES,
    HAPTIC_INTENSITY_DIFFERENCE,
    HAPTIC_LEVEL_UUID,
    HAPTIC_STEP_SECONDS,
    HAPTIC_TEST_SECONDS,
    HAPTIC_WRITE_TIMEOUT_SECONDS,
    RECONNECT_DELAY_SECONDS,
    SCAN_TIMEOUT_SECONDS,
)


def intensity_percentage(intensity: float) -> int:
    if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
        raise ValueError("震动强度必须是 0 到 1 之间的有限数值")
    return round(intensity * 100)


@dataclass(frozen=True)
class HapticCommand:
    motor_id: int
    trial_id: int
    context: str
    percentage: int
    requested_at: float
    acknowledged_at: float
    success: bool
    error: str = ""


class HapticController:
    """仅 tactile_MI 创建；两侧持久连接，任意时刻只有一个刺激任务。"""

    def __init__(
        self,
        on_fault: Callable[[str], None],
        on_command: Callable[[HapticCommand], None],
        *,
        demo: bool = False,
        time_scale: float = 1.0,
        scan_lock: asyncio.Lock | None = None,
        protocol: dict | None = None,
    ) -> None:
        self.on_fault = on_fault
        self.on_command = on_command
        self.demo = demo
        self.time_scale = time_scale
        self.protocol = protocol
        self._scan_lock = scan_lock if scan_lock is not None else asyncio.Lock()
        self._clients: dict[int, BleakClient] = {}
        self._write_locks = {side: asyncio.Lock() for side in HAPTIC_DEVICE_NAMES}
        self._connected = {side: False for side in HAPTIC_DEVICE_NAMES}
        self._statuses = {side: "等待连接" for side in HAPTIC_DEVICE_NAMES}
        self._motor_tasks: list[asyncio.Task] = []
        self._stimulus_task: asyncio.Task | None = None
        self._stimulus_cleanup = False
        self._activity = ""
        self.running = False
        self.last_fault = ""

    @property
    def ready(self) -> bool:
        return all(
            self._connected[side]
            and (self.demo or self._clients[side].is_connected)
            for side in HAPTIC_DEVICE_NAMES
        )

    @property
    def busy(self) -> bool:
        return bool(self._activity)

    def state_payload(self) -> dict:
        return {
            "ready": self.ready,
            "busy": self.busy,
            "activity": self._activity,
            "message": self.last_fault,
            "motors": [
                {
                    "motor_id": side,
                    "name": name,
                    "connected": self._connected[side],
                    "status": self._statuses[side],
                }
                for side, name in HAPTIC_DEVICE_NAMES.items()
            ],
        }

    async def start(self) -> None:
        self.running = True
        if self.demo:
            for side, name in HAPTIC_DEVICE_NAMES.items():
                self._connected[side] = True
                self._statuses[side] = f"已连接模拟 {name}"
        else:
            self._motor_tasks = [
                asyncio.create_task(self._run_motor(side), name=f"haptic-{name}")
                for side, name in HAPTIC_DEVICE_NAMES.items()
            ]

    async def stop(self) -> None:
        self.running = False
        await self.stop_stimulation()
        for task in self._motor_tasks:
            task.cancel()
        for task in self._motor_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._motor_tasks.clear()
        for side in HAPTIC_DEVICE_NAMES:
            self._connected[side] = False
            self._statuses[side] = "已断开"

    def _disconnected(self, side: int) -> None:
        was_connected = self._connected[side]
        self._connected[side] = False
        self._statuses[side] = "已断开，正在重连"
        if was_connected and self.running:
            self._fault(f"{HAPTIC_DEVICE_NAMES[side]} 连接断开")

    def _fault(self, message: str) -> None:
        self.last_fault = message
        self.on_fault(message)

    async def _run_motor(self, side: int) -> None:
        name = HAPTIC_DEVICE_NAMES[side]
        loop = asyncio.get_running_loop()
        while self.running:
            client = None
            try:
                self._statuses[side] = f"正在扫描 {name}"

                def matches(device, advertisement):
                    return name in {device.name, advertisement.local_name}

                # EEG 与两侧马达共用扫描锁，避免部分 BlueZ 适配器拒绝并发扫描。
                async with self._scan_lock:
                    device = await BleakScanner.find_device_by_filter(
                        matches, timeout=SCAN_TIMEOUT_SECONDS
                    )
                if device is None:
                    self._statuses[side] = f"未发现 {name}，正在重试"
                else:
                    disconnected = asyncio.Event()

                    def on_disconnected(disconnected_client, event=disconnected):
                        def notify():
                            if self._clients.get(side) is disconnected_client:
                                self._disconnected(side)
                            event.set()
                        loop.call_soon_threadsafe(notify)

                    self._statuses[side] = f"正在连接 {name}"
                    client = BleakClient(
                        device,
                        disconnected_callback=on_disconnected,
                        timeout=HAPTIC_CONNECT_TIMEOUT_SECONDS,
                    )
                    await client.connect()
                    self._clients[side] = client
                    if client.services.get_characteristic(HAPTIC_LEVEL_UUID) is None:
                        raise RuntimeError(f"{name} 缺少震动控制特征")
                    # 每次连接都先写零，只有完成初始化才允许开始采集。
                    await self._write(side, 0.0, context="connection")
                    if not client.is_connected:
                        raise RuntimeError(f"{name} 初始化期间断开")
                    self._connected[side] = True
                    self._statuses[side] = f"已连接 {name}"
                    await disconnected.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._statuses[side] = f"{name} 错误：{exc}"
                if self._connected[side]:
                    self._disconnected(side)
            finally:
                if client is not None:
                    self._disconnected(side)
                    if client.is_connected:
                        with suppress(Exception):
                            await self._write(side, 0.0, context="connection")
                    with suppress(Exception):
                        await asyncio.wait_for(
                            client.disconnect(), HAPTIC_CONNECT_TIMEOUT_SECONDS
                        )
                self._clients.pop(side, None)
            if self.running:
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    async def _write(
        self, side: int, intensity: float, *, context: str, trial_id: int = -1
    ) -> None:
        percentage = intensity_percentage(intensity)
        loop = asyncio.get_running_loop()
        async with self._write_locks[side]:
            requested = loop.time()
            error = ""
            success = False
            try:
                if not self.demo:
                    client = self._clients.get(side)
                    if client is None or not client.is_connected:
                        raise RuntimeError(f"{HAPTIC_DEVICE_NAMES[side]} 未连接")
                    await asyncio.wait_for(
                        client.write_gatt_char(
                            HAPTIC_LEVEL_UUID, bytes([percentage]), response=True
                        ),
                        HAPTIC_WRITE_TIMEOUT_SECONDS,
                    )
                success = True
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.on_command(HapticCommand(
                    motor_id=side,
                    trial_id=trial_id,
                    context=context,
                    percentage=percentage,
                    requested_at=requested,
                    acknowledged_at=loop.time(),
                    success=success,
                    error=error,
                ))

    async def _write_many(
        self, sides: tuple[int, ...], intensity: float, *, context: str, trial_id: int
    ) -> None:
        # 等待两侧都返回，单侧失败时也能在 finally 中停止另一侧。
        results = await asyncio.gather(
            *(self._write(side, intensity, context=context, trial_id=trial_id)
              for side in sides),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _zero_all(self, *, context: str = "stop", trial_id: int = -1) -> None:
        async def stop_side(side):
            if not self.demo:
                client = self._clients.get(side)
                if client is None or not client.is_connected:
                    return
            try:
                await self._write(side, 0.0, context=context, trial_id=trial_id)
            except Exception as exc:
                # 无法确认写零时，主动断连，沿用 XIAO 固件的断连停机机制。
                self._disconnected(side)
                client = self._clients.get(side)
                if client is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(
                            client.disconnect(), HAPTIC_CONNECT_TIMEOUT_SECONDS
                        )
                self._fault(f"{HAPTIC_DEVICE_NAMES[side]} 停止失败：{exc}")
        await asyncio.gather(*(stop_side(side) for side in HAPTIC_DEVICE_NAMES))

    def _check_start(self, intensity: float) -> None:
        intensity_percentage(intensity)
        if not self.ready:
            raise ValueError("两侧马达尚未全部连接")
        if self.busy:
            raise ValueError("震动正在执行，请等待结束")
        self.last_fault = ""

    def start_test(self, intensity: float) -> None:
        self._check_start(intensity)
        self._activity = "test"
        beginning = asyncio.get_running_loop().time()
        self._stimulus_task = asyncio.create_task(
            self._stimulate(
                tuple(HAPTIC_DEVICE_NAMES), intensity,
                beginning=beginning,
                deadline=beginning + (
                    self.protocol["test_seconds"] if self.protocol else HAPTIC_TEST_SECONDS
                ) * self.time_scale,
                alternating=False, trial_id=-1,
            ),
            name="haptic-test",
        )

    def start_trial(
        self, side: int, intensity: float, *, trial_id: int,
        beginning: float, deadline: float,
    ) -> None:
        self._check_start(intensity)
        if side not in HAPTIC_DEVICE_NAMES:
            raise ValueError("未知的马达侧别")
        self._activity = "trial"
        self._stimulus_task = asyncio.create_task(
            self._stimulate(
                (side,), intensity, beginning=beginning,
                deadline=deadline, alternating=True, trial_id=trial_id,
            ),
            name=f"haptic-trial-{trial_id}",
        )

    async def _stimulate(
        self, sides: tuple[int, ...], high: float, *, beginning: float,
        deadline: float, alternating: bool, trial_id: int,
    ) -> None:
        context = "trial" if alternating else "test"
        low = self.protocol["low_intensity"] if self.protocol else max(0.0, high - HAPTIC_INTENSITY_DIFFERENCE)
        step = (self.protocol["step_seconds"] if self.protocol else HAPTIC_STEP_SECONDS) * self.time_scale
        loop = asyncio.get_running_loop()
        try:
            previous_slot = -1
            while loop.time() < deadline:
                slot = int(max(0.0, loop.time() - beginning) / step)
                if slot != previous_slot:
                    level = low if alternating and slot % 2 else high
                    await self._write_many(
                        sides, level, context=context, trial_id=trial_id
                    )
                    previous_slot = slot
                wake_at = min(deadline, beginning + (slot + 1) * step)
                if not alternating:
                    wake_at = deadline
                # 使用绝对时刻，BLE 写入耗时不累积到下一档；迟到档不补发。
                await asyncio.sleep(max(0.0, wake_at - loop.time()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fault(f"震动执行失败：{exc}")
        finally:
            self._stimulus_cleanup = True
            try:
                await self._zero_all(context=context, trial_id=trial_id)
            finally:
                self._activity = ""
                self._stimulus_cleanup = False

    async def stop_stimulation(self) -> None:
        task = self._stimulus_task
        if task is not None:
            # 正常到期已在写零时等待完成，避免取消正在发出的停止命令。
            if not task.done() and not self._stimulus_cleanup:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self._stimulus_task = None
        # 任务可能尚未进入协程主体就被取消；此时仍必须清零并释放 busy。
        await self._zero_all()
        self._activity = ""
