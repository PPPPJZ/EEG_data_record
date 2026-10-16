"""跨平台单马达手动调试：python BLE/haptic_control.py --side left 或 right。

Commands:
  1         full configured strength
  0.5       half drive command
  0         stop
  ramp 1 0.5 2    ramp from 1.0 to 0.5 in two seconds
  q         stop and exit
"""

import argparse
import asyncio
import math
from pathlib import Path
import sys
import time
from bleak import BleakClient, BleakScanner

# 直接运行子目录脚本时，从项目根目录加载统一的硬件配置。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_LEFT, CLASS_RIGHT, HAPTIC_DEVICE_NAMES, HAPTIC_LEVEL_UUID

LEVEL_UUID = HAPTIC_LEVEL_UUID


def validate(value: float) -> None:
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError("Intensity must be a finite number between 0.0 and 1.0")


async def set_intensity(client: BleakClient, intensity: float) -> None:
    """Write a new state; this persists until another write or disconnect."""
    validate(intensity)
    percentage = round(intensity * 100)
    await client.write_gatt_char(LEVEL_UUID, bytes([percentage]), response=True)


async def ramp_intensity(client: BleakClient, start: float,
                         end: float, duration_s: float,
                         interval_s: float = 0.05) -> None:
    """Host-controlled ramp; the XIAO has no timer or ramp generator."""
    validate(start)
    validate(end)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Ramp duration must be > 0")

    await set_intensity(client, start)
    beginning = time.monotonic()
    while True:
        elapsed = time.monotonic() - beginning
        if elapsed >= duration_s:
            break
        await asyncio.sleep(min(interval_s, duration_s - elapsed))
        fraction = min((time.monotonic() - beginning) / duration_s, 1.0)
        await set_intensity(client, start + (end - start) * fraction)
    await set_intensity(client, end)


async def cancel_ramp(task) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def main(device_name: str) -> None:
    print(f"Scanning for {device_name}...")
    device = await BleakScanner.find_device_by_name(device_name, timeout=15.0)
    if device is None:
        print("Device not found. Check board power, Bluetooth, and macOS permission.")
        return

    print(f"Found: {device.name} ({device.address})")
    async with BleakClient(device) as client:
        print("Connected. Enter a number 0..1, 'ramp 1 0.5 2', or q to quit.")
        ramp_task = None
        try:
            while client.is_connected:
                line = (await asyncio.to_thread(input, "haptic> ")).strip()
                if not line:
                    continue
                parts = line.split()
                if parts[0].lower() in {"q", "quit", "exit"}:
                    break

                try:
                    if parts[0].lower() == "ramp" and len(parts) == 4:
                        start, end, seconds = map(float, parts[1:])
                        validate(start)
                        validate(end)
                        if not math.isfinite(seconds) or seconds <= 0:
                            raise ValueError("Duration must be positive and finite")
                        await cancel_ramp(ramp_task)
                        ramp_task = asyncio.create_task(
                            ramp_intensity(client, start, end, seconds)
                        )
                        print(f"Ramping {start:.2f} -> {end:.2f} over {seconds:g}s")
                    elif len(parts) == 1:
                        intensity = 0.0 if parts[0].lower() == "stop" else float(parts[0])
                        validate(intensity)
                        await cancel_ramp(ramp_task)
                        ramp_task = None
                        await set_intensity(client, intensity)
                        print(f"Intensity set to {intensity:.2f}")
                    else:
                        print("Use: 0..1 | stop | ramp <start> <end> <seconds> | q")
                except ValueError as error:
                    print(f"Ignored invalid command: {error}")
        finally:
            await cancel_ramp(ramp_task)
            if client.is_connected:
                await set_intensity(client, 0.0)
                print("Motor stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    args = parser.parse_args()
    side = CLASS_LEFT if args.side == "left" else CLASS_RIGHT
    try:
        asyncio.run(main(HAPTIC_DEVICE_NAMES[side]))
    except KeyboardInterrupt:
        print("\nInterrupted; BLE disconnect stops the motor on the XIAO.")
