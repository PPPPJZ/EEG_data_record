"""单线程后台 teacher 推理：采集和网页事件循环不运行模型或 MNE。"""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .control import ControlConfig, CursorController, RawRingBuffer, WINDOW_SAMPLES


class MockPredictor:
    """只供显式 --demo --mock-teacher 使用，周期性左右概率。"""
    def __init__(self):
        self.index = 0

    def predict_details(self, window):
        from types import SimpleNamespace
        self.index += 1
        p = 0.5 + 0.45 * math.sin(self.index / 15)
        return SimpleNamespace(probabilities=(1 - p, p), class_id=int(p >= 0.5))


class FeedbackRuntime:
    def __init__(
        self, model_path: str, *, device: str = "cuda:0", mock: bool = False,
        time_scale: float = 1.0, on_prediction: Callable[[dict], None] | None = None,
        on_fault: Callable[[str], None] | None = None,
    ):
        self.config = ControlConfig.load()
        self.cursor = CursorController(self.config)
        self.ring = RawRingBuffer()
        self.model_path = model_path
        self.device = device
        self.mock = mock
        self.time_scale = time_scale
        self.on_prediction = on_prediction
        self.on_fault = on_fault
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher")
        self.predictor = None
        self.ready = False
        self.status = "模型加载与预热中"
        self.error = ""
        self.busy = False
        self.test_result = None
        self.enabled = False
        self.context = ""
        self.generation = 0
        self.trial_id = -1
        self.sample_end = 0
        self.last_sample_time = None
        self.control_started_at = None
        self.last_inferred_total = -1
        self.prediction_count = 0
        self.last_prediction = None
        self.load_info = {}
        self._tasks = []
        self._closing = False

    async def start(self):
        # Windows 的 WinRT 蓝牙与 PyTorch DLL 初始化顺序敏感。
        # 先在线程中导入原生运行时，随后允许 BLE 扫描和模型预热并行。
        if not self.mock:
            try:
                await self._background(lambda: __import__("torch"))
            except Exception as exc:
                self.error = str(exc)
                self.status = f"模型初始化失败：{exc}"
                return
        self._tasks = [
            asyncio.create_task(self._initialize(), name="teacher-initialize"),
            asyncio.create_task(self._run(), name="teacher-feedback"),
        ]

    async def _background(self, function):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function)

    def _load(self):
        started = time.perf_counter()
        if self.mock:
            self.predictor = MockPredictor()
            digest = ""
        else:
            from adfcnn.adfcnn_lib.inference import Predictor
            path = Path(self.model_path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.predictor = Predictor(path, device=self.device)
            if self.predictor.preprocessing.lowpass_hz != 40.0:
                raise ValueError("本阶段要求 teacher 使用 40 Hz 窗口低通")
        # 所有延迟初始化在设备检查阶段完成，不计入模型测试。
        for _ in range(10):
            self.predictor.predict_details(np.zeros((750, 8), dtype=np.float32))
        return {
            "sha256": digest, "device": self.device, "mock": self.mock,
            "model_path": self.model_path,
            "load_and_warmup_ms": (time.perf_counter() - started) * 1000,
            "raw_window_shape": [750, 8], "sample_rate_hz": 250,
            "preprocessing": (
                asdict(self.predictor.preprocessing) if not self.mock else {"mock": True}
            ),
            "model_config": asdict(self.predictor.model_config) if not self.mock else {},
        }

    async def _initialize(self):
        try:
            self.load_info = await self._background(self._load)
            self.ready = True
            self.status = "模拟 teacher 已就绪" if self.mock else "Teacher 已加载并预热"
        except Exception as exc:
            self.error = str(exc)
            self.status = f"模型初始化失败：{exc}"

    async def close(self):
        self.disable()
        self._closing = True
        # 等待已提交的任务正常退出，避免 GPU 仍在工作时销毁对象。
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)
        self.predictor = None

    def begin(self, *, trial_id: int = -1, context: str = "preview"):
        if not self.ready or self.busy:
            raise ValueError("Teacher 未就绪或正在模型测试")
        self.generation += 1
        self.ring.reset()
        self.control_started_at = time.perf_counter()
        self.cursor.reset(self.control_started_at / self.time_scale)
        self.last_prediction = None
        self.last_inferred_total = -1
        self.last_sample_time = None
        self.prediction_count = 0
        self.trial_id = trial_id
        self.context = context
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.generation += 1
        self.cursor.freeze(time.perf_counter() / self.time_scale)

    def append(self, raw_uv: np.ndarray, sample_end: int):
        if not self.enabled:
            return
        now = time.perf_counter()
        # 信号中断后重新积累完整窗口，不继续使用断点两侧的数据。
        if self.last_sample_time is not None and now - self.last_sample_time > 0.25:
            self.generation += 1
            self.ring.reset()
            self.last_inferred_total = -1
            self.cursor.freeze(now / self.time_scale)
        self.ring.append(raw_uv)
        self.last_sample_time = now
        self.sample_end = sample_end

    def _infer(self, window):
        started = time.perf_counter_ns()
        prediction = self.predictor.predict_details(window)
        return prediction, (time.perf_counter_ns() - started) / 1e6

    async def test(self):
        if not self.ready or self.busy or self.enabled:
            raise ValueError("模型未就绪、测试中或光标预览正在运行")
        self.busy = True
        self.test_result = None

        def run_test():
            rng = np.random.default_rng()
            windows = rng.normal(0, 20, (10, 750, 8)).astype(np.float32)
            durations, classes = [], []
            for window in windows:
                result, ms = self._infer(window)
                durations.append(ms)
                classes.append(result.class_id)
            return {
                "count": 10, "mean_ms": float(np.mean(durations)),
                "max_ms": float(np.max(durations)), "durations_ms": durations,
                "class_ids": classes,
                "within_interval": max(durations) < self.config.inference_interval_ms,
                "scope": "raw window FIR + standardization + model + probabilities",
                "mock": self.mock,
            }
        try:
            self.test_result = await self._background(run_test)
            return self.test_result
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            self.status = f"模型测试失败：{exc}"
            raise ValueError(self.status) from exc
        finally:
            self.busy = False

    async def _run(self):
        interval = self.config.inference_interval_ms / 1000
        deadline = time.perf_counter()
        while not self._closing:
            # Python 3.11 的 time.sleep 在 Windows 使用高精度等待计时器，
            # 放在专用线程等待，避免 asyncio 的低精度默认时钟量化 40 ms。
            await self._background(lambda: time.sleep(max(0, deadline - time.perf_counter())))
            if self._closing:
                break
            deadline += interval
            if deadline < time.perf_counter():
                deadline = time.perf_counter() + interval  # 过载不排队补跑旧窗口
            if not self.ready or not self.enabled or self.busy:
                continue
            now = time.perf_counter()
            if self.last_sample_time is None or now - self.last_sample_time > 0.25:
                self.cursor.freeze(now / self.time_scale)
                continue
            if self.last_inferred_total == self.ring.total:
                continue
            window = self.ring.snapshot()
            if window is None:
                continue
            generation = self.generation
            sample_end = self.sample_end
            self.last_inferred_total = self.ring.total
            try:
                result, ms = await self._background(lambda: self._infer(window))
                if not self.enabled or generation != self.generation:
                    continue
                now = time.perf_counter()
                if now - self.last_sample_time > 0.25:
                    self.cursor.freeze(now / self.time_scale)
                    continue
                p = float(result.probabilities[1])
                self.cursor.update(p, now / self.time_scale)
                self.prediction_count += 1
                row = {
                    "trial_id": self.trial_id,
                    "window_start_sample": sample_end - WINDOW_SAMPLES,
                    "window_end_sample": sample_end,
                    "prediction_class_id": int(result.class_id),
                    "p_right": p, "signal": self.cursor.signal,
                    "smooth": self.cursor.smooth, "control": self.cursor.control,
                    "velocity": self.cursor.velocity, "cursor_x": self.cursor.x,
                    "inference_ms": ms, "completed_at": now,
                    "control_elapsed_s": (now - self.control_started_at) / self.time_scale,
                }
                self.last_prediction = row
                if self.on_prediction is not None:
                    self.on_prediction(row)
            except Exception as exc:
                self.error = str(exc)
                self.status = f"实时推理失败：{exc}"
                self.ready = False
                self.disable()
                if self.on_fault:
                    self.on_fault(self.status)

    def state_payload(self):
        now = time.perf_counter()
        if self.enabled:
            if self.last_sample_time is None or now - self.last_sample_time > 0.25:
                self.cursor.freeze(now / self.time_scale)
            else:
                self.cursor.advance(now / self.time_scale)
        return {
            "ready": self.ready, "status": self.status, "error": self.error,
            "busy": self.busy, "test_result": self.test_result,
            "enabled": self.enabled, "context": self.context, "mock": self.mock,
            "cursor_x": self.cursor.x, "velocity": self.cursor.velocity,
            "buffer_samples": self.ring.count, "prediction_count": self.prediction_count,
            "last_prediction": self.last_prediction, "config": asdict(self.config),
            "time_scale": self.time_scale, "load_info": self.load_info,
        }
