"""3 秒原始 EEG 的分类；也可直接 from reasoning import Predictor。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import sys
import numpy as np
sys.dont_write_bytecode = True
from adfcnn_lib.cli import run_main
from adfcnn_lib.inference import Predictor

# 参数配置区：预处理从 teacher 文件恢复，避免与训练参数不一致。
DEFAULT_DEVICE = "cuda:0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="teacher checkpoint 文件路径")
    parser.add_argument("--window", required=True, help="含 (750, 8) 原始 uV 的 .npy 文件")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda:0 / cuda:N / cpu")
    parser.add_argument("--details", action="store_true", help="同时输出类别名称和两类概率 JSON")
    args = parser.parse_args()
    raw_uv = np.load(args.window, allow_pickle=False)
    predictor = Predictor(args.model, device=args.device)
    prediction = predictor.predict_details(raw_uv)
    if args.details:
        print(json.dumps(asdict(prediction), ensure_ascii=True), flush=True)
    else:
        print(prediction.class_id, flush=True)


if __name__ == "__main__":
    run_main(main)
