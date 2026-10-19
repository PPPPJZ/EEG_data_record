"""三个脚本的共用命令行参数与错误输出。"""
from __future__ import annotations
import argparse
import sys
from typing import Callable
from .config import TrainingConfig
from .data import SessionData


def add_training_arguments(parser: argparse.ArgumentParser, defaults: TrainingConfig) -> None:
    parser.add_argument("--data", required=True, help="motor_imagery.h5 文件路径")
    parser.add_argument("--device", default=defaults.device, help="cuda:0 / cuda:N / cpu")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--verbose", action="store_true", help="将数据和进度信息输出到 stderr")


def training_from_args(args: argparse.Namespace, defaults: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=defaults.learning_rate, weight_decay=defaults.weight_decay,
        device=args.device, num_workers=defaults.num_workers,
    )


def describe_session(session: SessionData) -> None:
    print(f"Data: {session.source_path}", file=sys.stderr, flush=True)
    print(f"Trials: {session.summary}", file=sys.stderr, flush=True)


def epoch_reporter(label: str, epochs: int) -> Callable[[int, float, float], None]:
    def report(epoch: int, accuracy: float, loss: float) -> None:
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"{label} epoch={epoch}/{epochs} val_accuracy={accuracy:.4f} val_loss={loss:.4f}",
                file=sys.stderr, flush=True,
            )
    return report


def run_main(main: Callable[[], None]) -> None:
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
