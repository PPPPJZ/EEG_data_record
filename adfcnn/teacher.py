"""单 seed、固定 3 train / 1 val / 1 test，保存个人 Feedback 模型。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from adfcnn_lib.checkpoint import save_teacher
from adfcnn_lib.cli import (
    add_training_arguments, describe_session, epoch_reporter, run_main, training_from_args,
)
from adfcnn_lib.config import (
    TEACHER_FILENAME, TEACHER_SEED, TEACHER_TEST_FOLD,
    ModelConfig, PreprocessingConfig, TrainingConfig,
)
from adfcnn_lib.data import load_session, make_folds
from adfcnn_lib.training import train_fold

# 参数配置区：teacher 独立于 results 的配置。
TRAINING = TrainingConfig(
    epochs=300, batch_size=16, learning_rate=0.001, weight_decay=0.075,
    device="cuda:0", num_workers=0,
)
PREPROCESSING = PreprocessingConfig(lowpass_hz=40.0, standardization_std_floor=1e-4)
MODEL = ModelConfig(features=8, dropout=0.25, convolution_max_norm=2.0, log_floor=1e-6)
SEED = TEACHER_SEED
TEST_FOLD = TEACHER_TEST_FOLD
OUTPUT_FILENAME = TEACHER_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, TRAINING)
    parser.add_argument("--output", help="可选；默认保存到 H5 所在目录/adfcnn_teacher.pt")
    args = parser.parse_args()
    config = training_from_args(args, TRAINING)
    source_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else source_path.parent / OUTPUT_FILENAME
    if output == source_path:
        raise ValueError("模型输出路径不可与输入 H5 相同")
    session = load_session(source_path, PREPROCESSING)
    if args.verbose:
        describe_session(session)
    folds = make_folds(session.labels, SEED)
    progress = epoch_reporter("teacher", config.epochs) if args.verbose else None
    result = train_fold(session, folds, TEST_FOLD, SEED, config, MODEL, progress)
    saved = save_teacher(output, session, result, folds, TEST_FOLD, SEED, config, MODEL)
    print(f"Accuracy: {100 * result.test_accuracy:.2f}%", flush=True)
    print(f"Model: {saved}", flush=True)


if __name__ == "__main__":
    run_main(main)
