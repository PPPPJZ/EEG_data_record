"""seed 42、平衡五折、4 train / 1 test；stdout 仅输出最终平均 Test Accuracy。"""
from __future__ import annotations
import argparse
import sys
# results 不生成 Python 字节码缓存或实验文件。
sys.dont_write_bytecode = True
from adfcnn_lib.cli import (
    add_training_arguments, describe_session, epoch_reporter, run_main,
    training_from_args,
)
from adfcnn_lib.config import N_FOLDS, RESULTS_SEED, ModelConfig, PreprocessingConfig, TrainingConfig
from adfcnn_lib.data import load_session, make_folds
from adfcnn_lib.training import train_fold

# 参数配置区：每个入口显式指定实验参数，可直接在此修改。
TRAINING = TrainingConfig(
    epochs=2000, batch_size=16, learning_rate=0.001, weight_decay=0.075,
    device="cuda:0", num_workers=0,
)
PREPROCESSING = PreprocessingConfig(lowpass_hz=40.0, standardization_std_floor=1e-4)
MODEL = ModelConfig(features=8, dropout=0.25, convolution_max_norm=2.0, log_floor=1e-6)
SEED = RESULTS_SEED  # 划分和每折训练均固定为 42。


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, TRAINING)
    args = parser.parse_args()
    config = training_from_args(args, TRAINING)
    session = load_session(args.data, PREPROCESSING)
    if args.verbose:
        describe_session(session)
    accuracies = []
    folds = make_folds(session.labels, SEED)
    for test_fold in range(1, N_FOLDS + 1):
        label = f"seed={SEED} test_fold={test_fold}"
        progress = epoch_reporter(label, config.epochs) if args.verbose else None
        result = train_fold(session, folds, test_fold, SEED, config, MODEL, progress)
        accuracies.append(result.test_accuracy)
        if args.verbose:
            print(f"{label} Test Accuracy: {100 * result.test_accuracy:.2f}%", file=sys.stderr, flush=True)
        del result
    print(f"Accuracy: {100 * sum(accuracies) / len(accuracies):.2f}%", flush=True)


if __name__ == "__main__":
    run_main(main)
