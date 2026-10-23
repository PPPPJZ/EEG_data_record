"""固定 epoch 训练：results 最后测试一次，teacher 使用全部数据且不评估。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .config import ModelConfig, TrainingConfig
from .data import SessionData, split_four_train_one_test, validate_balanced_folds
from .model import ADFCNN
from .runtime import resolve_device, set_seed


@dataclass(frozen=True)
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    final_epoch: int
    final_train_loss: float
    final_learning_rate: float
    training_seed: int
    train_indices: np.ndarray


@dataclass(frozen=True)
class FoldResult(TrainingResult):
    test_accuracy: float
    test_loss: float
    test_indices: np.ndarray


def make_loader(
    session: SessionData, indices: np.ndarray, config: TrainingConfig,
    device: torch.device, shuffle: bool, seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(session.inputs[indices], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(session.labels[indices], dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator,
        num_workers=config.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )


@torch.inference_mode()
def evaluate(model: ADFCNN, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += len(labels)
    if count == 0:
        raise ValueError("评估集不可为空")
    loss, accuracy = loss_sum / count, correct / count
    if not np.isfinite(loss):
        raise RuntimeError("评估 loss 非有限数值")
    return loss, accuracy


def _train_final_model(
    session: SessionData, train_indices: np.ndarray, seed: int,
    config: TrainingConfig, model_config: ModelConfig | None = None,
    progress: Callable[[int, float, float], None] | None = None,
) -> tuple[ADFCNN, TrainingResult]:
    architecture = ModelConfig() if model_config is None else model_config
    if len(train_indices) == 0:
        raise ValueError("训练集不可为空")
    set_seed(seed)
    device = resolve_device(config.device)
    train_loader = make_loader(session, train_indices, config, device, True, seed)
    model = ADFCNN(architecture).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=0.0
    )
    for epoch in range(1, config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        loss_sum, count = 0.0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch} 训练 loss 非有限数值")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            count += len(labels)
        scheduler.step()
        train_loss = loss_sum / count
        if progress is not None:
            progress(epoch, train_loss, learning_rate)
    result = TrainingResult(
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        final_epoch=config.epochs, final_train_loss=train_loss,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        training_seed=seed, train_indices=train_indices,
    )
    return model, result


def train_fold(
    session: SessionData, fold_ids: np.ndarray, test_fold: int, seed: int,
    config: TrainingConfig, model_config: ModelConfig | None = None,
    progress: Callable[[int, float, float], None] | None = None,
) -> FoldResult:
    """四折训练满全部 epoch，最终模型仅在 test 上评估一次。"""
    validate_balanced_folds(session.labels, fold_ids)
    train_indices, test_indices = split_four_train_one_test(fold_ids, test_fold)
    # 与参考实验一致，每折均固定同一个 seed，不按 fold 编号偏移。
    model, result = _train_final_model(session, train_indices, seed, config, model_config, progress)
    device = next(model.parameters()).device
    test_loader = make_loader(session, test_indices, config, device, False, seed)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    return FoldResult(
        state_dict=result.state_dict, final_epoch=result.final_epoch,
        final_train_loss=result.final_train_loss, final_learning_rate=result.final_learning_rate,
        training_seed=result.training_seed, train_indices=result.train_indices,
        test_accuracy=test_accuracy, test_loss=test_loss, test_indices=test_indices,
    )


def train_all(
    session: SessionData, seed: int, config: TrainingConfig,
    model_config: ModelConfig | None = None,
    progress: Callable[[int, float, float], None] | None = None,
) -> TrainingResult:
    """全部有效带标签 trial 参与训练，不划分、不欠采样、不评估。"""
    indices = np.arange(len(session.labels), dtype=np.int64)
    _, result = _train_final_model(session, indices, seed, config, model_config, progress)
    return result
