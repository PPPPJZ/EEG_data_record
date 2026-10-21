"""训练仅在内存中保留最佳验证模型，测试集在选择完成后评估一次。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .config import ModelConfig, TrainingConfig
from .data import (
    SessionData, balanced_train_validation_split, split_fold_indices,
)
from .model import ADFCNN
from .runtime import resolve_device, set_seed


@dataclass(frozen=True)
class FoldResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    validation_accuracy: float
    validation_loss: float
    test_accuracy: float
    training_seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    class_balance: dict


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


def train_fold(
    session: SessionData, fold_ids: np.ndarray, test_fold: int, seed: int,
    config: TrainingConfig, model_config: ModelConfig | None = None,
    progress: Callable[[int, float, float], None] | None = None,
) -> FoldResult:
    architecture = ModelConfig() if model_config is None else model_config
    if len(fold_ids) != len(session.labels):
        raise ValueError("fold_ids 与 trial 数量不一致")
    # 每折使用独立且确定的模型初始化；teacher 与 results 的第一折一致。
    training_seed = seed + test_fold - 1
    if config.balance_train_validation:
        (
            train_indices,
            validation_indices,
            test_indices,
            class_balance,
        ) = balanced_train_validation_split(
            fold_ids, session.labels, test_fold, training_seed
        )
    else:
        train_indices, validation_indices, test_indices = split_fold_indices(
            fold_ids, test_fold
        )

        def counts(indices: np.ndarray) -> list[int]:
            return np.bincount(
                session.labels[indices], minlength=2
            ).astype(int).tolist()

        class_balance = {
            "method": "none",
            "train_original_class_counts": counts(train_indices),
            "train_balanced_class_counts": counts(train_indices),
            "validation_original_class_counts": counts(validation_indices),
            "validation_balanced_class_counts": counts(validation_indices),
            "test_class_counts": counts(test_indices),
            "dropped_train_indices": [],
            "dropped_validation_indices": [],
        }
    set_seed(training_seed)
    device = resolve_device(config.device)
    train_loader = make_loader(session, train_indices, config, device, True, training_seed)
    validation_loader = make_loader(session, validation_indices, config, device, False, training_seed)
    model = ADFCNN(architecture).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=0.0
    )
    best_state, best_epoch = None, 0
    best_accuracy, best_loss = -1.0, float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch} 训练 loss 非有限数值")
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        if (validation_accuracy, -validation_loss) > (best_accuracy, -best_loss):
            best_epoch, best_accuracy, best_loss = epoch, validation_accuracy, validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress is not None:
            progress(epoch, validation_accuracy, validation_loss)
    if best_state is None:
        raise RuntimeError("训练未选出有效模型")
    model.load_state_dict(best_state, strict=True)
    test_loader = make_loader(session, test_indices, config, device, False, training_seed)
    _, test_accuracy = evaluate(model, test_loader, device)
    return FoldResult(
        state_dict=best_state, best_epoch=best_epoch, validation_accuracy=best_accuracy,
        validation_loss=best_loss, test_accuracy=test_accuracy, training_seed=training_seed,
        train_indices=train_indices, validation_indices=validation_indices,
        test_indices=test_indices, class_balance=class_balance,
    )
