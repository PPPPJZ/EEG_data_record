"""自包含 ADFCNN；保留现有模型双分支、注意力和分类器的计算。"""
from __future__ import annotations
import math
import torch
from torch import nn
from .config import ModelConfig


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(inputs)


class ActSquare(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.square(inputs)


class ActLog(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(inputs, min=self.eps))


class ADFCNNBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        f, norm, drop = config.features, config.convolution_max_norm, config.dropout
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, f, kernel_size=(1, 125), padding="same", max_norm=norm),
            nn.BatchNorm2d(f),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, f, kernel_size=(1, 30), padding="same", max_norm=norm),
            nn.BatchNorm2d(f),
        )
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(f, f, (config.num_channels, 1), groups=f, bias=False, max_norm=norm),
            nn.BatchNorm2d(f), nn.ELU(), nn.Dropout(drop),
            Conv2dWithConstraint(f, f, kernel_size=(1, 1), max_norm=norm),
            nn.BatchNorm2d(f), nn.ELU(), nn.AvgPool2d((1, 32), stride=32), nn.Dropout(drop),
        )
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(f, f, (config.num_channels, 1), max_norm=norm),
            nn.BatchNorm2d(f), ActSquare(), nn.AvgPool2d((1, 75), stride=25),
            ActLog(config.log_floor), nn.Dropout(drop),
        )
        self.drop = nn.Dropout(drop)
        self.w_q = nn.Linear(f, f)
        self.w_k = nn.Linear(f, f)
        self.w_v = nn.Linear(f, f)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.spatial_1(self.spectral_1(inputs))
        second = self.spatial_2(self.spectral_2(inputs))
        combined = torch.cat((first, second), dim=3)
        batch, channels, height, width = combined.shape
        tokens = combined.reshape(batch, channels, height * width).permute(0, 2, 1)
        query = torch.nn.functional.normalize(self.w_q(tokens).permute(0, 2, 1), dim=-1)
        key = torch.nn.functional.normalize(self.w_k(tokens).permute(0, 2, 1), dim=-1)
        value = self.w_v(tokens).permute(0, 2, 1)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # 保留原 ADFCNN 的 reshape 次序，使架构计算与参考实现一致。
        attended = (attention.softmax(dim=-1) @ value).reshape(batch, height * width, channels)
        tokens = tokens + self.drop(attended)
        return self.drop(tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2))


class ADFCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = ADFCNNBackbone(config)
        # 750 点经两路池化得到 23 + 28 = 51 个位置。
        width = config.window_samples // 32 + (config.window_samples - 75) // 25 + 1
        self.classifier = nn.Sequential(
            nn.Conv2d(config.features, config.num_classes, (1, width)), nn.LogSoftmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (1, self.config.num_channels, self.config.window_samples)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(f"模型输入必须为 [batch, {expected}]，实际为 {tuple(inputs.shape)}")
        return self.classifier(self.backbone(inputs)).squeeze(3).squeeze(2)
