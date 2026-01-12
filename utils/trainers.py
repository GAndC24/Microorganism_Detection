import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from datasets import class_map_encoding
from utils import build_weak_boxes, get_img_contrast_loss

@dataclass
class Stage1TrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    min_lr: float  # min lr
    warmup_epochs: int
    weight_decay: float
    log_interval: int = 10  # 日志打印间隔（步）。
    max_grad_norm: Optional[float] = 5.0  # 梯度裁剪阈值（None 表示不裁剪）。

def _to_tensor(image: torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):  # 如果已是张量，直接返回。
        return image
    return TF.to_tensor(image)  # 否则将 PIL/ndarray 转为张量。

def _build_image_multi_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct multi-hot labels for a batch of images
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : multi-hot tensor, [B, num_classes]
    """
    labels = torch.zeros((len(targets), num_classes), dtype=torch.float32)
    for idx, target in enumerate(targets):
        if target["labels"].numel() == 0:
            continue
        labels[idx, target["labels"].unique()] = 1.0
    return labels

