from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils import Logger, get_img_contrast_loss, WBBLossConfig, supervised_contrastive_loss

@dataclass
class Stage1TrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    min_lr: float  # min lr
    warmup_epochs: int
    weight_decay: float
    continue_train: bool = False
    checkpoint_path: str = None
    w_img_loss : float = 0.5
    w_wbb_loss : float = 0.5

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

def _build_wb_one_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct one-hot labels for a batch of weak boxes
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : one-hot tensor, [R = num_wbb, num_classes]
    """
    all_labels = []
    for target in targets:
        all_labels.append(target["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # [R]
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels


class Stage1Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Stage1TrainerConfig,
        wbb_loss_config : WBBLossConfig,
    )-> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.wbb_loss_config = wbb_loss_config
        self.w_img_loss = config.w_img_loss
        self.w_wbb_loss = config.w_wbb_loss

        self.device = config.device
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                # Linear warm‑up
                LinearLR(self.optimizer, start_factor=self.config.lr * 0.01, end_factor=1.0,
                         total_iters=self.config.warmup_epochs),
                # cosine decay
                CosineAnnealingLR(self.optimizer, T_max=self.config.epochs - self.config.warmup_epochs,
                                  eta_min=self.config.lr * 1e-2)
            ],
            milestones=[self.config.warmup_epochs]
        )

        self.start_epoch = 1
        trainer_config = {
            'epochs': self.config.epochs,
            'lr': self.config.lr,
            'min_lr': self.config.min_lr,
            'warmup_epochs': self.config.warmup_epochs,
            'weight_decay': self.config.weight_decay,
        }
        self.logger = Logger(model_name="Stage1", trainer_config=trainer_config)

        self.dataset_MPs : Dict[str, torch.Tensor] = {}     # {class_id : prototype}

        if self.config.continue_train:
            # 加载检查点
            checkpoint = torch.load(self.config.checkpoint_path)

            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully.")

            # 设置当前训练轮数
            self.start_epoch = checkpoint['epoch'] + 1

            # 加载日志路径
            log_path = checkpoint['log_path']
            self.logger = Logger(model_name="Stage1", continue_existing=log_path)

            # 加载优化器状态
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            self.optimizer.load_state_dict(optimizer_state_dict)
            print("Optimizer state loaded successfully.")

            # 加载学习率调度器状态
            lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            print("Learning rate scheduler state loaded successfully.")

            # 加载类别原型
            self.dataset_MPs = checkpoint['dataset_MPs']

    def train(self)-> None:
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)

    def _train_one_epoch(self, epoch)-> None:
        epoch_total_loss = 0.0
        epoch_img_loss = 0.0
        epoch_wbb_loss = 0.0
        epoch_MFHA_loss = 0.0
        epoch_cam_loss = 0.0
        for iter, (images, target) in enumerate(self.train_loader, start=1):
            images = [img.to(self.config.device) for img in images]
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in target]

            X = torch.stack(images, dim=0)
            low_latent_features, high_feature_maps, high_aug_embedding_features, loss_cam, prototypes = self.model(X)
            img_multi_hot_labels = _build_image_multi_hot(targets, self.config.num_classes).to(self.config.device)
            wb_one_hot_labels = _build_wb_one_hot(targets, self.config.num_classes).to(self.config.device)

            loss_img = get_img_contrast_loss(low_latent_features, img_multi_hot_labels)
            loss_wbb = supervised_contrastive_loss(high_aug_embedding_features, wb_one_hot_labels, self.wbb_loss_config)
            loss_MFHA = self.w_img_loss * loss_img + self.w_wbb_loss * loss_wbb

            self.optimizer.zero_grad()
            loss_MFHA.backward()
            loss_cam.backward()

            self.optimizer.step()

            epoch_total_loss += (loss_MFHA.item() + loss_cam.item())
            epoch_img_loss += loss_img.item()
            epoch_wbb_loss += loss_wbb.item()
            epoch_MFHA_loss += loss_MFHA.item()
            epoch_cam_loss += loss_cam.item()

        self.lr_scheduler.step()
        num_iters = len(self.train_loader)
        average_total_loss = epoch_total_loss / num_iters
        average_img_loss = epoch_img_loss / num_iters
        average_wbb_loss = epoch_wbb_loss / num_iters
        average_MFHA_loss = epoch_MFHA_loss / num_iters
        average_cam_loss = epoch_cam_loss / num_iters

        self.logger.add_info(f"Epoch [{epoch}/{self.config.epochs}]"
                             f"Total Loss: {average_total_loss:.4f}, "
                             f"Image Contrast Loss: {average_img_loss:.4f}, "
                             f"Weak Box Contrast Loss: {average_wbb_loss:.4f}, "
                             f"MFHA Loss: {average_MFHA_loss:.4f}, "
                             f"CAM Loss: {average_cam_loss:.4f}\n"
                             )
        metrics = {
            'Epoch' : epoch,
            'Total Loss': average_total_loss,
            'Image Contrast Loss': average_img_loss,
            'Weak Box Contrast Loss': average_wbb_loss,
            'MFHA Loss': average_MFHA_loss,
            'CAM Loss': average_cam_loss,
        }
        self.logger.add_metrics(metrics)



