from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils import Logger, get_img_contrast_loss, SupConLossConfig, LossContrastMode, supervised_contrastive_loss, get_patch_loss
from tqdm.auto import tqdm
from torchmetrics import MetricCollection, Precision, Recall, F1Score, AUROC, Accuracy, AveragePrecision

@dataclass
class Stage1TrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    warm_up_lr_factor: float  # min_lr = warm_up_lr_factor * lr
    warmup_epochs: int
    weight_decay: float
    checkpoints_save_path: str
    model_save_path : str
    dataset_mps_save_path : str
    logger: Logger
    continue_train: bool = False
    checkpoint_path: str = None
    w_img_loss : float = 0.5    # weight for image contrast loss
    w_wbb_loss : float = 0.5    # weight for weak box contrast loss
    w_cam_loss : float = 0.5    # weight for cam loss
    w_patch_loss : float = 0.5  # weight for patch loss
    mp_ema_alpha: float = 0.9   # ema alpha for updating prototypes, [0.9, 0.99]

@dataclass
class LinearProbTrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    warm_up_lr_factor: float  # min_lr = warm_up_lr_factor * lr
    warmup_epochs: int
    weight_decay: float
    checkpoints_save_path: str
    model_save_path: str
    logger: Logger
    continue_train: bool = False
    checkpoint_path: str = None

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
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32).to(all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels

def _build_wboxes(targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Construct wboxes tensor from targets.
    :param targets: List of dictionaries containing bounding box information.
    :return: wboxes tensor of shape [R, 5], where each box is [batch_idx, x1, y1, x2, y2].
    """
    wboxes = []
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]  # Convert to [x1, y1, x2, y2]
        batch_indices = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        wboxes.append(torch.cat([batch_indices, boxes], dim=1))  # Combine batch_idx with boxes

    return torch.cat(wboxes, dim=0)  # Concatenate all boxes across the batch

class Stage1Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Stage1TrainerConfig,
        supcon_loss_config : SupConLossConfig,
    )-> None:
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.supcon_loss_config = supcon_loss_config
        self.w_img_loss = config.w_img_loss
        self.w_wbb_loss = config.w_wbb_loss
        self.w_cam_loss = config.w_cam_loss
        self.w_patch_loss = config.w_patch_loss
        self.mp_ema_alpha = config.mp_ema_alpha

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
                LinearLR(self.optimizer, start_factor=self.config.warm_up_lr_factor, end_factor=1.0),
                # cosine decay
                CosineAnnealingLR(self.optimizer, T_max=self.config.epochs - self.config.warmup_epochs,
                                  eta_min=self.config.lr * self.config.warm_up_lr_factor)
            ],
            milestones=[self.config.warmup_epochs]
        )

        self.start_epoch = 1

        self.logger = config.logger
        self.log_path = self.logger.get_log_dir()

        self.dataset_MPs : Dict[int, torch.Tensor] = {}     # {class_id : prototype}

        if self.config.continue_train:
            # 加载检查点
            checkpoint = torch.load(self.config.checkpoint_path)

            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully.")

            # 设置当前训练轮数
            self.start_epoch = checkpoint['epoch'] + 1

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

    @torch.no_grad()
    def _update_dataset_mps_ema(self, batch_prototypes: Dict[int, torch.Tensor]) -> None:
        if (batch_prototypes is None) or (len(batch_prototypes) == 0):
            return
        alpha = float(self.mp_ema_alpha)
        for cls_id, p_new in batch_prototypes.items():
            if p_new is None:
                continue

            key = cls_id
            p_new = p_new.detach().to(self.device)
            p_new = torch.nn.functional.normalize(p_new, dim=-1, eps=1e-6)

            if key not in self.dataset_MPs:  # 初次写入：直接存
                self.dataset_MPs[key] = p_new
            else:
                p_old = self.dataset_MPs[key].to(self.device)
                p_old = p_old.detach()

                p_updated = alpha * p_old + (1.0 - alpha) * p_new
                p_updated = torch.nn.functional.normalize(p_updated, dim=-1, eps=1e-6)

                self.dataset_MPs[key] = p_updated

    def _train_one_epoch(self, epoch) -> None:
        num_iters = len(self.train_loader)
        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_iters,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
            dynamic_ncols=True,  # 自适应终端宽度
        )

        epoch_total_loss = 0.0
        epoch_img_loss = 0.0
        epoch_wbb_loss = 0.0
        epoch_MFHA_loss = 0.0
        epoch_cam_loss = 0.0
        epoch_patch_loss = 0.0
        epoch_patch_cls_loss = 0.0
        epoch_patch_supcon_loss = 0.0
        for iter, (images, target) in pbar:
            images = [img.to(self.config.device) for img in images]
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in target]

            wboxes = _build_wboxes(targets).to(self.config.device)
            wb_one_hot_labels = _build_wb_one_hot(targets, self.config.num_classes).to(self.config.device)
            X = torch.stack(images, dim=0)
            low_latent_features, high_feature_maps, high_aug_embedding_features, loss_cam, prototypes, patch_logits, contrast_patch_features = self.model(X, wboxes, wb_one_hot_labels)
            img_multi_hot_labels = _build_image_multi_hot(targets, self.config.num_classes).to(self.config.device)

            self._update_dataset_mps_ema(prototypes)

            loss_img = get_img_contrast_loss(low_latent_features, img_multi_hot_labels)
            loss_wbb = supervised_contrastive_loss(high_aug_embedding_features, wb_one_hot_labels, self.supcon_loss_config)
            loss_MFHA = self.w_img_loss * loss_img + self.w_wbb_loss * loss_wbb
            # loss_patch_cls = get_patch_loss(patch_logits, wb_one_hot_labels)
            supcon_loss_config = self.supcon_loss_config
            supcon_loss_config.contrast_mode = LossContrastMode.ONE_VIEW
            loss_patch_supcon = supervised_contrastive_loss(contrast_patch_features, wb_one_hot_labels, supcon_loss_config)
            # loss_patch = loss_patch_cls + loss_patch_supcon
            loss_patch = loss_patch_supcon
            loss = loss_MFHA + self.w_cam_loss * loss_cam + self.w_patch_loss * loss_patch

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_total_loss += loss.item()
            epoch_img_loss += loss_img.item()
            epoch_wbb_loss += loss_wbb.item()
            epoch_MFHA_loss += loss_MFHA.item()
            epoch_cam_loss += loss_cam.item()
            epoch_patch_loss += loss_patch.item()
            # epoch_patch_cls_loss += loss_patch_cls.item()
            # epoch_patch_supcon_loss += loss_patch_supcon.item()

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "MFHA": f"{loss_MFHA.item():.4f} ",
                "CAM": f"{loss_cam.item():.4f} ",
                "Patch": f"{loss_patch.item():.4f} ",
                # "patch_cls" : f"{loss_patch_cls.item():.4f} ",
                # "patch_supcon" : f"{loss_patch_supcon.item():.4f} ",
                "img": f"{loss_img.item():.4f} ",
                "wbb": f"{loss_wbb.item():.4f} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_total_loss = epoch_total_loss / num_iters
        average_img_loss = epoch_img_loss / num_iters
        average_wbb_loss = epoch_wbb_loss / num_iters
        average_MFHA_loss = epoch_MFHA_loss / num_iters
        average_cam_loss = epoch_cam_loss / num_iters
        average_patch_loss = epoch_patch_loss / num_iters
        # average_patch_cls_loss = epoch_patch_cls_loss / num_iters
        # average_patch_supcon_loss = epoch_patch_supcon_loss / num_iters

        self.logger.add_info(
            f"Epoch [{epoch}/{self.config.epochs}]"
            f"Total Loss: {average_total_loss:.4f}, "
            f"Image Contrast Loss: {average_img_loss:.4f}, "
            f"Weak Box Contrast Loss: {average_wbb_loss:.4f}, "
            f"MFHA Loss: {average_MFHA_loss:.4f}, "
            f"CAM Loss: {average_cam_loss:.4f} "
            f"Patch Loss: {average_patch_loss:.4f}\n"
            # f"Patch CLS Loss : {average_patch_cls_loss:.4f} "
            # f"Patch SupCon Loss : {average_patch_supcon_loss:.4f} \n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'Image Contrast Loss': average_img_loss,
            'Weak Box Contrast Loss': average_wbb_loss,
            'MFHA Loss': average_MFHA_loss,
            'CAM Loss': average_cam_loss,
            'Patch Loss': average_patch_loss,
            # 'Patch CLS Loss': average_patch_cls_loss,
            # 'Patch SupCon Loss': average_patch_supcon_loss,
        }
        self.logger.add_metrics(metrics)

    def _save_checkpoint(self, current_epoch : int, checkpoints_save_path : str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": current_epoch,
            "log_path": self.log_path,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "dataset_MPs": self.dataset_MPs,
        }

        file_path = f"{checkpoints_save_path}/checkpoint_{current_epoch}.pth"
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def _save_model(self, model_save_path : str, model_name : str = 'Vgg16_backbone'):
        model_state_dict = self.model.encoder.state_dict()
        model_file_path = f"{model_save_path}/{model_name}.pth"
        torch.save(model_state_dict, model_file_path)
        print(f"Model parameters saved to {model_file_path}")

    def _save_dataset_mps(self, dataset_mps_save_path : str):
        file_path = f"{dataset_mps_save_path}/dataset_MPs.pth"
        torch.save(self.dataset_MPs, file_path)
        print(f"Dataset morphological prototypes saved to {file_path}")

    def train(self)-> None:
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            checkpoints_save_path = self.config.checkpoints_save_path
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.logger.end_train()
        model_save_path = self.config.model_save_path
        self._save_model(model_save_path=model_save_path)
        dataset_mps_save_path = self.config.dataset_mps_save_path
        self._save_dataset_mps(dataset_mps_save_path=dataset_mps_save_path)

class LinearProbTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: LinearProbTrainerConfig,
    )->None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.model = self.model.to(self.config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                # Linear warm‑up
                LinearLR(self.optimizer, start_factor=self.config.warm_up_lr_factor, end_factor=1.0),
                # cosine decay
                CosineAnnealingLR(self.optimizer, T_max=self.config.epochs - self.config.warmup_epochs,
                                  eta_min=self.config.lr * self.config.warm_up_lr_factor)
            ],
            milestones=[self.config.warmup_epochs]
        )

        self.start_epoch = 1

        self.logger = config.logger
        self.log_path = self.logger.get_log_dir()

        # 定义损失函数
        self.loss_func = nn.BCEWithLogitsLoss()

        if self.config.continue_train:
            # 加载检查点
            checkpoint = torch.load(self.config.checkpoint_path)

            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully.")

            # 设置当前训练轮数
            self.start_epoch = checkpoint['epoch'] + 1

            # 加载优化器状态
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            self.optimizer.load_state_dict(optimizer_state_dict)
            print("Optimizer state loaded successfully.")

            # 加载学习率调度器状态
            lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            print("Learning rate scheduler state loaded successfully.")

    def train(self)-> None:
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            checkpoints_save_path = self.config.checkpoints_save_path
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.logger.end_train()
        model_save_path = self.config.model_save_path
        self._save_model(model_save_path=model_save_path)

    def _train_one_epoch(self, epoch) -> None:
        num_iters = len(self.train_loader)
        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_iters,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
            dynamic_ncols=True,  # 自适应终端宽度
        )

        epoch_loss = 0.0
        for iter, (X, Y) in pbar:
            X, Y = X.to(self.config.device), Y.to(self.config.device)
            logits = self.model(X)
            pred_logits = torch.stack([logits[label].squeeze(1) for label in logits.keys()], dim=1)
            loss = self.loss_func(pred_logits, Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "Iter Loss: ": f"{loss.item():.4f} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_loss = epoch_loss / num_iters

        train_metrics = self._get_metrics("train")
        val_metrics = self._get_metrics("val")
        train_acc = train_metrics["acc"]
        val_acc = val_metrics["acc"]
        train_mAP = train_metrics["mAP"]
        val_mAP = val_metrics["mAP"]

        self.logger.add_info(
            f"Epoch [{epoch}/{self.config.epochs}]"
            f"Loss: {average_loss:.4f} "
            f"Train Accuracy: {train_acc * 100:.2f}% "
            f"Val Accuracy: {val_acc * 100:.2f}% "
            f"Train mAP: {train_mAP:.4f} "
            f"Val mAP: {val_mAP:.4f} \n"
        )
        metrics = {
            'Epoch': epoch,
            'Loss': average_loss,
            'Train Accuracy': train_acc,
            'Val Accuracy': val_acc,
            'Train mAP': train_mAP,
            'Val mAP': val_mAP,
        }
        self.logger.add_metrics(metrics)

    def _get_metrics(self, mode : str)-> Dict[str, float]:
        """
        :param mode: 'train' or 'val'
        :return: metrics dict, {"acc": acc, "mAP": mAP}
        """
        if mode == "train":
            loader = self.train_loader
        elif mode == "val":
            loader = self.val_loader
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.model.eval()
        num_classes = 7
        device = self.config.device
        acc_metric = Accuracy(task="multilabel", num_labels=num_classes, average="macro").to(device)
        map_metric = AveragePrecision(task="multilabel", num_labels=num_classes, average="macro").to(device)

        num_iters = len(loader)
        pbar = tqdm(
            enumerate(loader, start=1),
            total=num_iters,
            desc=f"Evaluating {mode} Metrics",
            leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
            dynamic_ncols=True,  # 自适应终端宽度
        )
        with torch.no_grad():
            for i, (X, Y) in pbar:
                X, Y = X.to(self.config.device), Y.to(self.config.device)
                logits = self.model(X)
                logits_tensor = torch.stack(
                    [logits[label].squeeze(1) for label in logits.keys()],
                    dim=1
                )  # [B, C]
                probs = torch.sigmoid(logits_tensor)     # for mAP (continuous)
                preds = (probs > 0.5).int()  # for Acc (binary)
                acc_metric.update(preds, Y.int())
                map_metric.update(probs, Y.int())

        acc = acc_metric.compute().cpu().item()
        mAP = map_metric.compute().cpu().item()

        return {"acc": acc, "mAP": mAP}

    def _save_checkpoint(self, current_epoch : int, checkpoints_save_path : str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": current_epoch,
            "log_path": self.log_path,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

        file_path = f"{checkpoints_save_path}/checkpoint_{current_epoch}.pth"
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def _save_model(self, model_save_path : str, model_name : str = 'Vgg16_backbone_linear_prob'):
        model_state_dict = self.model.encoder.state_dict()
        file_path = f"{model_save_path}/{model_name}.pth"
        torch.save(model_state_dict, file_path)
        print(f"Model parameters saved to {file_path}")


def build_stage1_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    trainer_config: Stage1TrainerConfig,
    supcon_loss_config : SupConLossConfig,
)-> Stage1Trainer:
    trainer = Stage1Trainer(
        model=model,
        train_loader=train_loader,
        config=trainer_config,
        supcon_loss_config=supcon_loss_config,
    )

    return trainer

def build_LinearProb_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trainer_config: LinearProbTrainerConfig,
)-> LinearProbTrainer:
    trainer = LinearProbTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
    )

    return trainer




