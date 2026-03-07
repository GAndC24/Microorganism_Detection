import os.path
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils import Logger, get_img_contrast_loss, SupConLossConfig, LossContrastMode, supervised_contrastive_loss, get_patch_loss
from tqdm.auto import tqdm
from torchmetrics import MetricCollection, Precision, Recall, F1Score, AUROC, Accuracy, AveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
class Stage2TrainerConfig:
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
    w_ccam_loss: float  # weight for loss_ccam
    w_constrain_loss: float  # weight for loss_constrain
    w_rpn_loss: float   # weight for loss_rpn
    w_obj_loss: float   # weight for loss_obj
    # w_match_loss: float     # weight for matching loss
    ema_alpha: float  # EMA alpha for background prototype update
    continue_train: bool = False
    checkpoint_path: str = None

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
            loss_patch_cls = get_patch_loss(patch_logits, wb_one_hot_labels)
            supcon_loss_config = self.supcon_loss_config
            supcon_loss_config.contrast_mode = LossContrastMode.ONE_VIEW
            loss_patch_supcon = supervised_contrastive_loss(contrast_patch_features, wb_one_hot_labels, supcon_loss_config)
            loss_patch = loss_patch_cls + loss_patch_supcon
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
            epoch_patch_cls_loss += loss_patch_cls.item()
            epoch_patch_supcon_loss += loss_patch_supcon.item()

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "MFHA": f"{loss_MFHA.item():.4f} ",
                "CAM": f"{loss_cam.item():.4f} ",
                "Patch": f"{loss_patch.item():.4f} ",
                "patch_cls" : f"{loss_patch_cls.item():.4f} ",
                "patch_supcon" : f"{loss_patch_supcon.item():.4f} ",
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
        average_patch_cls_loss = epoch_patch_cls_loss / num_iters
        average_patch_supcon_loss = epoch_patch_supcon_loss / num_iters

        self.logger.add_info(
            f"Epoch [{epoch}/{self.config.epochs}]"
            f"Total Loss: {average_total_loss:.4f}, "
            f"Image Contrast Loss: {average_img_loss:.4f}, "
            f"Weak Box Contrast Loss: {average_wbb_loss:.4f}, "
            f"MFHA Loss: {average_MFHA_loss:.4f}, "
            f"CAM Loss: {average_cam_loss:.4f}\n"
            f"Patch Loss: {average_patch_loss:.4f} "
            f"Patch CLS Loss : {average_patch_cls_loss:.4f} "
            f"Patch SupCon Loss : {average_patch_supcon_loss:.4f} \n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'Image Contrast Loss': average_img_loss,
            'Weak Box Contrast Loss': average_wbb_loss,
            'MFHA Loss': average_MFHA_loss,
            'CAM Loss': average_cam_loss,
            'Patch Loss': average_patch_loss,
            'Patch CLS Loss': average_patch_cls_loss,
            'Patch SupCon Loss': average_patch_supcon_loss,
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

class Stage2Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Stage2TrainerConfig,
    )-> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.model.to(self.config.device)

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

        self.bg_prototype : Dict[str, torch.Tensor] = {}

        self.rpn_teacher = copy.deepcopy(self.model.rpn)
        self.rpn_teacher.to(self.config.device)
        for p in self.rpn_teacher.parameters():
            p.requires_grad = False
        self.rpn_teacher.eval()

        self.start_epoch = 1

        self.log_path = self.config.logger.get_log_dir()

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

            # 加载背景原型
            self.bg_prototype = checkpoint['bg_prototype']
            print("Background prototype loaded successfully.")

            # 加载 RPN teacher 参数
            rpn_teacher_state_dict = checkpoint['rpn_teacher_state_dict']
            self.rpn_teacher.load_state_dict(rpn_teacher_state_dict)

    def _build_gt_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        :return: GT targets[i] = {"boxes": [Mi,4], "labels": [Mi]}
        """
        out = []
        for t in targets:
            if ("gt_boxes" in t) and ("gt_labels" in t):
                boxes = t["gt_boxes"]
                labels = t["gt_labels"]
            else:
                boxes = t["boxes"]
                labels = t["labels"]

            out.append({
                "boxes": boxes.float(),
                "labels": labels.long(),
            })
        return out

    def _get_mAP_metric(
        self,
        mode : str, # 'train' or 'eval'
    )-> float:
        """
        :return: mAP@[0.5:0.95]
        """
        self.model.eval()

        if mode == "train":
            loader = self.train_loader
        elif mode == "val":
            loader = self.val_loader

        map_metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=torch.arange(0.5, 0.96, 0.05).tolist(),  # 0.50:0.05:0.95
            max_detection_thresholds=[1, 10, 100],
        ).to(self.config.device)

        num_iters = len(loader)
        pbar = tqdm(
            enumerate(loader, start=1),
            total=num_iters,
            desc=f"Computing {mode} mAP@[0.5:0.95]",
            leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
            dynamic_ncols=True,  # 自适应终端宽度
        )

        with torch.no_grad():
            for iter, (images, target) in pbar:
                images = [img.to(self.config.device) for img in images]
                gt_targets = self._build_gt_targets([{k: v.to(self.config.device) for k, v in t.items()} for t in target])

                X = torch.stack(images, dim=0)
                detections = self.model("inference", X)

                preds = []
                for p in detections:
                    # copy tensors (avoid in-place side effects)
                    boxes = p["boxes"].detach()
                    scores = p["scores"].detach()
                    labels = p["labels"].detach()

                    # drop background (labels <= 0)
                    keep = labels > 0
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    preds.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    })

                targets = []
                for t in gt_targets:
                    boxes = t["boxes"].detach()
                    labels = t["labels"].detach()

                    labels = labels + 1  # shift to [1..C]

                    targets.append({
                        "boxes": boxes,
                        "labels": labels,
                    })

                map_metric.update(preds, targets)

        metric = map_metric.compute()

        return float(metric["map"])

    @torch.no_grad()
    def _update_bg_prototype_ema(
        self,
        batch_bg_prototype : torch.Tensor  # batch background prototype, shape [D]
    )-> None:
        """
        Update background prototype using Exponential Moving Average (EMA).
        :return:
        """

        k = 'bg'
        m = self.config.ema_alpha
        if k not in self.bg_prototype:       # first time initialization
            self.bg_prototype[k] = batch_bg_prototype
        else:
            self.bg_prototype[k] = m * batch_bg_prototype + (1.0 - m) * self.bg_prototype[k]

    @torch.no_grad()
    def _update_rpn_teacher_ema(self) -> None:
        m = float(self.config.ema_alpha)
        student_state = self.model.rpn.state_dict()
        teacher_state = self.rpn_teacher.state_dict()

        for k, student_tensor in student_state.items():
            if not torch.is_tensor(student_tensor):
                teacher_state[k] = student_tensor
                continue

            student_tensor = student_tensor.detach()
            if k in teacher_state and torch.is_floating_point(teacher_state[k]):
                teacher_state[k].mul_(m).add_(student_tensor, alpha=1.0 - m)
            else:
                teacher_state[k] = student_tensor.clone()

    def _train_one_epoch(self, epoch) -> None:
        vis_root = os.path.join(self.log_path, "visualizations")
        vis_dir = os.path.join(vis_root, f"epoch_{epoch}")
        os.makedirs(vis_dir, exist_ok=True)

        num_iters = len(self.train_loader)
        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_iters,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
            dynamic_ncols=True,  # 自适应终端宽度
        )

        epoch_losses : Dict[str, float] = {
            "total_loss" : 0.0,
            "ccam_loss" : 0.0,
            "constrain_loss" : 0.0,
            "proto_loss" : 0.0,
            "pull_loss" : 0.0,
            "push_loss" : 0.0,
            "obj_loss" : 0.0,
            "rpn_loss" : 0.0,
            "rpn_obj_loss" : 0.0,
            "rpn_reg_loss" : 0.0,
            # "match_loss" : 0.0,
            # "match_cls_loss" : 0.0,
            # "match_l1_loss" : 0.0,
            # "match_giou_loss" : 0.0,
        }

        epoch_pseudo_labels_mAP = 0.0

        for iter, (images, target) in pbar:
            images = [img.to(self.config.device) for img in images]
            gt_targets = self._build_gt_targets([{k: v.to(self.config.device) for k, v in t.items()} for t in target])
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in target]

            wboxes = _build_wboxes(targets).to(self.config.device)
            wb_one_hot_labels = _build_wb_one_hot(targets, self.config.num_classes).to(self.config.device)
            X = torch.stack(images, dim=0)
            out = self.model(
                imgs=X,
                wboxes=wboxes,
                wb_labels=wb_one_hot_labels,
                bg_prototype=self.bg_prototype,
                gt_targets=gt_targets,
                rpn_teacher=self.rpn_teacher,
                vis_dir=vis_dir,
                epoch=epoch,
                it=iter
            )

            self._update_bg_prototype_ema(out["batch_bg_prototype"])

            loss_ccam = out["loss_ccam"]

            constrain_losses_dict = out["constrain_losses_dict"]
            loss_constrain = constrain_losses_dict["loss_constrain"]
            loss_proto = constrain_losses_dict["loss_proto"]
            loss_pull = constrain_losses_dict["loss_pull"]
            loss_push = constrain_losses_dict["loss_push"]

            loss_obj = out["loss_obj"]

            rpn_losses_dict = out["rpn_losses_dict"]
            loss_rpn_obj = rpn_losses_dict["loss_objectness"]
            loss_rpn_reg = rpn_losses_dict["loss_rpn_box_reg"]
            loss_rpn = loss_rpn_obj + loss_rpn_reg

            # loss_match = match_losses_dict["loss_match"]
            # loss_match_cls = match_losses_dict["loss_match_cls"]
            # loss_match_l1 = match_losses_dict["loss_match_l1"]
            # loss_match_giou = match_losses_dict["loss_match_giou"]

            loss = self.config.w_ccam_loss * loss_ccam + self.config.w_constrain_loss * loss_constrain + self.config.w_obj_loss * loss_obj + self.config.w_rpn_loss * loss_rpn

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self._update_rpn_teacher_ema()

            epoch_losses["total_loss"] += loss.item()
            epoch_losses["ccam_loss"] += loss_ccam.item()
            epoch_losses["constrain_loss"] += loss_constrain.item()
            epoch_losses["obj_loss"] += loss_obj.item()
            epoch_losses["proto_loss"] += loss_proto.item()
            epoch_losses["pull_loss"] += loss_pull.item()
            epoch_losses["push_loss"] += loss_push.item()
            epoch_losses["rpn_loss"] += loss_rpn.item()
            epoch_losses["rpn_obj_loss"] += loss_rpn_obj.item()
            epoch_losses["rpn_reg_loss"] += loss_rpn_reg.item()
            # epoch_losses["match_loss"] += loss_match.item()

            pseudo_labels_mAP = out["pseudo_labels_mAP"]
            epoch_pseudo_labels_mAP += pseudo_labels_mAP

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "CCAM": f"{loss_ccam.item():.4f} ",
                "RPN": f"{loss_rpn.item():.4f} ",
                "Constrain": f"{loss_constrain.item():.4f} ",
                "Obj" : f"{loss_obj.item():.4f} ",
                # "Match": f"{loss_match.item():.4f} ",
                "p_mAP": f"{pseudo_labels_mAP} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_total_loss = epoch_losses["total_loss"] / num_iters
        average_ccam_loss = epoch_losses["ccam_loss"] / num_iters
        average_constrain_loss = epoch_losses["constrain_loss"] / num_iters
        average_proto_loss = epoch_losses["proto_loss"] / num_iters
        average_pull_loss = epoch_losses["pull_loss"] / num_iters
        average_push_loss = epoch_losses["push_loss"] / num_iters
        average_obj_loss = epoch_losses["obj_loss"] / num_iters
        average_rpn_loss = epoch_losses["rpn_loss"] / num_iters
        average_rpn_obj_loss = epoch_losses["rpn_obj_loss"] / num_iters
        average_rpn_reg_loss = epoch_losses["rpn_reg_loss"] / num_iters
        # average_match_loss = epoch_match_loss / num_iters
        # average_match_cls_loss = epoch_match_cls_loss / num_iters
        # average_match_l1_loss = epoch_match_l1_loss / num_iters
        # average_match_giou_loss = epoch_match_giou_loss / num_iters

        average_pseudo_labels_mAP = epoch_pseudo_labels_mAP / num_iters

        self.config.logger.add_info(
            f"Epoch [{epoch}/{self.config.epochs}]"
            f"Total Loss: {average_total_loss:.4f}, "
            f"CCAM Loss: {average_ccam_loss:.4f}, "
            f"Constrain Loss: {average_constrain_loss:.4f}, "
            f"Obj Loss: {average_obj_loss:.4f}, "
            f"RPN Loss: {average_rpn_loss:.4f},\n"
            # f"Match Loss: {average_match_loss:.4f},\n"
            f"Proto Loss: {average_proto_loss:.4f}, "
            f"Pull Loss: {average_pull_loss:.4f}, "
            f"Push Loss: {average_push_loss:.4f}, "
            f"RPN Obj Loss: {average_rpn_obj_loss:.4f}, "
            f"RPN Reg Loss: {average_rpn_reg_loss:.4f},\n"
            # f"Match CLS Loss: {average_match_cls_loss:.4f}, "
            # f"Match L1 Loss: {average_match_l1_loss:.4f}, "
            # f"Match GIoU Loss: {average_match_giou_loss:.4f}\n"
            f"Pseudo Labels mAP@[0.5:0.95]: {average_pseudo_labels_mAP:.4f}\n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'CCAM Loss': average_ccam_loss,
            'Constrain Loss': average_constrain_loss,
            'Proto Loss' : average_proto_loss,
            'Pull Loss' : average_pull_loss,
            'Push Loss' : average_push_loss,
            'Obj Loss': average_obj_loss,
            'RPN Loss': average_rpn_loss,
            'RPN Obj Loss': average_rpn_obj_loss,
            'RPN Reg Loss': average_rpn_reg_loss,
            # 'Match Loss': average_match_loss,
            # 'Match CLS Loss': average_match_cls_loss,
            # 'Match L1 Loss': average_match_l1_loss,
            # 'Match GIoU Loss': average_match_giou_loss,
            'Pseudo Labels mAP@[0.5:0.95]': average_pseudo_labels_mAP,
        }
        self.config.logger.add_metrics(metrics)

    def train(self)-> None:
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            checkpoints_save_path = self.config.checkpoints_save_path
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.config.logger.end_train()
        model_save_path = self.config.model_save_path
        self._save_model(model_save_path=model_save_path)

    def _save_checkpoint(self, current_epoch : int, checkpoints_save_path : str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "rpn_teacher_state_dict": self.rpn_teacher.state_dict(),
            "bg_prototype": self.bg_prototype,
            "epoch": current_epoch,
            "log_path": self.log_path,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

        file_path = f"{checkpoints_save_path}/checkpoint_{current_epoch}.pth"
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def _save_model(self, model_save_path : str, model_name : str = 'MP_RCNN'):
        model_state_dict = self.model.encoder.state_dict()
        model_file_path = f"{model_save_path}/{model_name}.pth"
        torch.save(model_state_dict, model_file_path)
        print(f"Model parameters saved to {model_file_path}")

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

def build_stage2_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trainer_config: Stage2TrainerConfig,
)-> Stage2Trainer:
    trainer = Stage2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
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




