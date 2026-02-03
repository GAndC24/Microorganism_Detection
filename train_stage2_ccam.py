import argparse
import os
import torch
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from datasets import UrinarySedimentDataset, detection_collate_fn
from models import build_Stage2CCAM_model, Stage2CCAMConfig
from utils import Stage2CCAMTrainerConfig, build_stage2_ccam_trainer, vgg_layer_out_size_ratio_maps, Logger

def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def _get_args()-> argparse:
    p = argparse.ArgumentParser(description="Training Config for Stage 2 CCAM")

    # config file path
    p.add_argument('--config', type=str, default='./configs/config_stage2CCAM.yaml', help='Path to the config file.')

    return p.parse_args()

def _parse_to_list_of_int(x, allow_range=True):
    """
    将配置中的参数解析为 List[int]
    支持：
      - list[int]
      - "1,2,3"
      - "1-4"
      - 单个 int
    """
    if x is None:
        return None

    # 已经是 list
    if isinstance(x, list):
        return [int(v) for v in x]

    # 单个 int
    if isinstance(x, int):
        return [x]

    # 字符串
    if isinstance(x, str):
        x = x.strip()

        # 范围形式 "a-b"
        if allow_range and "-" in x:
            start, end = x.split("-")
            return list(range(int(start), int(end) + 1))

        # 逗号分隔 "a,b,c"
        return [int(v) for v in x.split(",")]

    raise ValueError(f"Unsupported type for parsing: {type(x)}")

def _parse_to_list_of_float(
    x: Union[List[float], float, int, str, None],
    allow_range: bool = True,
    range_step: float = 1.0,
) -> Optional[List[float]]:
    """
    将配置中的参数解析为 List[float]

    支持：
      - list[float] / list[int]
      - "1.0,2.5,3"
      - "1.0-4.0"（需指定 range_step）
      - 单个 float / int

    :param allow_range: 是否允许 a-b 形式
    :param range_step: float 范围展开时的步长（仅在 a-b 时使用）
    """
    if x is None:
        return None

    # 已经是 list
    if isinstance(x, list):
        return [float(v) for v in x]

    # 单个数值
    if isinstance(x, (int, float)):
        return [float(x)]

    # 字符串
    if isinstance(x, str):
        x = x.strip()

        # 范围形式 "a-b"
        if allow_range and "-" in x:
            start_str, end_str = x.split("-")
            start = float(start_str)
            end = float(end_str)

            if range_step <= 0:
                raise ValueError("range_step must be positive for float range")

            values = []
            v = start
            # 避免浮点误差导致遗漏 end
            while v <= end + 1e-9:
                values.append(round(v, 10))
                v += range_step
            return values

        # 逗号分隔 "a,b,c"
        return [float(v) for v in x.split(",")]

    raise ValueError(f"Unsupported type for parsing: {type(x)}")


def main()-> None:
    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    config = _load_yaml(args.config)

    # runtime configurations
    runtime_config = config["runtime"]
    device = torch.device(runtime_config["device"])
    seed = runtime_config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(
        "-----Runtime Configurations-----\n"
        f"  Device: {device}\n"
        f"  Seed: {seed}\n"
    )

    # data configurations
    data_config = config["data"]
    data_root = data_config["data_root"]
    batch_size = data_config["batch_size"]
    img_size = data_config["img_size"]
    img_size = (img_size, img_size)
    print(
        "-----Data Configurations:-----\n"
        f"  Data Root: {data_root}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Image Size: {img_size}\n"
    )

    # stage 2 CCAM model configurations
    stage2_ccam_model_config = config["stage2_ccam_model_config"]
    in_c = stage2_ccam_model_config["in_c"]
    freeze_backbone = bool(stage2_ccam_model_config["freeze_backbone"])
    ccam_threshold = stage2_ccam_model_config["ccam_threshold"]
    roi_out_size_h2wb = stage2_ccam_model_config["roi_out_size_h2wb"]
    roi_out_size_h2wb = (roi_out_size_h2wb, roi_out_size_h2wb)
    backbone_weights_path = stage2_ccam_model_config["backbone_weights_path"]
    print(
        "-----Stage 2 CCAM Model Configurations-----\n"
        f"  Input Channels: {in_c}\n"
        f"  Freeze Backbone: {freeze_backbone}\n"
        f"  CCAM Threshold: {ccam_threshold}\n"
        f"  ROI Out Size H2WB: {roi_out_size_h2wb}\n"
        f"  Backbone Weights Path: {backbone_weights_path}\n"
    )

    # stage 2 CCAM trainer configurations
    stage2_ccam_trainer_config = config["stage2_ccam_trainer_config"]
    num_classes = stage2_ccam_trainer_config["num_classes"]
    epochs = stage2_ccam_trainer_config["epochs"]
    lr = stage2_ccam_trainer_config["lr"]
    warm_up_lr_factor = stage2_ccam_trainer_config["warm_up_lr_factor"]
    warmup_epochs = stage2_ccam_trainer_config["warmup_epochs"]
    weight_decay = stage2_ccam_trainer_config["weight_decay"]
    checkpoints_save_path = stage2_ccam_trainer_config["checkpoints_save_path"]
    model_save_path = stage2_ccam_trainer_config["model_save_path"]
    continue_train = bool(stage2_ccam_trainer_config["continue_train"])
    checkpoint_path = stage2_ccam_trainer_config["checkpoint_path"]
    print(
        "-----Stage 2 CCAM Trainer Configurations-----\n"
        f"  Number of Classes: {num_classes}\n"
        f"  Epochs: {epochs}\n"
        f"  Learning Rate: {lr}\n"
        f"  Min Learning Rate: {lr * warm_up_lr_factor}\n"
        f"  Warmup Epochs: {warmup_epochs}\n"
        f"  Weight Decay: {weight_decay}\n"
        f"  Checkpoints Save Path: {checkpoints_save_path}\n"
        f"  Model Save Path: {model_save_path}\n"
        f"  Continue Train: {continue_train}\n"
        f"  Checkpoint Path: {checkpoint_path}\n"
    )

    # Initialize Dataset
    # data preprocessing and augmentations transforms
    transform_aug = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 轻度几何扰动
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(degrees=15),
        # 亮度/对比度扰动
        T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.8),
        T.RandomAutocontrast(p=0.2),
        # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    ])

    train_dataset = UrinarySedimentDataset(root=data_root, split="train", transforms=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=detection_collate_fn)

    # Initialize Model
    spatial_scale_h2wb = vgg_layer_out_size_ratio_maps[29]
    stage2_ccam_config = Stage2CCAMConfig(
        device=device,
        in_c=in_c,
        freeze_backbone=freeze_backbone,
        ccam_threshold=ccam_threshold,
        roi_out_size_h2wb=roi_out_size_h2wb,
        spatial_scale_h2wb = spatial_scale_h2wb
    )
    model = build_Stage2CCAM_model(stage2_ccam_config, backbone_weights_path)

    # Initialize Trainer
    if continue_train:
        checkpoint = torch.load(checkpoint_path)
        logger = Logger(model_name="Stage2_CCAM", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="Stage2_CCAM", config=config)

    stage2_ccam_trainer_config = Stage2CCAMTrainerConfig(
        device=device,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        warm_up_lr_factor=warm_up_lr_factor,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        checkpoints_save_path=checkpoints_save_path,
        model_save_path=model_save_path,
        logger=logger,
        continue_train=continue_train,
        checkpoint_path=checkpoint_path,
    )
    trainer = build_stage2_ccam_trainer(model, train_loader, stage2_ccam_trainer_config)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()