import argparse
import os
import torch
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from datasets import UrinarySedimentDataset, detection_collate_fn
from models import MPRCNNConfig, build_MP_RCNN_model
from utils import *

def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def _get_args()-> argparse:
    p = argparse.ArgumentParser(description="Training Config for MP_RCNN")

    # config file path
    p.add_argument('--config', type=str, default='./configs/config_mp_rcnn.yaml', help='Path to the config file.')


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
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    args = _get_args()

    config = _load_yaml(args.config)

    # runtime configurations
    runtime_config = config["runtime"]

    device = torch.device(runtime_config["device"])

    seed = runtime_config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # data configurations
    data_config = config["data"]

    img_size = data_config["img_size"]
    img_size = (img_size, img_size)

    # model configurations
    model_config = config["model"]

    freeze_backbone = bool(model_config["freeze_backbone"])

    rpn_anchor_sizes = tuple(_parse_to_list_of_int(model_config["rpn_anchor_sizes"]))
    rpn_anchor_aspect_ratios = tuple(_parse_to_list_of_float(model_config["rpn_anchor_aspect_ratios"]))
    rpn_pre_nms_top_n = _parse_to_list_of_int(model_config["rpn_pre_nms_top_n"])
    rpn_pre_nms_top_n = {
        "training" : rpn_pre_nms_top_n[0],
        "testing" : rpn_pre_nms_top_n[1]
    }
    rpn_post_nms_top_n = _parse_to_list_of_int(model_config["rpn_post_nms_top_n"])
    rpn_post_nms_top_n = {
        "training" : rpn_post_nms_top_n[0],
        "testing" : rpn_post_nms_top_n[1]
    }

    roi_out_size_h2p = model_config["roi_out_size_h2p"]
    roi_out_size_h2p = (roi_out_size_h2p, roi_out_size_h2p)
    spatial_scale_h2p = vgg_layer_out_size_ratio_maps[29]


    # trainer configurations
    trainer_config = config["trainer"]

    continue_train = bool(trainer_config["continue_train"])


    # Initialize Dataset
    # data preprocessing and augmentations transforms
    transform_aug = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 轻度几何扰动
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(degrees=15),
        # 亮度/对比度扰动
        T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.8),
        # T.RandomAutocontrast(p=0.2),
        # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    ])

    train_dataset = UrinarySedimentDataset(root=data_config["data_root"], split="train", transforms=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=data_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True, collate_fn=detection_collate_fn)
    val_dataset = UrinarySedimentDataset(root=data_config["data_root"], split="val", transforms=transform_aug)
    val_loader = DataLoader(val_dataset, batch_size=data_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, collate_fn=detection_collate_fn)


    # Initialize Model
    mp_rcnn_config = MPRCNNConfig(
        device=device,
        img_size=img_size,
        num_classes=data_config["num_classes"],
        in_c=model_config["in_c"],
        freeze_backbone=freeze_backbone,
        hidden_dim=model_config["hidden_dim"],
        rpn_anchor_sizes=rpn_anchor_sizes,
        rpn_anchor_aspect_ratios=rpn_anchor_aspect_ratios,
        rpn_fg_iou_thresh=model_config["rpn_fg_iou_thresh"],
        rpn_bg_iou_thresh=model_config["rpn_bg_iou_thresh"],
        rpn_batch_size_per_image=model_config["rpn_batch_size_per_image"],
        rpn_pre_nms_top_n=rpn_pre_nms_top_n,
        rpn_post_nms_top_n=rpn_post_nms_top_n,
        rpn_nms_thresh=model_config["rpn_nms_thresh"],
        det_fg_iou_thresh=model_config["det_fg_iou_thresh"],
        det_bg_iou_thresh=model_config["det_bg_iou_thresh"],
        det_batch_size_per_image=model_config["det_batch_size_per_image"],
        det_positive_fraction=model_config["det_positive_fraction"],
        det_score_thresh=model_config["det_score_thresh"],
        det_nms_thresh=model_config["det_nms_thresh"],
        detections_per_img=model_config["detections_per_img"],
        roi_out_size_h2p=roi_out_size_h2p,
        spatial_scale_h2p=spatial_scale_h2p
    )
    model = build_MP_RCNN_model(mp_rcnn_config, model_config["backbone_weights_path"])

    # Initialize Trainer
    if continue_train:
        checkpoint = torch.load(trainer_config["checkpoint_path"])
        logger = Logger(model_name="MP R-CNN", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="MP R-CNN", config=config)


    mp_rcnn_trainer_config = MP_RCNNTrainerConfig(
        num_classes=data_config["num_classes"],
        device=device,
        epochs=trainer_config["epochs"],
        lr=trainer_config["lr"],
        warm_up_lr_factor=trainer_config["warm_up_lr_factor"],
        warmup_epochs=trainer_config["warmup_epochs"],
        weight_decay=trainer_config["weight_decay"],
        checkpoints_save_path=trainer_config["checkpoints_save_path"],
        model_save_path=trainer_config["model_save_path"],
        logger=logger,
        continue_train = continue_train,
        checkpoint_path = trainer_config["checkpoint_path"],
    )
    trainer = build_MP_RCNN_trainer(model, train_loader, val_loader, mp_rcnn_trainer_config)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()