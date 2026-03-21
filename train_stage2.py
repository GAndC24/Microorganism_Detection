import argparse
import os
import torch
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from datasets import UrinarySedimentDataset, detection_collate_fn
from models import Stage2Config, build_Stage2_model
from utils import Stage2TrainerConfig, build_stage2_trainer, vgg_layer_out_size_ratio_maps, Logger

def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def _get_args()-> argparse:
    p = argparse.ArgumentParser(description="Training Config for Stage 2")

    # config file path
    p.add_argument('--config', type=str, default='./configs/config_stage2.yaml', help='Path to the config file.')

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
    num_classes = data_config["num_classes"]
    print(
        "-----Data Configurations:-----\n"
        f"  Data Root: {data_root}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Image Size: {img_size}\n"
        f"  Num Classes: {num_classes}\n"
    )

    # stage 2 model configurations
    stage2_model_config = config["stage2_model_config"]

    in_c = stage2_model_config["in_c"]
    freeze_backbone = bool(stage2_model_config["freeze_backbone"])

    w_proto_loss = stage2_model_config["w_proto_loss"]
    w_pull_loss = stage2_model_config["w_pull_loss"]
    w_push_loss = stage2_model_config["w_push_loss"]

    w_prototype_sim = stage2_model_config["w_prototype_sim"]
    w_ccam_score = stage2_model_config["w_ccam_score"]
    w_obj_score = stage2_model_config["w_obj_score"]

    keep_iou_thr = stage2_model_config["keep_iou_thr"]

    hidden_dim = stage2_model_config["hidden_dim"]
    embed_dim = stage2_model_config["embed_dim"]

    rpn_anchor_sizes = tuple(_parse_to_list_of_int(stage2_model_config["rpn_anchor_sizes"]))
    rpn_anchor_aspect_ratios = tuple(_parse_to_list_of_float(stage2_model_config["rpn_anchor_aspect_ratios"]))
    rpn_fg_iou_thresh = stage2_model_config["rpn_fg_iou_thresh"]
    rpn_bg_iou_thresh = stage2_model_config["rpn_bg_iou_thresh"]
    rpn_batch_size_per_image = stage2_model_config["rpn_batch_size_per_image"]
    rpn_pre_nms_top_n = _parse_to_list_of_int(stage2_model_config["rpn_pre_nms_top_n"])
    rpn_pre_nms_top_n = {
        "training" : rpn_pre_nms_top_n[0],
        "testing" : rpn_pre_nms_top_n[1]
    }
    rpn_post_nms_top_n = _parse_to_list_of_int(stage2_model_config["rpn_post_nms_top_n"])
    rpn_post_nms_top_n = {
        "training" : rpn_post_nms_top_n[0],
        "testing" : rpn_post_nms_top_n[1]
    }
    rpn_nms_thresh = stage2_model_config["rpn_nms_thresh"]

    ccam_threshold = stage2_model_config["ccam_threshold"]

    cost_class = stage2_model_config["cost_class"]
    cost_bbox = stage2_model_config["cost_bbox"]
    cost_giou = stage2_model_config["cost_giou"]

    rpn_pseudo_nms_thr = stage2_model_config["rpn_pseudo_nms_thr"]
    rpn_pseudo_topk = stage2_model_config["rpn_pseudo_topk"]

    match_focal_alpha = stage2_model_config["match_focal_alpha"]
    match_focal_gamma = stage2_model_config["match_focal_gamma"]
    lambda_match_cls = stage2_model_config["lambda_match_cls"]
    lambda_match_l1 = stage2_model_config["lambda_match_l1"]
    lambda_match_giou = stage2_model_config["lambda_match_giou"]

    roi_out_size_h2wb = stage2_model_config["roi_out_size_h2wb"]
    roi_out_size_h2wb = (roi_out_size_h2wb, roi_out_size_h2wb)

    roi_out_size_h2p = stage2_model_config["roi_out_size_h2p"]
    roi_out_size_h2p = (roi_out_size_h2p, roi_out_size_h2p)

    backbone_weights_path = stage2_model_config["backbone_weights_path"]
    dataset_mps_path = stage2_model_config["dataset_mps_path"]
    print(
        "-----Stage 2 Model Configurations-----\n"
        f"  Input Channels: {in_c}\n"
        f"  Freeze Backbone: {freeze_backbone}\n"
        f"  Weight Proto Loss: {w_proto_loss}\n"
        f"  Weight Pull Loss: {w_pull_loss}\n"
        f"  Weight Push Loss: {w_push_loss}\n"
        f"  Weight Prototype Similarity: {w_prototype_sim}\n"
        f"  Weight CCAM Score: {w_ccam_score}\n"
        f"  Weight Obj Score: {w_obj_score}\n"
        f"  Keep IOU Thresh: {keep_iou_thr}\n"
        f"  Hidden Dimension: {hidden_dim}\n"
        f"  Embedding Dimension: {embed_dim}\n"
        f"  RPN Anchor Sizes: {rpn_anchor_sizes}\n"
        f"  RPN Anchor Aspect Ratios: {rpn_anchor_aspect_ratios}\n"
        f"  RPN FG IOU Thresh: {rpn_fg_iou_thresh}\n"
        f"  RPN BG IOU Thresh: {rpn_bg_iou_thresh}\n"
        f"  RPN Batch Size Per Image: {rpn_batch_size_per_image}\n"
        f"  RPN Pre NMS Top N: {rpn_pre_nms_top_n}\n"
        f"  RPN Post NMS Top N: {rpn_post_nms_top_n}\n"
        f"  RPN NMS Thresh: {rpn_nms_thresh}\n"
        f"  CCAM Threshold: {ccam_threshold}\n"
        f"  Cost Class: {cost_class}\n"
        f"  Cost BBox: {cost_bbox}\n"
        f"  Cost GIOU: {cost_giou}\n"
        f"  RPN Pseudo NMS Thresh: {rpn_pseudo_nms_thr}\n"
        f"  RPN Pseudo TopK: {rpn_pseudo_topk}\n"
        f"  Match Focal Alpha: {match_focal_alpha}\n"
        f"  Match Focal Gamma: {match_focal_gamma}\n"
        f"  Lambda Match Cls: {lambda_match_cls}\n"
        f"  Lambda Match L1: {lambda_match_l1}\n"
        f"  Lambda Match GIOU: {lambda_match_giou}\n"
        f"  ROI Out Size H2WB: {roi_out_size_h2wb}\n"
        f"  ROI Out Size H2P: {roi_out_size_h2p}\n"
        f"  Backbone Weights Path: {backbone_weights_path}\n"
        f"  Dataset MPs Path: {dataset_mps_path}\n"
    )

    # stage 2 trainer configurations
    stage2_trainer_config = config["stage2_trainer_config"]
    epochs = stage2_trainer_config["epochs"]
    lr = stage2_trainer_config["lr"]
    warm_up_lr_factor = stage2_trainer_config["warm_up_lr_factor"]
    warmup_epochs = stage2_trainer_config["warmup_epochs"]
    warmup_phase_epochs = stage2_trainer_config["warmup_phase_epochs"]
    weight_decay = stage2_trainer_config["weight_decay"]
    checkpoints_save_path = stage2_trainer_config["checkpoints_save_path"]
    model_save_path = stage2_trainer_config["model_save_path"]
    continue_train = bool(stage2_trainer_config["continue_train"])
    checkpoint_path = stage2_trainer_config["checkpoint_path"]
    w_ccam_loss = stage2_trainer_config["w_ccam_loss"]
    w_constrain_loss = stage2_trainer_config["w_constrain_loss"]
    w_rpn_loss = stage2_trainer_config["w_rpn_loss"]
    w_obj_loss = stage2_trainer_config["w_obj_loss"]
    w_match_loss= stage2_trainer_config["w_match_loss"]
    ema_alpha = stage2_trainer_config["ema_alpha"]
    print(
        "-----Stage 2 Trainer Configurations-----\n"
        f"  Epochs: {epochs}\n"
        f"  Learning Rate: {lr}\n"
        f"  Min Learning Rate: {lr * warm_up_lr_factor}\n"
        f"  Warmup Epochs: {warmup_epochs}\n"
        f"  Warmup Phase Epochs: {warmup_phase_epochs}\n"
        f"  Weight Decay: {weight_decay}\n"
        f"  Checkpoints Save Path: {checkpoints_save_path}\n"
        f"  Model Save Path: {model_save_path}\n"
        f"  Continue Train: {continue_train}\n"
        f"  Checkpoint Path: {checkpoint_path}\n"
        f"  Weight CCAM Loss: {w_ccam_loss}\n"
        f"  Weight Constrain Loss: {w_constrain_loss}\n"
        f"  Weight RPN Loss: {w_rpn_loss}\n"
        f"  Weight Obj Loss: {w_obj_loss}\n"
        f"  EMA Alpha: {ema_alpha}\n"
        f"  Weight Match Loss: {w_match_loss}\n"
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
    val_dataset = UrinarySedimentDataset(root=data_root, split="val", transforms=transform_aug)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=detection_collate_fn)

    # Initialize Model
    spatial_scale_h2wb = spatial_scale_h2p = vgg_layer_out_size_ratio_maps[29]
    stage2_config = Stage2Config(
        device=device,
        img_size=img_size,
        num_classes=num_classes,
        in_c=in_c,
        freeze_backbone=freeze_backbone,
        w_proto_loss=w_proto_loss,
        w_pull_loss=w_pull_loss,
        w_push_loss=w_push_loss,
        w_prototype_sim=w_prototype_sim,
        w_ccam_score=w_ccam_score,
        w_obj_score=w_obj_score,
        keep_iou_thr=keep_iou_thr,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        rpn_anchor_sizes=rpn_anchor_sizes,
        rpn_anchor_aspect_ratios=rpn_anchor_aspect_ratios,
        rpn_fg_iou_thresh=rpn_fg_iou_thresh,
        rpn_bg_iou_thresh=rpn_bg_iou_thresh,
        rpn_batch_size_per_image=rpn_batch_size_per_image,
        rpn_pre_nms_top_n=rpn_pre_nms_top_n,
        rpn_post_nms_top_n=rpn_post_nms_top_n,
        rpn_nms_thresh=rpn_nms_thresh,
        ccam_threshold=ccam_threshold,
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        rpn_pseudo_nms_thr=rpn_pseudo_nms_thr,
        rpn_pseudo_topk=rpn_pseudo_topk,
        match_focal_alpha=match_focal_alpha,
        match_focal_gamma=match_focal_gamma,
        lambda_match_cls=lambda_match_cls,
        lambda_match_l1=lambda_match_l1,
        lambda_match_giou=lambda_match_giou,
        roi_out_size_h2wb=roi_out_size_h2wb,
        spatial_scale_h2wb = spatial_scale_h2wb,
        roi_out_size_h2p=roi_out_size_h2p,
        spatial_scale_h2p=spatial_scale_h2p
    )
    model = build_Stage2_model(stage2_config, backbone_weights_path, dataset_mps_path)

    # Initialize Trainer
    if continue_train:
        checkpoint = torch.load(checkpoint_path)
        logger = Logger(model_name="Stage2", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="Stage2", config=config)

    stage2_trainer_config = Stage2TrainerConfig(
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        warm_up_lr_factor=warm_up_lr_factor,
        warmup_epochs=warmup_epochs,
        # warmup_phase_epochs=warmup_phase_epochs,
        weight_decay=weight_decay,
        checkpoints_save_path=checkpoints_save_path,
        model_save_path=model_save_path,
        logger=logger,
        w_ccam_loss=w_ccam_loss,
        w_constrain_loss=w_constrain_loss,
        w_rpn_loss=w_rpn_loss,
        w_obj_loss=w_obj_loss,
        # w_match_loss=w_match_loss,
        ema_alpha=ema_alpha,
        continue_train = continue_train,
        checkpoint_path = checkpoint_path,
    )
    trainer = build_stage2_trainer(model, train_loader, val_loader, stage2_trainer_config)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()