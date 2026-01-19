import argparse
import os
import torch
import yaml
from typing import Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from datasets import UrinarySedimentDataset, detection_collate_fn
from models import Stage1Config, build_Stage1_model
from utils import Stage1TrainerConfig, WBBLossConfig, build_stage1_trainer, vgg_layer_out_c_maps, vgg_layer_out_size_ratio_maps, Logger

def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def _get_args()-> argparse:
    p = argparse.ArgumentParser(description="Training Config for Stage 1")

    # config file path
    p.add_argument('--config', type=str, default='./configs/config_stage1.yaml', help='Path to the config file.')

    return p.parse_args()

def _parse_to_list(x, allow_range=True):
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
    print(
        "-----Data Configurations:-----\n"
        f"  Data Root: {data_root}\n"
        f"  Batch Size: {batch_size}\n"
    )

    # stage 1 model configurations
    stage1_model_config = config["stage1_model_config"]
    num_classes = stage1_model_config["num_classes"]
    embed_dim = stage1_model_config["embed_dim"]
    img_size = stage1_model_config["img_size"]
    img_size = (img_size, img_size)
    hidden_dim = stage1_model_config["hidden_dim"]
    layer_indices = _parse_to_list(stage1_model_config["layer_indices"])
    mask_threshold = stage1_model_config["mask_threshold"]
    gaussian_sigma = stage1_model_config["gaussian_sigma"]
    patch_size = stage1_model_config["patch_size"]
    components_range = _parse_to_list(stage1_model_config["components_range"])
    random_state = stage1_model_config["random_state"]
    max_iter = stage1_model_config["max_iter"]
    roi_out_size_mid = stage1_model_config["roi_out_size_mid"]
    roi_out_size_mid = (roi_out_size_mid, roi_out_size_mid)
    roi_out_size_high = stage1_model_config["roi_out_size_high"]
    roi_out_size_high = (roi_out_size_high, roi_out_size_high)
    sampling_ratio = stage1_model_config["sampling_ratio"]
    print(
        "-----Stage 1 Model Configurations-----\n"
        f"  Num Classes: {num_classes}\n"
        f"  Embed Dim: {embed_dim}\n"
        f"  Image Size: {img_size}\n"
        f"  Hidden Dim: {hidden_dim}\n"
        f"  Layer Indices: {layer_indices}\n"
        f"  Mask Threshold: {mask_threshold}\n"
        f"  Gaussian Sigma: {gaussian_sigma}\n"
        f"  Patch Size: {patch_size}\n"
        f"  Components Range: {components_range}\n"
        f"  Random State: {random_state}\n"
        f"  Max Iter: {max_iter}\n"
        f"  ROI Out Size Mid: {roi_out_size_mid}\n"
        f"  ROI Out Size High: {roi_out_size_high}\n"
        f"  Sampling Ratio: {sampling_ratio}\n"
    )

    # stage 1 trainer configurations
    stage1_trainer_config = config["stage1_trainer_config"]
    epochs = stage1_trainer_config["epochs"]
    lr = stage1_trainer_config["lr"]
    warm_up_lr_factor = stage1_trainer_config["warm_up_lr_factor"]
    warmup_epochs = stage1_trainer_config["warmup_epochs"]
    weight_decay = stage1_trainer_config["weight_decay"]
    checkpoints_save_path = stage1_trainer_config["checkpoints_save_path"]
    model_save_path = stage1_trainer_config["model_save_path"]
    dataset_mps_save_path = stage1_trainer_config["dataset_mps_save_path"]
    continue_train = bool(stage1_trainer_config["continue_train"])
    checkpoint_path = stage1_trainer_config["checkpoint_path"]
    w_img_loss = stage1_trainer_config["w_img_loss"]
    w_wbb_loss = stage1_trainer_config["w_wbb_loss"]
    w_cam_loss = stage1_trainer_config["w_cam_loss"]
    w_patch_loss = stage1_trainer_config["w_patch_loss"]
    mp_ema_alpha = stage1_trainer_config["mp_ema_alpha"]
    print(
        "-----Stage 1 Trainer Configurations-----\n"
        f"  Epochs: {epochs}\n"
        f"  Learning Rate: {lr}\n"
        f"  Min Learning Rate: {lr * warm_up_lr_factor}\n"
        f"  Warmup Epochs: {warmup_epochs}\n"
        f"  Weight Decay: {weight_decay}\n"
        f"  Checkpoints Save Path: {checkpoints_save_path}\n"
        f"  Model Save Path: {model_save_path}\n"
        f"  Dataset MPs Save Path: {dataset_mps_save_path}\n"
        f"  Continue Train: {continue_train}\n"
        f"  Checkpoint Path: {checkpoint_path}\n"
        f"  Weight Image Loss: {w_img_loss}\n"
        f"  Weight Weak Box Loss: {w_wbb_loss}\n"
        f"  Weight CAM Loss: {w_cam_loss}\n"
        f"  Weight Patch Loss: {w_patch_loss}\n"
        f"  MP EMA Alpha: {mp_ema_alpha}\n"
    )

    # weak box contrast loss configurations
    wbb_loss_config = config["wbb_loss_config"]
    temperature = wbb_loss_config["temperature"]
    positives_cap = wbb_loss_config["positives_cap"]
    print(
        "-----Weak Box Contrast Loss Configurations-----\n"
        f"  Temperature: {temperature}\n"
        f"  Positives Cap: {positives_cap}\n"
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
    in_c = vgg_layer_out_c_maps[layer_indices[1]]
    spatial_scale_mid = vgg_layer_out_size_ratio_maps[layer_indices[1]]
    spatial_scale_high = vgg_layer_out_size_ratio_maps[layer_indices[2]]
    stage1_config = Stage1Config(
        num_classes=num_classes,
        embed_dim=embed_dim,
        img_size=img_size,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        layer_indices=layer_indices,
        mask_threshold=mask_threshold,
        gaussian_sigma=gaussian_sigma,
        in_c=in_c,
        patch_size=patch_size,
        components_range=components_range,
        random_state=random_state,
        max_iter=max_iter,
        roi_out_size_mid=roi_out_size_mid,
        roi_out_size_high=roi_out_size_high,
        spatial_scale_mid=spatial_scale_mid,
        spatial_scale_high=spatial_scale_high,
        sampling_ratio=sampling_ratio,
    )
    model = build_Stage1_model(stage1_config)

    # Initialize Trainer
    wbb_loss_config = WBBLossConfig(
        temperature=temperature,
        positives_cap=positives_cap
    )

    if continue_train:
        checkpoint = torch.load(checkpoint_path)
        logger = Logger(model_name="Stage1", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="Stage1", config=config)

    stage1_trainer_config = Stage1TrainerConfig(
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        warm_up_lr_factor=warm_up_lr_factor,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        checkpoints_save_path=checkpoints_save_path,
        model_save_path=model_save_path,
        dataset_mps_save_path=dataset_mps_save_path,
        logger=logger,
        continue_train=continue_train,
        checkpoint_path=checkpoint_path,
        w_img_loss=w_img_loss,
        w_wbb_loss=w_wbb_loss,
        w_cam_loss=w_cam_loss,
        w_patch_loss=w_patch_loss,
        mp_ema_alpha=mp_ema_alpha
    )
    trainer = build_stage1_trainer(model, train_loader, stage1_trainer_config, wbb_loss_config)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()