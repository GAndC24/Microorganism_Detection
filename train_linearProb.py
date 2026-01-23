import argparse
import os
import torch
import yaml
from typing import Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from datasets import LinearProbDataset
from models import LinearProbConfig, build_LinearProb_model
from utils import LinearProbTrainerConfig, build_LinearProb_trainer, Logger

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
    p.add_argument('--config', type=str, default='./configs/config_linearProb.yaml', help='Path to the config file.')

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
    img_size = data_config["img_size"]
    print(
        "-----Data Configurations:-----\n"
        f"  Data Root: {data_root}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Image Size: {img_size}\n"
    )

    # linearProb model configurations
    linearProb_model_config = config["linearProb_model_config"]
    in_c = linearProb_model_config["in_c"]
    in_size = linearProb_model_config["in_size"]
    out_dim = linearProb_model_config["out_dim"]
    backbone_weights_path = linearProb_model_config["backbone_weights_path"]
    print(
        "-----LinearProb Model Configurations-----\n"
        f"  Input Channels: {in_c}\n"
        f"  Input Size: {in_size}\n"
        f"  Output Dimension: {out_dim}\n"
        f"  Backbone Weights Path: {backbone_weights_path}\n"
    )

    # linearProb trainer configurations
    linearProb_trainer_config = config["linearProb_trainer_config"]
    num_classes = linearProb_trainer_config["num_classes"]
    epochs = linearProb_trainer_config["epochs"]
    lr = linearProb_trainer_config["lr"]
    warm_up_lr_factor = linearProb_trainer_config["warm_up_lr_factor"]
    warmup_epochs = linearProb_trainer_config["warmup_epochs"]
    weight_decay = linearProb_trainer_config["weight_decay"]
    checkpoints_save_path = linearProb_trainer_config["checkpoints_save_path"]
    model_save_path = linearProb_trainer_config["model_save_path"]
    continue_train = bool(linearProb_trainer_config["continue_train"])
    checkpoint_path = linearProb_trainer_config["checkpoint_path"]
    print(
        "-----LinearProb Trainer Configurations-----\n"
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
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LinearProbDataset(root=data_root, split="train", transforms=transform)
    val_dataset = LinearProbDataset(root=data_root, split="val", transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    linearProb_model_config = LinearProbConfig(
        in_c=in_c,
        in_size=in_size,
        out_dim=out_dim,
    )
    model = build_LinearProb_model(linearProb_model_config, backbone_weights_path)

    # Initialize Trainer
    if continue_train:
        checkpoint = torch.load(checkpoint_path)
        logger = Logger(model_name="LinearProb", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="LinearProb", config=config)

    linearProb_trainer_config = LinearProbTrainerConfig(
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        warm_up_lr_factor=warm_up_lr_factor,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        checkpoints_save_path=checkpoints_save_path,
        model_save_path=model_save_path,
        continue_train=continue_train,
        checkpoint_path=checkpoint_path,
        logger=logger,
        device=device,
    )
    trainer = build_LinearProb_trainer(model, train_loader, val_loader, linearProb_trainer_config)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()