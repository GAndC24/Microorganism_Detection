import argparse
import os
import torch
import yaml
from typing import Dict, Any, List
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from datasets import UrinarySedimentDataset
from models import PrototypeCheckerConfig, build_PrototypeChecker_model
from tqdm.auto import tqdm

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

def _build_boxes(targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Construct boxes tensor from targets.
    :param targets: List of dictionaries containing bounding box information.
    :return: GT boxes tensor of shape [R, 5], where each box is [batch_idx, x1, y1, x2, y2].
    """
    gt_boxes = []
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]  # Convert to [x1, y1, x2, y2]
        batch_indices = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        gt_boxes.append(torch.cat([batch_indices, boxes], dim=1))  # Combine batch_idx with boxes

    return torch.cat(gt_boxes, dim=0)  # Concatenate all boxes across the batch

def _build_boxes_label(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct one-hot labels for a batch of boxes
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : one-hot tensor, [R = num_boxes, num_classes]
    """
    all_labels = []
    for target in targets:
        all_labels.append(target["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # [R]
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32).to(all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels

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
    num_classes = data_config["num_classes"]
    print(
        "-----Data Configurations:-----\n"
        f"  Data Root: {data_root}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Image Size: {img_size}\n"
        f"  Num Classes: {num_classes}\n"
    )

    # PrototypeChecker model configurations
    prototypeChecker_config = config["prototypeChecker_config"]
    in_c = prototypeChecker_config["in_c"]
    embed_dim = prototypeChecker_config["embed_dim"]
    patch_size = prototypeChecker_config["patch_size"]
    roi_out_size = prototypeChecker_config["roi_out_size"]
    spatial_scale = prototypeChecker_config["spatial_scale"]
    sampling_ratio = prototypeChecker_config["sampling_ratio"]
    backbone_weights_path = prototypeChecker_config["backbone_weights_path"]
    patch_embed_weights_path = prototypeChecker_config["patch_embed_weights_path"]
    dataset_MPs_path = prototypeChecker_config["dataset_MPs_path"]
    print(
        "-----PrototypeChecker Model Configurations-----\n"
        f"  Input Channels: {in_c}\n"
        f"  Embed Dim: {embed_dim}\n"
        f"  Patch Size: {patch_size}\n"
        f"  ROI Out Size: {roi_out_size}\n"
        f"  Spatial Scale: {spatial_scale}\n"
        f"  Sampling Ratio: {sampling_ratio}\n"
        f"  Backbone Weights Path: {backbone_weights_path}\n"
        f"  Patch Embed Weights Path: {patch_embed_weights_path}\n"
    )

    # Initialize Dataset
    # data preprocessing and augmentations transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = UrinarySedimentDataset(root=data_root, split="train", transforms=transform)
    val_dataset = UrinarySedimentDataset(root=data_root, split="val", transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    prototypeChecker_config = PrototypeCheckerConfig(
        in_c=in_c,
        embed_dim=embed_dim,
        patch_size=patch_size,
        roi_out_size=roi_out_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio
    )
    model = build_PrototypeChecker_model(prototypeChecker_config, backbone_weights_path, patch_embed_weights_path)
    model = model.to(device)
    model.eval()

    # Load Dataset Morphological Prototypes
    prototypes = torch.load(dataset_MPs_path).to(device)

    # Compute similarity on train data
    num_iters = len(train_loader)
    pbar = tqdm(
        enumerate(train_loader, start=1),
        total=num_iters,
        desc=f"Computing Similarity",
        leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
        dynamic_ncols=True,  # 自适应终端宽度
    )

    sims = {k: 0.0 for k in range(num_classes)}
    with torch.no_grad():
        for iter, (images, target) in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in target]

            boxes = _build_boxes(targets)
            boxes_labels = _build_boxes_label(targets, num_classes)

            batch_sims = model(images, boxes, boxes_labels, prototypes)
            for cls_id, sim in batch_sims.items():
                sims[cls_id] += sim.item()

    average_sims = {k: v / num_iters for k, v in sims.items()}
    print("-----Prototype & GT Similarities on Train Set-----")
    for cls_id, avg_sim in average_sims.items():
        print(f"  Class {cls_id}: Average Similarity = {avg_sim:.4f}")

    # Compute similarity on val data
    num_iters = len(val_loader)
    pbar = tqdm(
        enumerate(val_loader, start=1),
        total=num_iters,
        desc=f"Computing Similarity",
        leave=False,  # 一个epoch结束后不保留整条进度条（日志更干净）
        dynamic_ncols=True,  # 自适应终端宽度
    )

    sims = {k: 0.0 for k in range(num_classes)}
    with torch.no_grad():
        for iter, (images, target) in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in target]

            boxes = _build_boxes(targets)
            boxes_labels = _build_boxes_label(targets, num_classes)

            batch_sims = model(images, boxes, boxes_labels, prototypes)
            for cls_id, sim in batch_sims.items():
                sims[cls_id] += sim.item()
    average_sims = {k: v / num_iters for k, v in sims.items()}
    print("-----Prototype & GT Similarities on Val Set-----")
    for cls_id, avg_sim in average_sims.items():
        print(f"  Class {cls_id}: Average Similarity = {avg_sim:.4f}")

if __name__ == "__main__":
    main()