import os
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import parse_voc_xml
from torchvision import tv_tensors

Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)

class_map_encoding = {'cast' : 0, 'cryst' : 1, 'epith' : 2, 'epithn' : 3, 'eryth' : 4, 'leuko' : 5, 'mycete' : 6}
class_map_decoding = {v: k for k, v in class_map_encoding.items()}

def detection_collate_fn(batch : List)-> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    '''batch : List of (image, target)'''
    images, targets = zip(*batch)       # 拆成两个元组
    return list(images), list(targets)

class UrinarySedimentDataset(Dataset):
    def __init__(
        self,
        root: str,      # dataset root path
        split: str,       # dataset split, "train" or "val" or "test"
        transforms: Optional[Callable] = None,      # data transforms
    )-> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.split = split
        self.transforms = transforms

        self.ann_wb_dir = os.path.join(self.root, "Annotations_wb")  # wb xml annotation folder
        self.ann_gt_dir = os.path.join(self.root, "Annotations_gt")  # gt xml annotation folder
        self.img_dir = os.path.join(self.root, "JPEGImages")  # jpg images folder
        self.set_dir = os.path.join(self.root, "ImageSets", "Main")  # dataset split folder

        # load images
        imageset_txt = os.path.join(self.set_dir, f"{split}.txt")
        if not os.path.isfile(imageset_txt):
            raise FileNotFoundError(f"ImageSet file not found: {imageset_txt}")
        with open(imageset_txt, "r", encoding="utf-8") as f:
            self.img_ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, Dict[str, Any]]:
        image_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        wb_xml_path = os.path.join(self.ann_wb_dir, f"{image_id}.xml")
        gt_xml_path = os.path.join(self.ann_gt_dir, f"{image_id}.xml")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.isfile(wb_xml_path):
            raise FileNotFoundError(f"Annotation not found: {wb_xml_path}")
        if not os.path.isfile(gt_xml_path):
            raise FileNotFoundError(f"Annotation not found: {gt_xml_path}")

        image_pil = Image.open(img_path).convert("RGB")
        W, H = image_pil.size  # PIL: (W, H)

        image_info, wbs = parse_voc_xml(wb_xml_path, class_map_encoding)
        _, gts = parse_voc_xml(gt_xml_path, class_map_encoding)

        boxes : List[Box] = []
        labels : List[int] = []
        gt_boxes : List[Box] = []
        gt_labels : List[int] = []
        for wb in wbs:
            boxes.append(wb['box'])
            labels.append(wb['class_id'])
        for gt in gts:
            gt_boxes.append(gt['box'])
            gt_labels.append(gt['class_id'])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)  # [N,4]
        labels_tensor = torch.tensor(labels, dtype=torch.int64)  # [N]
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)   # [N,4]
        gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.int64)   # [N]

        image = tv_tensors.Image(image_pil)
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format="XYXY",
            canvas_size=(H, W)
        )
        gt_boxes_tv = tv_tensors.BoundingBoxes(
            gt_boxes_tensor,
            format="XYXY",
            canvas_size=(H, W)
        )
        target: Dict[str, Any] = {
            "boxes": boxes_tv,
            "labels": labels_tensor,
            "gt_boxes": gt_boxes_tv,
            "gt_labels": gt_labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

class LinearProbDataset(Dataset):
    def __init__(
        self,
        root: str,  # dataset root path
        split: str,  # dataset split, "train" or "val" or "test"
        transforms: Optional[Callable] = None,  # data transforms
    )-> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.split = split
        self.transforms = transforms

        self.ann_dir = os.path.join(self.root, "Annotations_wb")  # xml annotation folder
        self.img_dir = os.path.join(self.root, "JPEGImages")  # jpg images folder
        self.set_dir = os.path.join(self.root, "ImageSets", "Main")  # dataset split folder

        # load images
        imageset_txt = os.path.join(self.set_dir, f"{split}.txt")
        if not os.path.isfile(imageset_txt):
            raise FileNotFoundError(f"ImageSet file not found: {imageset_txt}")
        with open(imageset_txt, "r", encoding="utf-8") as f:
            self.img_ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        image_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        xml_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        image_pil = Image.open(img_path).convert("RGB")

        image_info, wbs = parse_voc_xml(xml_path, class_map_encoding)

        labels: List[int] = []
        for wb in wbs:
            labels.append(wb['class_id'])

        # get multi-hot labels
        multi_hot_label = torch.zeros(7)
        for idx in labels:
            multi_hot_label[idx] = 1.0

        image = self.transforms(image_pil)

        return image, multi_hot_label



