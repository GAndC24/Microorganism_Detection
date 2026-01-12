import os
from typing import Callable, Dict, List, Optional, Tuple, Any, TypedDict
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import parse_voc_xml

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

        self.ann_dir = os.path.join(self.root, "Annotations")  # xml annotation folder
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
        xml_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        image = Image.open(img_path).convert("RGB")

        image_info, wbs = parse_voc_xml(xml_path, class_map_encoding)

        boxes : List[Box] = []
        labels : List[int] = []
        for wb in wbs:
            boxes.append(wb['box'])
            labels.append(wb['class_id'])

        target = {
            "boxes" : torch.tensor(boxes, dtype=torch.float32),    # [N, 4]
            "labels" : torch.tensor(labels, dtype=torch.int64),  # [N]
            "image_id" : torch.tensor([idx], dtype=torch.int64),
            # "image_size" : (image_info['width'], image_info['height']),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


