# Construct Weak Bounding Boxes
from typing import List, Tuple, Dict, Optional, Iterable
import heapq
import math

Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)
LabeledBox = Tuple[Box, int]              # (box, class_id)
WeakBox = Tuple[Box, int]                 # (weak_box, class_id)

class Cluster:
    """一个 cluster 对应同一类别的一组实例框。"""
    boxes: List[Box]
    cls: int

# Calculate the area of a bounding box
def area(b: Box) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

# Merge the proposal boxes to create weak boxes
def union_rect(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)

# Check if the center of a box is within a given rectangle(new weak box)
def center_in_rect(box: Box, rect: Box) -> bool:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = rect
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

# Calculate Intersection over Union (IoU) between two boxes
def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0

# Check the purity of a candidate weak box
def purity_check_center(candidate: Box, target_cls: int, gt: List[LabeledBox]) -> bool:
    """纯度：候选弱框内不允许出现其它类别实例（以中心点落入作为判定）。"""
    for (b, c) in gt:
        if c != target_cls and center_in_rect(b, candidate):
            return False
    return True

# Check the purity of a candidate weak box
def purity_check_iou(candidate: Box, target_cls: int, gt: List[LabeledBox], eps: float = 1e-6) -> bool:
    """纯度：若候选弱框与其它类别任何实例框 IoU > eps，则视为混入。"""
    for (b, c) in gt:
        if c != target_cls and iou(candidate, b) > eps:
            return False
    return True

def build_weak_boxes(
    gt: List[LabeledBox],   # Ground-truth
    purity_mode: str = "center",    # Purity check mode: "center" or "iou"
    iou_eps: float = 1e-6,  # IoU threshold
) -> List[WeakBox]:
    # 1. 按类别分组，分别构建弱框
    classes_in_img = sorted({c for (_, c) in gt})
    weak_boxes: List[WeakBox] = []

