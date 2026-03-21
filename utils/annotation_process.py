# Construct Weak Bounding Boxes
from typing import List, Tuple, Iterable, TypedDict, Dict, Optional
import heapq
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import os


Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)

class LabeledBox(TypedDict):
    """gt and weak box"""
    box: Box
    class_id: int

# WeakBox 与 LabeledBox 结构一致
WeakBox = LabeledBox

@dataclass
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

# Clip a box to image boundary
def clip_box(b: Box, img_w: float, img_h: float) -> Box:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    # ensure valid ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

# Expand a box around its center (used to weaken singleton instances)
def expand_box(b: Box, expand_ratio: float = 1.0, min_expand: float = 2.0, image_size: Tuple[float, float] = None) -> Box:
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dw = max(w * expand_ratio, min_expand)
    dh = max(h * expand_ratio, min_expand)
    nx1 = cx - (w / 2.0 + dw)
    ny1 = cy - (h / 2.0 + dh)
    nx2 = cx + (w / 2.0 + dw)
    ny2 = cy + (h / 2.0 + dh)
    out = (nx1, ny1, nx2, ny2)
    if image_size is not None:
        img_w, img_h = image_size
        out = clip_box(out, img_w, img_h)
    return out

# Post-process singleton weak boxes so that WB is not identical to GT while still satisfying purity
def weaken_singleton_box(
    b: Box,
    target_cls: int,
    gt: List[LabeledBox],
    purity_mode: str = 'iou',
    iou_eps: float = 1e-6,
    expand_ratio: float = 1.0,
    min_expand: float = 2.0,
    max_iter: int = 10,
    image_size: Tuple[float, float] = None,
) -> Box:
    # Try expanded candidates; if purity fails, gradually reduce expansion
    r = expand_ratio
    m = min_expand
    for _ in range(max_iter):
        cand = expand_box(b, expand_ratio=r, min_expand=m, image_size=image_size)
        if purity_mode == 'iou':
            ok = purity_check_iou(cand, target_cls, gt, eps=iou_eps)
        else:
            ok = purity_check_center(cand, target_cls, gt)
        if ok:
            return cand
        r *= 0.5
        m *= 0.5
    # If no expansion can satisfy purity, return a minimally expanded box (numerical epsilon)
    eps = 1e-3
    x1, y1, x2, y2 = b
    out = (x1 - eps, y1 - eps, x2 + eps, y2 + eps)
    if image_size is not None:
        img_w, img_h = image_size
        out = clip_box(out, img_w, img_h)
    return out

# Check the purity of a candidate weak box
def purity_check_center(candidate: Box, target_cls: int, gt: List[LabeledBox]) -> bool:
    """纯度：候选弱框内不允许出现其它类别实例（以中心点落入作为判定）。"""
    for lb in gt:
        b, c = lb["box"], lb["class_id"]
        if c != target_cls and center_in_rect(b, candidate):
            return False
    return True

# Check the purity of a candidate weak box
def purity_check_iou(candidate: Box, target_cls: int, gt: List[LabeledBox], eps: float = 1e-6) -> bool:
    """纯度：若候选弱框与其它类别任何实例框 IoU > eps，则视为混入。"""
    for lb in gt:
        b, c = lb["box"], lb["class_id"]
        if c != target_cls and iou(candidate, b) > eps:
            return False
    return True

# Build weak bounding boxes from ground-truth labeled boxes
def build_weak_boxes(
    gt: List[LabeledBox],   # Ground-truth
    purity_mode: str = "iou",    # Purity check mode: "center" or "iou"
    iou_eps: float = 1e-6,  # IoU threshold
    weaken_singletons: bool = True,  # whether to weaken singleton instances
    expand_ratio: float = 1.0,        # expansion ratio for singleton weak boxes
    expand_min_px: float = 2.0,       # minimum expansion (in pixels) for singleton weak boxes
    image_size: Tuple[float, float] = None,  # (img_w, img_h)
) -> List[WeakBox]:
    # 1. 按类别分组，分别构建弱框
    classes_in_img = sorted({lb["class_id"] for lb in gt})
    weak_boxes: List[WeakBox] = []

    for cls in classes_in_img:
        # 2. 初始化：每个同类实例框作为一个 cluster
        init_boxes = [lb["box"] for lb in gt if lb["class_id"] == cls]
        if not init_boxes:
            continue

        clusters: List[Cluster] = [Cluster([b], cls) for b in init_boxes]

        n = len(clusters)
        if n == 1:
            wb = union_rect(clusters[0].boxes)
            if weaken_singletons:
                wb = weaken_singleton_box(wb, cls, gt, purity_mode=purity_mode, iou_eps=iou_eps,
                                         expand_ratio=expand_ratio, min_expand=expand_min_px, image_size=image_size)
            if image_size is not None:
                img_w, img_h = image_size
                wb = clip_box(wb, img_w, img_h)
            if purity_mode == 'iou':
                is_pure = purity_check_iou(wb, cls, gt, eps=iou_eps)
            else:
                is_pure = purity_check_center(wb, cls, gt)
            if not is_pure:    # 极端情况下（其他类中心点恰好在该框内），跳过
                continue
            weak_boxes.append({"box": wb, "class_id": cls})
            continue

        # 3. min priority PQ: (cost, p, q, ver_p, ver_q)
        #    cost = Area(union(p,q)) - Area(p) - Area(q)
        pq: List[Tuple[float, int, int, int, int]] = []

        alive = [True] * n  # for lazy deletion
        version = [0] * n  # cluster 内容更新时 +1，用于过滤 dead pair

        rect_cache: List[Box] = [union_rect((clusters[i].boxes)) for i in range(n)]     # 预计算当前 cluster rect

        # 初始化所有 pair
        for p in range(n):
            for q in range(p + 1, n):
                rp, rq = rect_cache[p], rect_cache[q]
                r_pq = union_rect([rp, rq])
                if image_size is not None:
                    img_w, img_h = image_size
                    r_pq = clip_box(r_pq, img_w, img_h)
                cost = area(r_pq) - area(rp) - area(rq)
                heapq.heappush(pq, (cost, p, q, version[p], version[q]))

        # 4. 贪心合并 + lazy deletion
        while pq:
            cost, p, q, vp, vq = heapq.heappop(pq)

            # 过期/失效 pair：lazy deletion
            if not alive[p] or not alive[q]:
                continue
            if vp != version[p] or vq != version[q]:
                continue

            # 尝试合并
            merged_boxes = clusters[p].boxes + clusters[q].boxes
            cand = union_rect(merged_boxes)

            if image_size is not None:
                img_w, img_h = image_size
                cand = clip_box(cand, img_w, img_h)

            # purity check
            if purity_mode == 'iou':
                is_pure = purity_check_iou(cand, cls, gt, eps=iou_eps)
            else:
                is_pure = purity_check_center(cand, cls, gt)
            if not is_pure:
                continue

            # 接受合并：q 合并进 p，q 失效
            clusters[p].boxes = merged_boxes
            alive[q] = False

            # 更新 p 的 version 与 rect_cache
            version[p] += 1
            rect_cache[p] = union_rect(clusters[p].boxes)

            # 把新的 (p, k) pair 入队（旧 pair 不删除，靠 lazy deletion 过滤）
            for k in range(n):
                if k == p or not alive[k]:
                    continue
                pp, qq = (p, k) if p < k else (k, p)
                rp, rk = rect_cache[pp], rect_cache[qq]
                r_pq = union_rect([rp, rk])
                if image_size is not None:
                    img_w, img_h = image_size
                    r_pq = clip_box(r_pq, img_w, img_h)
                cost = area(r_pq) - area(rp) - area(rk)
                heapq.heappush(pq, (cost, pp, qq, version[pp], version[qq]))

        # output alive clusters as weak boxes
        for i in range(n):
            if not alive[i]:
                continue
            wb = union_rect(clusters[i].boxes)
            if weaken_singletons and len(clusters[i].boxes) == 1:
                wb = weaken_singleton_box(wb, cls, gt, purity_mode=purity_mode, iou_eps=iou_eps,
                                         expand_ratio=expand_ratio, min_expand=expand_min_px, image_size=image_size)
            if image_size is not None:
                img_w, img_h = image_size
                wb = clip_box(wb, img_w, img_h)
            # 最终 check purity
            if purity_mode == 'iou':
                is_pure = purity_check_iou(wb, cls, gt, eps=iou_eps)
            else:
                is_pure = purity_check_center(wb, cls, gt)
            if not is_pure:
                raise RuntimeError(f"Purity violated in final weak box for class={cls}")
                # return []
            weak_boxes.append({"box": wb, "class_id": cls})

    return weak_boxes

# read the VOC XML annotation file
def parse_voc_xml(xml_path: str, class_map : Dict) -> Tuple[Dict, List[LabeledBox]]:
    '''读取 VOC XML 标注文件并返回 image_info 和 boxes'''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取 image_info
    size = root.find("size")
    image_info = {
        "width": int(size.find("width").text),
        "height": int(size.find("height").text),
        "depth": int(size.find("depth").text),
    }

    # 读取 boxes
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")

        class_id = class_map[label]

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        gt : LabeledBox = {
            "box" : (xmin, ymin, xmax, ymax),
            "class_id" : class_id
        }
        boxes.append(gt)

    return image_info, boxes

# read the VOC XML annotation file
def parse_voc_xml_sam3(xml_path: str, class_map : Dict) -> Tuple[Dict, List[Dict]]:
    '''读取 VOC XML 标注文件并返回 image_info 和 boxes'''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取 image_info
    size = root.find("size")
    image_info = {
        "width": int(size.find("width").text),
        "height": int(size.find("height").text),
        "depth": int(size.find("depth").text),
    }

    # 读取 boxes
    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        score = float(obj.find("score").text)

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        gt = {
            "box" : (xmin, ymin, xmax, ymax),
            "score": score
        }
        boxes.append(gt)

    return image_info, boxes

# Indent XML for pretty printing
def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """让 ElementTree 输出更美观（带缩进换行）。"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

# Save the weak boxes to VOC XML annotation file
def save_weak_boxes_as_voc_xml(
    xml_save_path: str,     # output xml path
    image_info: Dict,       # at least include {"width": int, "height": int, "depth": int}
    weak_boxes: List[dict],                 # List[WeakBox]
    id2class: Dict[int, str],               # {class_id : class_name}
    filename: Optional[str] = None,     # xml <filename>
    folder: Optional[str] = None,       # xml <folder>
    path: Optional[str] = None,     # xml <path>
    database: str = None,      # xml <database>
    segmented: int = 0,     # xml <segmented>
    clip: bool = True,      # whether to clip boxes to image boundary
) -> None:
    w = int(image_info["width"])
    h = int(image_info["height"])
    d = int(image_info.get("depth", 3))

    if filename is None:
        filename = os.path.splitext(os.path.basename(xml_save_path))[0] + ".jpg"
    if folder is None:
        folder = ""
    if path is None:
        path = filename

    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = str(folder)
    ET.SubElement(root, "filename").text = str(filename)
    ET.SubElement(root, "path").text = str(path)

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = str(database)

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(d)

    ET.SubElement(root, "segmented").text = str(int(segmented))

    for wb in weak_boxes:
        cls_id = int(wb["class_id"])
        cls_name = id2class.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = wb["box"]

        # 裁剪到图像边界
        if clip:
            x1 = max(0.0, min(float(x1), float(w)))
            y1 = max(0.0, min(float(y1), float(h)))
            x2 = max(0.0, min(float(x2), float(w)))
            y2 = max(0.0, min(float(y2), float(h)))

        # 保证坐标顺序正确
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        xmin = int(round(x1))
        ymin = int(round(y1))
        xmax = int(round(x2))
        ymax = int(round(y2))

        xmin = max(0, min(xmin, w))
        ymin = max(0, min(ymin, h))
        xmax = max(0, min(xmax, w))
        ymax = max(0, min(ymax, h))

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(cls_name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    _indent_xml(root)
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(xml_save_path), exist_ok=True)
    tree.write(xml_save_path, encoding="utf-8", xml_declaration=True)