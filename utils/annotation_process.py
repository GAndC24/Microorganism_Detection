# Construct Weak Bounding Boxes
from typing import List, Tuple, Iterable
import heapq

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

# Build weak bounding boxes from ground-truth labeled boxes
def build_weak_boxes(
    gt: List[LabeledBox],   # Ground-truth
    purity_mode: str = "iou",    # Purity check mode: "center" or "iou"
    iou_eps: float = 1e-6,  # IoU threshold
) -> List[WeakBox]:
    # 1. 按类别分组，分别构建弱框
    classes_in_img = sorted({c for (_, c) in gt})
    weak_boxes: List[WeakBox] = []

    for cls in classes_in_img:
        # 2. 初始化：每个同类实例框作为一个 cluster
        init_boxes = [b for (b, c) in gt if c == cls]
        if not init_boxes:
            continue

        clusters: List[Cluster] = [Cluster([b], cls) for b in init_boxes]

        n = len(clusters)
        if n == 1:
            wb = union_rect(clusters[0].boxes)
            if purity_mode == 'iou':
                is_pure = purity_check_iou(wb, cls)
            else:
                is_pure = purity_check_center(wb, cls)
            if not is_pure:    # 极端情况下（其他类中心点恰好在该框内），跳过
                pass
            weak_boxes.append((wb, cls))
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
                cost = area(r_pq) - area(rp) - area(rk)
                heapq.heappush(pq, (cost, pp, qq, version[pp], version[qq]))

        # output alive clusters as weak boxes
        for i in range(n):
            if not alive[i]:
                continue
            wb = union_rect(clusters[i].boxes)
            # 最终 check purity
            if purity_mode == 'iou':
                is_pure = purity_check_iou(wb, cls, gt, eps=iou_eps)
            else:
                is_pure = purity_check_center(wb, cls, gt)
            if not is_pure:
                raise RuntimeError(f"Purity violated in final weak box for class={cls}")
            weak_boxes.append((wb, cls))

    return weak_boxes

