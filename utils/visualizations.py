from typing import Dict, List, Tuple
from PIL import Image
import cv2
import numpy as np
import torch
import os
import random

# Visualize the annotations on the image
def draw_boxes(image: Image, boxes: List, class_map : Dict) -> Image:
    '''在图像上绘制边界框'''
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for b in boxes:
        xmin, ymin, xmax, ymax = map(int, b["box"])
        label = class_map[b["class_id"]]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image

# Visualize matched box pairs
def _to_uint8_img(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: Tensor [C,H,W], typically float in [0,1] or [-mean/std]
    return: BGR uint8 image for cv2
    """
    img = img_chw.detach().cpu().float()

    # 尽量鲁棒：若像是标准化后的数据，就做 min-max 拉伸
    if img.numel() == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if img.min() < 0 or img.max() > 1.0:
        mn, mx = img.min(), img.max()
        if (mx - mn) > 1e-6:
            img = (img - mn) / (mx - mn)
        else:
            img = torch.zeros_like(img)

    img = (img * 255.0).clamp(0, 255).byte()  # [C,H,W]
    img = img.permute(1, 2, 0).contiguous().numpy()  # HWC, RGB
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    # cv2 uses BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _box_xyxy_from_5(box5: torch.Tensor) -> np.ndarray:
    # box5: [5] = [batch_idx, x1,y1,x2,y2]
    b = box5.detach().cpu().float().numpy()
    return b[1:5]


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [4] xyxy
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def visualize_hungarian_matches(
    imgs: torch.Tensor,                 # [B,C,H,W]
    sparse_proposals: list,             # List[Dict] each has "box"[5], "class_id", optional "score","sim"
    seed_proposals: list,               # List[Dict] each has "box"[5], "class_id"
    matched_pairs: list,                # List[Tuple[sparse_dict, seed_dict]]
    wboxes: torch.Tensor = None,        # [R,5] optional weak boxes
    gt_targets=None,                   # NEW: List[Dict] or Tensor[G,5]
    gt_labels: torch.Tensor = None,     # NEW: when gt_targets is Tensor[G,5], optionally provide labels [G]
    save_dir: str = "./results/visualizations/matched_boxes/",
    max_images: int = 16,
    pick_mode: str = "all",      # "all" or "random"
    draw_unmatched_sparse: bool = True,
    draw_unmatched_seed: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
):
    """
    输出每张图一张可视化结果到 save_dir。
    颜色约定（BGR, cv2）：
    - GT:    红色   (0,0,255)
    - wbox:  黄色   (0,255,255)
    - seed:  蓝色   (255,0,0)
    - sparse:绿色   (0,255,0)
    - matched中心连线：紫色 (255,0,255)
    """
    os.makedirs(save_dir, exist_ok=True)
    assert imgs.dim() == 4, f"imgs should be [B,C,H,W], got {imgs.shape}"
    B, _, H, W = imgs.shape

    # --------- 1) 按 batch 分桶：sparse/seed/matched/wboxes ---------
    sparse_by_b = [[] for _ in range(B)]
    for sp in sparse_proposals:
        b = int(sp["box"][0].item())
        if 0 <= b < B:
            sparse_by_b[b].append(sp)

    seed_by_b = [[] for _ in range(B)]
    for sd in seed_proposals:
        b = int(sd["box"][0].item())
        if 0 <= b < B:
            seed_by_b[b].append(sd)

    matched_by_b = [[] for _ in range(B)]
    for sp, sd in matched_pairs:
        b = int(sp["box"][0].item())
        if 0 <= b < B:
            matched_by_b[b].append((sp, sd))

    wboxes_by_b = [[] for _ in range(B)]
    if wboxes is not None:
        for i in range(wboxes.shape[0]):
            b = int(wboxes[i, 0].item())
            if 0 <= b < B:
                wboxes_by_b[b].append(wboxes[i])

    # --------- 2) 按 batch 分桶：GT（兼容两种格式）---------
    gt_by_b = [[] for _ in range(B)]  # list of (xyxy_np, cls_int_or_None)
    if gt_targets is not None:
        # Case A: List[Dict], len=B
        if isinstance(gt_targets, (list, tuple)):
            # 允许 user 传入 list len=B，每项 dict: {"boxes":[Ng,4], "labels":[Ng]}
            if len(gt_targets) == B and all(isinstance(x, dict) for x in gt_targets):
                for b in range(B):
                    boxes = gt_targets[b].get("boxes", None)
                    labels = gt_targets[b].get("labels", None)
                    if boxes is None:
                        continue
                    boxes = boxes.detach().cpu().float()
                    if labels is not None:
                        labels = labels.detach().cpu().long()
                    for i in range(boxes.shape[0]):
                        xyxy = boxes[i].numpy()
                        cls = int(labels[i].item()) if labels is not None and i < labels.numel() else None
                        gt_by_b[b].append((xyxy, cls))
            else:
                # 如果你传入的是别的 list 结构，这里直接提示
                raise ValueError(
                    "gt_targets as list/tuple must be List[Dict] with len=B, "
                    "each dict has keys: 'boxes' ([N,4]) and optional 'labels' ([N])."
                )
        # Case B: Tensor[G,5] = [b,x1,y1,x2,y2]
        elif torch.is_tensor(gt_targets):
            gt5 = gt_targets
            assert gt5.dim() == 2 and gt5.size(1) == 5, f"gt_targets tensor must be [G,5], got {gt5.shape}"
            gt5 = gt5.detach().cpu().float()
            if gt_labels is not None:
                gt_labels_cpu = gt_labels.detach().cpu().long()
                assert gt_labels_cpu.numel() == gt5.size(0), "gt_labels length must match gt_targets"
            else:
                gt_labels_cpu = None

            for i in range(gt5.size(0)):
                b = int(gt5[i, 0].item())
                if 0 <= b < B:
                    xyxy = gt5[i, 1:5].numpy()
                    cls = int(gt_labels_cpu[i].item()) if gt_labels_cpu is not None else None
                    gt_by_b[b].append((xyxy, cls))
        else:
            raise ValueError("Unsupported gt_targets type. Use List[Dict] or Tensor[G,5].")

    # --------- 3) 选择要输出的图 ---------
    idxs = list(range(B))
    if pick_mode == "random":
        random.shuffle(idxs)
    idxs = idxs[: min(max_images, B)]

    # 用于判断 unmatched（用box+cls做近似hash）
    def _id(sp_or_sd: dict) -> int:
        box = sp_or_sd["box"][1:5].detach().cpu().float().numpy()
        cls = int(sp_or_sd.get("class_id", -1))
        key = tuple(np.round(box, 1).tolist() + [cls])
        return hash(key)

    # --------- 4) 绘制每张图 ---------
    for b in idxs:
        canvas = _to_uint8_img(imgs[b])  # BGR

        # (a) GT：红色（最先画，避免盖住其它信息时你看不见GT）
        for xyxy, cls in gt_by_b[b]:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)
            tag = "gt" if cls is None else f"gt c{cls}"
            cv2.putText(
                canvas, tag,
                (int(x1), max(0, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA
            )

        # (b) wboxes：黄
        for wb in wboxes_by_b[b]:
            x1, y1, x2, y2 = wb[1:5].detach().cpu().float().numpy()
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), thickness)
            cv2.putText(
                canvas, "wbox",
                (int(x1), max(0, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA
            )

        # (c) seed：蓝
        for sd in seed_by_b[b]:
            x1, y1, x2, y2 = _box_xyxy_from_5(sd["box"])
            cls = int(sd.get("class_id", -1))
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness)
            cv2.putText(
                canvas, f"seed c{cls}",
                (int(x1), max(0, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA
            )

        # (d) sparse：绿
        for sp in sparse_by_b[b]:
            x1, y1, x2, y2 = _box_xyxy_from_5(sp["box"])
            cls = int(sp.get("class_id", -1))
            score = sp.get("score", None)
            sim = sp.get("sim", None)

            text = f"spr c{cls}"
            if score is not None:
                try:
                    text += f" s{float(score):.2f}"
                except Exception:
                    pass
            if sim is not None:
                try:
                    text += f" sim{float(sim):.2f}"
                except Exception:
                    pass

            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
            cv2.putText(
                canvas, text,
                (int(x1), min(H - 2, int(y2) + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1, cv2.LINE_AA
            )

        # (e) matched pairs：紫色连线 + IoU
        matched_s_ids = set()
        matched_t_ids = set()
        for sp, sd in matched_by_b[b]:
            matched_s_ids.add(_id(sp))
            matched_t_ids.add(_id(sd))

            s_xyxy = _box_xyxy_from_5(sp["box"])
            t_xyxy = _box_xyxy_from_5(sd["box"])
            iou = _iou_xyxy(s_xyxy, t_xyxy)

            sx = int((s_xyxy[0] + s_xyxy[2]) * 0.5)
            sy = int((s_xyxy[1] + s_xyxy[3]) * 0.5)
            tx = int((t_xyxy[0] + t_xyxy[2]) * 0.5)
            ty = int((t_xyxy[1] + t_xyxy[3]) * 0.5)

            cv2.line(canvas, (sx, sy), (tx, ty), (255, 0, 255), 2)

            cls = int(sd.get("class_id", -1))
            score = sp.get("score", None)
            sim = sp.get("sim", None)
            msg = f"m c{cls} iou{iou:.2f}"
            if score is not None:
                try:
                    msg += f" s{float(score):.2f}"
                except Exception:
                    pass
            if sim is not None:
                try:
                    msg += f" sim{float(sim):.2f}"
                except Exception:
                    pass

            cv2.putText(
                canvas, msg,
                (min(sx, tx), max(0, min(sy, ty) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), 1, cv2.LINE_AA
            )

        # (f) optional：unmatched 用淡色框画出来
        if draw_unmatched_sparse:
            for sp in sparse_by_b[b]:
                if _id(sp) in matched_s_ids:
                    continue
                x1, y1, x2, y2 = _box_xyxy_from_5(sp["box"])
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 1)

        if draw_unmatched_seed:
            for sd in seed_by_b[b]:
                if _id(sd) in matched_t_ids:
                    continue
                x1, y1, x2, y2 = _box_xyxy_from_5(sd["box"])
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (180, 0, 0), 1)

        out_path = os.path.join(save_dir, f"match_b{b}.jpg")
        cv2.imwrite(out_path, canvas)

# Visualize CCAM
def to_uint8_img(img: np.ndarray) -> np.ndarray:
    """
    img: HxW or HxWxC, can be float in [0,1] or [0,255], or uint8.
    return: uint8 in [0,255]
    """
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    # 如果像是 0~1，就放大
    if x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def denorm_imagenet(img_chw: np.ndarray,
                     mean=(0.485,0.456,0.406),
                     std=(0.229,0.224,0.225)) -> np.ndarray:
    # img_chw: (3,H,W) float
    x = img_chw.astype(np.float32).copy()
    for c in range(3):
        x[c] = x[c] * std[c] + mean[c]
    x = np.clip(x, 0, 1)
    return x  # (3,H,W) in [0,1]

# def overlay_ccam_on_image(
#     img_bgr: np.ndarray,          # 原图 (H,W,3) BGR 或 RGB 都行，但输出按 BGR 处理
#     ccam_hw1: np.ndarray,         # CCAM (Hc,Wc,1) float [0,1] 或 uint8
#     wbox_xyxy: Tuple[int,int,int,int],  # (x1,y1,x2,y2) 原图坐标
#     alpha: float = 0.45,
#     colormap: int = cv2.COLORMAP_JET,
#     normalize_ccam: bool = True,
# ) -> np.ndarray:
#     """
#     把 CCAM 映射到原图 ROI 区域并叠加可视化。
#     返回：overlay 后的 BGR 图
#     """
#     x1, y1, x2, y2 = map(int, wbox_xyxy)
#     H, W = img_bgr.shape[:2]
#     # clamp
#     x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
#     y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
#     if x2 <= x1 + 1 or y2 <= y1 + 1:
#         return img_bgr.copy()
#
#     roi_w = x2 - x1
#     roi_h = y2 - y1
#
#     # --- prepare ccam to uint8 ---
#     ccam = ccam_hw1
#     if ccam.ndim == 3 and ccam.shape[-1] == 1:
#         ccam = ccam[..., 0]  # (Hc,Wc)
#     ccam = ccam.astype(np.float32)
#
#     if normalize_ccam:
#         # 防止全常数导致除0
#         cmin, cmax = float(ccam.min()), float(ccam.max())
#         if cmax > cmin + 1e-6:
#             ccam = (ccam - cmin) / (cmax - cmin)
#         else:
#             ccam = np.zeros_like(ccam, dtype=np.float32)
#
#     ccam_u8 = np.clip(ccam * 255.0, 0, 255).astype(np.uint8)  # (Hc,Wc)
#
#     # --- resize to ROI size ---
#     ccam_roi_u8 = cv2.resize(ccam_u8, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)
#
#     # --- colorize ---
#     heat_color = cv2.applyColorMap(ccam_roi_u8, colormap)  # (roi_h,roi_w,3) BGR
#
#     # --- overlay ---
#     out = img_bgr.copy()
#     roi = out[y1:y2, x1:x2]
#     out[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, heat_color, alpha, 0)
#
#     # 可选：画出弱框边界
#     cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     return out
def overlay_ccam_on_image(
    img_bgr: np.ndarray,                     # 原图 (H,W,3) BGR
    ccam_hw1: np.ndarray,                    # CCAM (Hc,Wc,1) or (Hc,Wc) float/uint8
    wbox_xyxy: Tuple[int, int, int, int],    # (x1,y1,x2,y2) 原图坐标
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    normalize_ccam: bool = True,
    # ---- new ----
    gt_boxes=None,                           # (N,4) or list, xyxy in image coords
    gt_labels=None,                          # (N,) optional
    seed_boxes=None,                         # (M,4)/(M,5) or List[Dict] seed_proposals
    seed_labels=None,                        # (M,) optional (if seed_boxes is array/tensor)
    seed_scores=None,                        # (M,) optional
    draw_wbox: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    把 CCAM 映射到原图 ROI 区域并叠加可视化，同时可选绘制：
      - wbox(绿)
      - gt(红)
      - seed(蓝)
    返回：overlay 后的 BGR 图
    """
    def _to_np_boxes_xyxy(x):
        """Return np.ndarray (K,4) float32, or None."""
        if x is None:
            return None
        if isinstance(x, list):
            if len(x) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            # list of dict (seed_proposals)
            if isinstance(x[0], dict) and "box" in x[0]:
                boxes = []
                for d in x:
                    b = d["box"]
                    if torch.is_tensor(b):
                        b = b.detach().cpu().float().numpy()
                    b = np.array(b, dtype=np.float32).reshape(-1)
                    if b.size == 5:
                        boxes.append(b[1:5])
                    else:
                        boxes.append(b[:4])
                return np.stack(boxes, axis=0).astype(np.float32)
            # list of [x1,y1,x2,y2]
            arr = np.array(x, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] == 5:
                arr = arr[:, 1:5]
            return arr[:, :4].astype(np.float32)

        if torch.is_tensor(x):
            x = x.detach().cpu().float().numpy()

        arr = np.array(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == 5:
            arr = arr[:, 1:5]
        return arr[:, :4].astype(np.float32)

    def _clamp_xyxy(x1, y1, x2, y2, W, H):
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(0, min(int(x2), W))
        y2 = max(0, min(int(y2), H))
        return x1, y1, x2, y2

    # ---------------- base checks ----------------
    x1, y1, x2, y2 = map(int, wbox_xyxy)
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, W, H)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return img_bgr.copy()

    roi_w, roi_h = x2 - x1, y2 - y1

    # ---------------- prepare ccam ----------------
    ccam = ccam_hw1
    if ccam.ndim == 3 and ccam.shape[-1] == 1:
        ccam = ccam[..., 0]  # FIX: was ccam[. 0]
    ccam = ccam.astype(np.float32)

    if normalize_ccam:
        cmin, cmax = float(ccam.min()), float(ccam.max())
        if cmax > cmin + 1e-6:
            ccam = (ccam - cmin) / (cmax - cmin)
        else:
            ccam = np.zeros_like(ccam, dtype=np.float32)

    ccam_u8 = np.clip(ccam * 255.0, 0, 255).astype(np.uint8)              # (Hc,Wc)
    ccam_roi_u8 = cv2.resize(ccam_u8, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(ccam_roi_u8, colormap)                 # (roi_h,roi_w,3) BGR

    # ---------------- overlay ----------------
    out = img_bgr.copy()
    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, heat_color, alpha, 0)

    # ---------------- draw boxes ----------------
    if draw_wbox:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        # cv2.putText(out, "wbox", (x1, max(0, y1 - 4)),
        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1, cv2.LINE_AA)

    # gt (red)
    gt_arr = _to_np_boxes_xyxy(gt_boxes)
    if gt_arr is not None and gt_arr.shape[0] > 0:
        for i in range(gt_arr.shape[0]):
            gx1, gy1, gx2, gy2 = gt_arr[i]
            gx1, gy1, gx2, gy2 = _clamp_xyxy(gx1, gy1, gx2, gy2, W, H)
            cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 0, 255), thickness)
            tag = "gt"
            if gt_labels is not None:
                try:
                    tag = f"gt c{int(gt_labels[i])}"
                except Exception:
                    pass
            # cv2.putText(out, tag, (gx1, max(0, gy1 - 4)),
            #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)

    # seed (blue)
    # 1) if seed_boxes is List[Dict] (seed_proposals), we can also extract class_id/score per box
    if isinstance(seed_boxes, list) and len(seed_boxes) > 0 and isinstance(seed_boxes[0], dict):
        for d in seed_boxes:
            b = d.get("box", None)
            if b is None:
                continue
            if torch.is_tensor(b):
                b = b.detach().cpu().float().numpy()
            b = np.array(b, dtype=np.float32).reshape(-1)
            if b.size == 5:
                sx1, sy1, sx2, sy2 = b[1:5]
            else:
                sx1, sy1, sx2, sy2 = b[:4]
            sx1, sy1, sx2, sy2 = _clamp_xyxy(sx1, sy1, sx2, sy2, W, H)
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (255, 0, 0), thickness)

            msg = "seed"
            cls = d.get("class_id", None)
            if cls is not None:
                msg += f" c{int(cls)}"
            sc = d.get("score", None)
            if sc is not None:
                try:
                    msg += f" s{float(sc):.2f}"
                except Exception:
                    pass
            # cv2.putText(out, msg, (sx1, min(H - 2, sy2 + 14)),
            #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        seed_arr = _to_np_boxes_xyxy(seed_boxes)
        if seed_arr is not None and seed_arr.shape[0] > 0:
            for i in range(seed_arr.shape[0]):
                sx1, sy1, sx2, sy2 = seed_arr[i]
                sx1, sy1, sx2, sy2 = _clamp_xyxy(sx1, sy1, sx2, sy2, W, H)
                cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (255, 0, 0), thickness)

                msg = "seed"
                if seed_labels is not None:
                    try:
                        msg += f" c{int(seed_labels[i])}"
                    except Exception:
                        pass
                if seed_scores is not None:
                    try:
                        msg += f" s{float(seed_scores[i]):.2f}"
                    except Exception:
                        pass
                # cv2.putText(out, msg, (sx1, min(H - 2, sy2 + 14)),
                #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

    return out
