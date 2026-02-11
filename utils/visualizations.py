from typing import Dict, List, Tuple, Optional
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

# visualize for stage 2 debugging: overlay CCAM heatmap on the original image, and draw GT/pseudo/det boxes.
def _denorm_imagenet_to_u8_bgr(img_chw: np.ndarray) -> np.ndarray:
    """
    img_chw: float32, shape [3,H,W], usually ImageNet normalized RGB
    return: uint8 BGR image, shape [H,W,3]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    x = img_chw.astype(np.float32) * std + mean
    x = np.clip(x, 0.0, 1.0)
    img_rgb = (np.transpose(x, (1, 2, 0)) * 255.0).astype(np.uint8)  # HWC RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def _safe_xyxy_int(box_xyxy, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = int(round(float(x1))); y1 = int(round(float(y1)))
    x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _draw_boxes(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    color_bgr: Tuple[int, int, int],
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    prefix: str = "",
    thickness: int = 2,
    font_scale: float = 0.5,
):
    H, W = img_bgr.shape[:2]
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = _safe_xyxy_int(boxes_xyxy[i], W, H)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)

        # # build text: e.g., "GT:3", "PL:2@0.73", "DET:5@0.91"
        # txt_parts = []
        # if prefix:
        #     txt_parts.append(prefix)
        #
        # if labels is not None:
        #     txt_parts.append(str(int(labels[i])))
        #
        # if scores is not None:
        #     txt_parts.append(f"{float(scores[i]):.2f}")
        #
        # if len(txt_parts) > 0:
        #     text = ":".join([txt_parts[0], ",".join(txt_parts[1:])]) if len(txt_parts) > 1 else txt_parts[0]
        #     # text background
        #     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        #     y_text = max(0, y1 - 4)
        #     cv2.rectangle(img_bgr, (x1, max(0, y_text - th - 4)), (x1 + tw + 4, y_text), color_bgr, -1)
        #     cv2.putText(
        #         img_bgr, text, (x1 + 2, y_text - 2),
        #         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        #     )


def _overlay_ccam_roi(
    img_bgr: np.ndarray,
    ccam_hw: np.ndarray,             # (Hc,Wc) or (Hc,Wc,1), float
    wbox_xyxy: Tuple[float, float, float, float],
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    draw_wbox: bool = True,
    wbox_color_bgr: Tuple[int, int, int] = (0, 255, 255),  # yellow
    wbox_thickness: int = 2,
):
    """
    Put CCAM heatmap into the weak-box ROI area on the original image.
    """
    H, W = img_bgr.shape[:2]

    # sanitize CCAM
    if ccam_hw.ndim == 3:
        ccam_hw = ccam_hw[..., 0]
    ccam_hw = ccam_hw.astype(np.float32)
    ccam_hw = np.nan_to_num(ccam_hw, nan=0.0, posinf=0.0, neginf=0.0)

    # normalize to [0,1] for visualization (robust)
    mn, mx = float(ccam_hw.min()), float(ccam_hw.max())
    if mx > mn:
        ccam_n = (ccam_hw - mn) / (mx - mn)
    else:
        ccam_n = np.zeros_like(ccam_hw, dtype=np.float32)

    x1, y1, x2, y2 = _safe_xyxy_int(wbox_xyxy, W, H)
    roi_w = max(1, x2 - x1 + 1)
    roi_h = max(1, y2 - y1 + 1)

    # resize CCAM to ROI size
    heat = cv2.resize(ccam_n, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (heat * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, colormap)  # BGR

    roi = img_bgr[y1:y2 + 1, x1:x2 + 1]
    blended = cv2.addWeighted(roi, 1.0 - alpha, heat_color, alpha, 0.0)
    img_bgr[y1:y2 + 1, x1:x2 + 1] = blended

    if draw_wbox:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), wbox_color_bgr, wbox_thickness)


def visualize_stage2_debug_one_image(
    img_chw: torch.Tensor,                     # [3,H,W], normalized RGB
    wboxes_this_img: Optional[torch.Tensor],    # [Ri,4] xyxy in image coords (float)
    ccams_this_img: Optional[torch.Tensor],     # [Ri,1,Hc,Wc] or [Ri,Hc,Wc] (float)
    final_det: Optional[Dict[str, torch.Tensor]],
    pseudo: Optional[Dict[str, torch.Tensor]],
    gt: Optional[Dict[str, torch.Tensor]],
    save_path: str,
    alpha_ccam: float = 0.45,
    max_det: int = 50,
    max_pseudo: int = 50,
):
    """
    Draw on ONE original image:
      - CCAM (for each weak box ROI)
      - GT boxes (red)
      - Pseudo-label boxes (blue)
      - Final detections (green)
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- base image ---
    img_np = img_chw.detach().cpu().numpy()
    img_bgr = _denorm_imagenet_to_u8_bgr(img_np)

    H, W = img_bgr.shape[:2]

    # --- CCAM overlay for each wbox ---
    if (wboxes_this_img is not None) and (ccams_this_img is not None):
        wbs = wboxes_this_img.detach().cpu()
        ccs = ccams_this_img.detach().cpu()

        # unify shape: [Ri,Hc,Wc]
        if ccs.ndim == 4:  # [Ri,1,Hc,Wc]
            ccs = ccs[:, 0]
        for i in range(min(wbs.shape[0], ccs.shape[0])):
            x1, y1, x2, y2 = wbs[i].tolist()
            _overlay_ccam_roi(
                img_bgr,
                ccam_hw=ccs[i].numpy(),
                wbox_xyxy=(x1, y1, x2, y2),
                alpha=alpha_ccam,
                draw_wbox=True,
            )

    # --- GT (red) ---
    if gt is not None and "boxes" in gt:
        gt_boxes = gt["boxes"].detach().cpu().numpy()
        gt_labels = gt.get("labels")
        gt_labels = gt_labels.detach().cpu().numpy() if gt_labels is not None else None
        _draw_boxes(img_bgr, gt_boxes, color_bgr=(0, 0, 255), labels=gt_labels, prefix="GT", thickness=2)

    # --- Pseudo (blue) ---
    if pseudo is not None and "boxes" in pseudo:
        pb = pseudo["boxes"].detach().cpu().numpy()
        pl = pseudo.get("labels")
        ps = pseudo.get("scores")
        pl = pl.detach().cpu().numpy() if pl is not None else None
        ps = ps.detach().cpu().numpy() if ps is not None else None

        if ps is not None and pb.shape[0] > 0:
            order = np.argsort(-ps)
            order = order[:max_pseudo]
            pb = pb[order]
            if pl is not None: pl = pl[order]
            ps = ps[order]
        else:
            pb = pb[:max_pseudo]
            if pl is not None: pl = pl[:max_pseudo]
            if ps is not None: ps = ps[:max_pseudo]

        _draw_boxes(img_bgr, pb, color_bgr=(255, 0, 0), labels=pl, scores=ps, prefix="PL", thickness=2)

    # --- Final detections (green) ---
    if final_det is not None and "boxes" in final_det:
        db = final_det["boxes"].detach().cpu().numpy()
        dl = final_det.get("labels")
        ds = final_det.get("scores")
        dl = dl.detach().cpu().numpy() if dl is not None else None
        ds = ds.detach().cpu().numpy() if ds is not None else None

        if ds is not None and db.shape[0] > 0:
            order = np.argsort(-ds)
            order = order[:max_det]
            db = db[order]
            if dl is not None: dl = dl[order]
            ds = ds[order]
        else:
            db = db[:max_det]
            if dl is not None: dl = dl[:max_det]
            if ds is not None: ds = ds[:max_det]

        _draw_boxes(img_bgr, db, color_bgr=(0, 255, 0), labels=dl, scores=ds, prefix="DET", thickness=2)

    # --- legend ---
    legend = "GT=Red | PL=Blue | DET=Green | WBOX=Yellow | CCAM=Heat"
    cv2.rectangle(img_bgr, (5, 5), (5 + 8 * len(legend), 28), (0, 0, 0), -1)
    cv2.putText(img_bgr, legend, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img_bgr)


def visualize_stage2_debug_batch(
    imgs: torch.Tensor,                         # [B,3,H,W]
    wboxes: Optional[torch.Tensor],              # [R,5] (batch_idx,x1,y1,x2,y2)
    ccam: Optional[torch.Tensor],                # [R,1,Hc,Wc]
    final_detections: List[Dict[str, torch.Tensor]],
    pseudo_labels: List[Dict[str, torch.Tensor]],
    gt_targets: Optional[List[Dict[str, torch.Tensor]]],
    out_dir: str,
    step_tag: str = "",
):
    """
    Batch wrapper: save B images into out_dir.
    - For image b:
        wboxes_this_img: [Rb,4]
        ccams_this_img:  [Rb,1,Hc,Wc]
    """
    os.makedirs(out_dir, exist_ok=True)
    B = imgs.shape[0]

    for b in range(B):
        # gather wboxes/ccams for this image
        wb_b = None
        cc_b = None
        if (wboxes is not None) and (ccam is not None):
            mask = (wboxes[:, 0].long() == b)
            if mask.any():
                wb_b = wboxes[mask, 1:5]   # [Rb,4]
                cc_b = ccam[mask]          # [Rb,1,Hc,Wc]

        gt_b = gt_targets[b] if gt_targets is not None else None
        pseudo_b = pseudo_labels[b] if pseudo_labels is not None else None
        det_b = final_detections[b] if final_detections is not None else None

        name = f"stage2_debug_b{b}"
        if step_tag:
            name += f"_{step_tag}"
        save_path = os.path.join(out_dir, name + ".png")

        visualize_stage2_debug_one_image(
            img_chw=imgs[b],
            wboxes_this_img=wb_b,
            ccams_this_img=cc_b,
            final_det=det_b,
            pseudo=pseudo_b,
            gt=gt_b,
            save_path=save_path,
        )
