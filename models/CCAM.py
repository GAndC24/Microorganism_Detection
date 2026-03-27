# CCAM for object location
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any
from torchvision.models import vgg16
from dataclasses import dataclass
from utils import *
import cv2


@dataclass
class CCAMConfig:
    device: torch.device  # device
    freeze_backbone: bool  # freeze backbone weights
    in_c: int  # input channels
    threshold: float  # threshold


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


# CCAM
class CCAM(nn.Module):
    def __init__(
        self,
        config : CCAMConfig,       # Stage2 CCAM configuration
        backbone : nn.Module,  # Default VGG-16 with ImageNet pretrained weights
    )-> None:
        super(CCAM, self).__init__()

        self.encoder = backbone
        if config.freeze_backbone:     # freeze backbone weights
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.config = config

        # CCAM Generator
        self.ccam_generator = CCAMGenerator(in_c=self.config.in_c)


    def _check_scoremap_validity(self, scoremap):
        if not isinstance(scoremap, np.ndarray):
            raise TypeError("Scoremap must be a numpy array; it is {}."
                            .format(type(scoremap)))
        if scoremap.dtype != np.float32:
            raise TypeError("Scoremap must be of np.float type; it is of {} type."
                            .format(scoremap.dtype))
        if len(scoremap.shape) != 2:
            raise ValueError("Scoremap must be a 2D array; it is {}D."
                             .format(len(scoremap.shape)))
        if np.isnan(scoremap).any():
            raise ValueError("Scoremap must not contain nans.")
        if (scoremap > 1).any() or (scoremap < 0).any():
            raise ValueError("Scoremap must be in range [0, 1]."
                             "scoremap.min()={}, scoremap.max()={}."
                             .format(scoremap.min(), scoremap.max()))


    def _compute_bboxes_from_scoremaps(
        self,
        scoremap,
        scoremap_threshold_list,
        factor,
        multi_contour_eval=False
    ):
        """
        Copy from: https://github.com/clovaai/wsolevaluation
        Args:
            scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
            scoremap_threshold_list: iterable
            multi_contour_eval: flag for multi-contour evaluation

        Returns:
            estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
                at each cam threshold
            number_of_box_list: list of the number of boxes at each cam threshold
        """

        self._check_scoremap_validity(scoremap)
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

        def scoremap2bbox(threshold):
            _, thr_gray_heatmap = cv2.threshold(
                src=scoremap_image,
                thresh=int(threshold * np.max(scoremap_image)),
                maxval=255,
                type=cv2.THRESH_BINARY)
            contours = cv2.findContours(
                image=thr_gray_heatmap,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

            if len(contours) == 0:
                return np.asarray([[0, 0, 0, 0]]), 1

            if not multi_contour_eval:
                contours = [max(contours, key=cv2.contourArea)]

            estimated_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x0, y0, x1, y1 = x, y, x + w, y + h
                x1 = min(x1, width - 1)
                y1 = min(y1, height - 1)

                estimated_boxes.append([x0 * factor, y0 * factor, x1 * factor, y1 * factor])

            return np.asarray(estimated_boxes), len(contours)

        estimated_boxes_at_each_thr = []
        number_of_box_list = []
        for threshold in scoremap_threshold_list:
            boxes, number_of_box = scoremap2bbox(threshold)
            estimated_boxes_at_each_thr.append(boxes)
            number_of_box_list.append(number_of_box)

        return estimated_boxes_at_each_thr, number_of_box_list


    def _get_box_scores(
        self,
        ccam_i: np.ndarray,
        boxes: Any,
        topk_ratio: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute a score for each predicted box using Top-k mean over the CCAM
        activations inside the box region.
        """
        if isinstance(boxes, list):
            if len(boxes) == 0:
                return torch.empty(0, dtype=torch.float32, device=self.config.device)
            boxes = torch.cat(boxes, dim=0)

        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=boxes.device)

        if not isinstance(ccam_i, np.ndarray):
            ccam_i = np.asarray(ccam_i, dtype=np.float32)

        height, width = ccam_i.shape
        ccam_tensor = torch.from_numpy(ccam_i).to(device=boxes.device, dtype=torch.float32)

        scores = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            x1 = max(0, min(int(np.floor(x1)), width - 1))
            y1 = max(0, min(int(np.floor(y1)), height - 1))
            x2 = max(0, min(int(np.ceil(x2)), width - 1))
            y2 = max(0, min(int(np.ceil(y2)), height - 1))

            if x2 < x1 or y2 < y1:
                scores.append(torch.tensor(0.0, device=boxes.device))
                continue

            region = ccam_tensor[y1:y2 + 1, x1:x2 + 1].reshape(-1)
            if region.numel() == 0:
                scores.append(torch.tensor(0.0, device=boxes.device))
                continue

            k = max(1, int(np.ceil(region.numel() * topk_ratio)))
            topk_values = torch.topk(region, k=k, largest=True).values
            scores.append(topk_values.mean())

        return torch.stack(scores)


    def forward(
        self,
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        should_invert : bool = True     # whether to invert CCAM
    )-> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        :return:
        - loss_ccam: CCAM loss
        - ccam_boxes: CCAM boxes list, len=B, each dict
                    {
                        "boxes": Tensor, [N_i, 4], each row is (x1, y1, x2, y2)
                        "scores": Tensor, [N_i], confidence scores for each box
                    }
        """
        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]


        # -----get CCAM and loss_ccam-----
        ccam, loss_ccam = self.ccam_generator(high_feature_maps)  # [B, 1, H_h, W_h]
        if should_invert:
            ccam = 1 - ccam

        # ------get boxes from CCAM-----
        B = ccam.shape[0]
        ccam_boxes: List[Dict[str, torch.Tensor]] = [{} for _ in range(B)]
        pred_boxes = [[] for _ in range(B)]
        for i in range(B):
            ccam_i = ccam[i, 0, :, :].detach().cpu().numpy().astype(np.float32)

            estimated_boxes_at_each_thr, _ = self._compute_bboxes_from_scoremaps(
                scoremap=ccam_i,
                scoremap_threshold_list=[self.config.threshold],
                factor=1.0,
                multi_contour_eval=False
            )

            pred_boxes[i].append(torch.from_numpy(estimated_boxes_at_each_thr[0]).float().to(self.config.device))  # [N_i, 4]

            scores = self._get_box_scores(ccam_i, pred_boxes[i])

            ccam_boxes[i].update({
                "boxes": pred_boxes[i],  # Tensor, [N_i, 4], each row is (x1, y1, x2, y2)
                "scores": scores,  # Tensor
            })


        return loss_ccam, ccam_boxes


def build_CCAM_model(
    config: CCAMConfig
)-> CCAM:
    backbone = vgg16(pretrained=True).features

    model = CCAM(
        config=config,
        backbone=backbone,
    )

    return model
