# Morphological Prototype R-CNN
import os.path
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Union, Optional, Literal
from dataclasses import dataclass
# from utils import random_masking, add_gaussian_noise, MorphologicalPrototypeGenerator, FeatureHook, build_vgg16_backbone_with_hook, vgg_layer_out_c_maps, CCAMGenerator, HungarianMatcher, visualize_hungarian_matches
from utils import *
from torchvision.ops import RoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models import vgg16
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
import numpy as np
import cv2
from torchvision.ops import nms
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import box_iou


@dataclass
class Stage1Config:
    num_classes: int  # number of classes
    embed_dim: int  # embedding dimension
    img_size: Tuple[int, int]  # input image size
    batch_size: int  # batch size
    hidden_dim: int  # MLP hidden dimension
    layer_indices: List[int]  # feature layer indices, [low, mid, high]
    mask_threshold : float  # random masking threshold
    gaussian_sigma : float  # gaussian noise standard deviation
    # for mp_generator
    in_c: int  # input channels
    patch_size: int  # patch size
    components_range: List  # list of number of components for GMM
    random_state: int  # random state for GMM(seed)
    max_iter: int  # max iteration for EM
    roi_out_size_mid: Tuple[int, int]  # output size for middle feature maps
    roi_out_size_high: Tuple[int, int]  # output size for high feature maps
    spatial_scale_mid: float  # spatial scale for middle feature maps
    spatial_scale_high : float # spatial scale for high feature maps
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag

@dataclass
class Stage2Config:
    device: torch.device  # device
    img_size: Tuple[int, int]  # input image size
    num_classes: int  # number of classes
    in_c : int  # high-level feature maps channels
    freeze_backbone: bool  # freeze backbone weights
    # for compute constrain loss
    w_proto_loss : float  # weight for loss_proto
    w_pull_loss : float # weight for loss_pull
    w_push_loss : float # weight for loss_push
    # for compute foreground scores
    w_prototype_sim : float  # weight for prototype similarity
    w_ccam_score : float  # weight for CCAM score
    w_obj_score : float  # weight for object score
    # for select sparse proposal
    keep_iou_thr: float  # keep IoU threshold
    # for projection head
    hidden_dim: int     # MLP hidden dimension
    embed_dim: int      # embedding dimension
    # for RPN
    rpn_anchor_sizes: Tuple[int]  # anchor sizes
    rpn_anchor_aspect_ratios: Tuple[float]  # anchor aspect ratios
    rpn_fg_iou_thresh : float  # foreground IoU threshold
    rpn_bg_iou_thresh : float  # background IoU threshold
    rpn_batch_size_per_image : int # RPN batch size per image
    rpn_pre_nms_top_n: Dict[str, int]  # pre NMS top N, {"training": int, "testing": int}
    rpn_post_nms_top_n: Dict[str, int]  # post NMS top N, {"training": int, "testing": int}
    rpn_nms_thresh: float  # RPN NMS threshold
    # for CCAM
    ccam_threshold : float # CCAM threshold
    # for HungarianMatcher
    cost_class: float  # classification cost weight
    cost_bbox: float  # bbox L1 cost weight
    cost_giou: float  # bbox GIoU cost weight
    # for Pseudo Labels
    rpn_pseudo_nms_thr : float  # NMS threshold for pseudo labels
    rpn_pseudo_topk : int  # top-k proposals for pseudo labels
    # for Match Loss
    match_focal_alpha : float   # focal loss alpha
    match_focal_gamma : float   # focal loss gamma
    lambda_match_cls: float     # weight for match classification loss
    lambda_match_l1: float    # weight for match L1 loss
    lambda_match_giou: float    # weight for match giou loss
    # # for RoI Head
    # det_fg_iou_thresh: float  # foreground IoU threshold
    # det_bg_iou_thresh: float  # background IoU threshold
    # det_batch_size_per_image: int  # detection batch size per image
    # det_positive_fraction: float  # detection positive fraction
    # det_score_thresh: float  # detection score threshold
    # det_nms_thresh: float   # detection NMS threshold
    # detections_per_img: int # number of detections per image
    # for RoI Align
    roi_out_size_h2wb: Tuple[int, int]  # output size for high feature maps to weak box features
    spatial_scale_h2wb: float  # spatial scale for high feature maps to weak box features
    # roi_out_size_wb2p: Tuple[int]  # output size for weak box features to proposal box features
    # spatial_scale_wb2p: float  # spatial scale
    roi_out_size_h2p: Tuple[int, int]  # output size for high feature maps to proposal box features
    spatial_scale_h2p: float    # spatial scale for high feature maps to proposal box features
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True    # aligned flag

@dataclass
class LinearProbConfig:
    in_c : int # feature map channels
    in_size: int  # feature map size
    out_dim : int = 1  # output dimension

@dataclass
class PrototypeCheckerConfig:
    in_c: int  # input channels
    embed_dim: int  # embedding dimension
    patch_size : int # patch size
    roi_out_size: Tuple[int, int]  # output size for high feature maps
    spatial_scale : float  # spatial scale for high feature maps
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

# MP R-CNN Stage 1: Multi-Hierarchy Feature Alignment and Construct Prototypes
class Stage1(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16
        hook: FeatureHook,         # Feature Hook
        mp_generator: MorphologicalPrototypeGenerator,  # Morphological Prototype Generator
        config : Stage1Config       # Stage1 configuration
    )-> None:
        super(Stage1, self).__init__()

        self.encoder = backbone
        self.hook = hook
        self.mp_generator = mp_generator
        self.num_classes = config.num_classes
        self.num_prototypes = config.num_classes
        self.img_size = config.img_size
        self.hidden_dim = config.hidden_dim
        self.embed_dim = config.embed_dim
        self.mask_threshold = config.mask_threshold
        self.gaussian_sigma = config.gaussian_sigma

        self.gap_l = nn.AdaptiveAvgPool2d((1, 1))
        low_out_c = vgg_layer_out_c_maps[config.layer_indices[0]]
        self.bn_l = nn.BatchNorm1d(low_out_c)

        self.roi_align = RoIAlign(
            output_size=config.roi_out_size_high,
            spatial_scale=config.spatial_scale_high,
            sampling_ratio=config.sampling_ratio,
            aligned=config.aligned,
        )
        C3 = vgg_layer_out_c_maps[config.layer_indices[2]]
        H3, W3 = config.roi_out_size_high
        hidden_dim_in = int(C3 * H3 * W3)
        self.proj_h = nn.Sequential(
            nn.Flatten(),  # [N, C3, H3, W3] -> [N, C3*H3*W3]
            nn.Linear(hidden_dim_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.embed_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m)->None:
        """
        Initialize weights for Linear and BatchNorm layers.
        :param m: Module to initialize
        """
        if isinstance(m, nn.Linear):  # Check if the module is a Linear layer
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
            if m.bias is not None:  # Initialize bias to zero if it exists
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):  # Check if the module is a BatchNorm layer
            nn.init.constant_(m.weight, 1.0)  # Initialize scale (gamma) to 1
            nn.init.constant_(m.bias, 0)  # Initialize shift (beta) to 0

    def forward(
        self,
        x: torch.Tensor,
        wboxes: torch.Tensor,  # weak boxes, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor  # class label for weak boxes, [R, num_classes]
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        :return:
        - low-level latent features, [B, C1]
        - high-level feature maps, [B, C2, H2, W2]
        - high-level embedding features, [B, D]
        - CAM loss
        - prototypes, {class_id, prototype tensor}
        - patch logits, [R * k, D]
        - patch features for SupCon, [R, view = 1, D]
        """
        # -----get multi-level feature maps-----
        self.hook.clear()
        _ = self.encoder(x)
        feature_maps = self.hook.outputs
        low_feature_maps = feature_maps['low']      # [B, C1, H1, W1]
        mid_feature_maps = feature_maps['mid']      # [B, C2, H2, W2]
        high_feature_maps = feature_maps['high']        # [B, C3, H3, W3]

        # -----get low-level latent features-----
        low_latent_features = self.gap_l(low_feature_maps)  # [B, C1, 1, 1]
        B, C1, _, _ = low_latent_features.shape
        low_latent_features = low_latent_features.view(B, C1)
        low_latent_features = self.bn_l(low_latent_features)  # [B, C1]

        # -----get high-level embedding features-----
        # RoI Align to get weak box features
        roi_features = self.roi_align(high_feature_maps, wboxes)  # [R = num_wboxes, C3, H3, W3]
        # Random Masking
        masked_roi_features = random_masking(roi_features, tau_drop=self.mask_threshold)  # [R, C3, H3, W3]
        # Add Gaussian Noise
        noise_roi_features = add_gaussian_noise(masked_roi_features, sigma=self.gaussian_sigma) # [R, C3, H3, W3]
        roi_views = torch.stack(
            [masked_roi_features, noise_roi_features],
            dim=1
        )  # [R, V=2, C3, H3, W3], view0=masked, view1=noise
        R, V, C3, H3, W3 = roi_views.shape
        roi_views_flat = roi_views.view(R * V, C3, H3, W3)  # [R*V, C3, H3, W3]
        high_aug_embedding_features = self.proj_h(roi_views_flat)  # [R*V, D]
        high_aug_embedding_features = high_aug_embedding_features.view(R, V, self.embed_dim)  # [R, V, D]

        # -----construct prototypes-----
        loss_cam, prototypes, patch_logits, contrast_patch_features = self.mp_generator(mid_feature_maps, wboxes, wb_labels)        # {class_id : prototype tensor}

        return low_latent_features, high_feature_maps, high_aug_embedding_features, loss_cam, prototypes, patch_logits, contrast_patch_features

# MP R-CNN Stage 2: Unsupervised Proposal Generation and Object Detection
class Stage2(nn.Module):
    def __init__(
        self,
        config : Stage2Config,       # Stage2 configuration
        backbone : nn.Module,  # Default VGG-16 with aligned weights
        dataset_mps: Dict[int, torch.Tensor]  # {class_id : prototype tensor}
    )-> None:
        super(Stage2, self).__init__()

        self.encoder = backbone
        if config.freeze_backbone:     # freeze backbone weights
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.config = config
        self.dataset_mps = dataset_mps

        # RoI Align for high-level feature maps to weak box feature maps
        self.roi_align_h2wb = RoIAlign(
            output_size=self.config.roi_out_size_h2wb,
            spatial_scale=self.config.spatial_scale_h2wb,
            sampling_ratio=self.config.sampling_ratio,
            aligned=self.config.aligned,
        )
        # RoI Align for high-level feature maps to proposal box feature maps
        self.roi_align_h2p = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=self.config.roi_out_size_h2p,
            sampling_ratio=self.config.sampling_ratio,
        )

        # RPN
        anchor_generator = AnchorGenerator(
            sizes=(self.config.rpn_anchor_sizes,),
            aspect_ratios=(self.config.rpn_anchor_aspect_ratios,)
        )
        rpn_head = RPNHead(in_channels=self.config.in_c, num_anchors=anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
                    anchor_generator=anchor_generator,
                    head=rpn_head,
                    fg_iou_thresh=self.config.rpn_fg_iou_thresh,
                    bg_iou_thresh=self.config.rpn_bg_iou_thresh,
                    batch_size_per_image=self.config.rpn_batch_size_per_image,
                    positive_fraction=0.5,
                    pre_nms_top_n={"training": self.config.rpn_pre_nms_top_n['training'], "testing": self.config.rpn_pre_nms_top_n['testing']},
                    post_nms_top_n={"training": self.config.rpn_post_nms_top_n['training'], "testing": self.config.rpn_post_nms_top_n['testing']},
                    nms_thresh=self.config.rpn_nms_thresh
        )
        self.rpn_pseudo_label_generator = build_RPN_pseudo_label_generator(device=config.device)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Projection Head
        hidden_dim_in = self.config.in_c
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(config.embed_dim)
        )

        # Object Classifier
        self.obj_classifier = nn.Linear(config.embed_dim, self.config.num_classes + 1)  # num_classes + 1(for background class)

        # CCAM Generator
        self.ccam_generator = CCAMGenerator(in_c=self.config.in_c)

        # # One-to-One Matcher
        # self.matcher = HungarianMatcher(
        #     cost_class=self.config.cost_class,
        #     cost_bbox=self.config.cost_bbox,
        #     cost_giou=self.config.cost_giou
        # )

        # # Box Head
        # resolution = self.config.roi_out_size_h2p[0]
        # box_head = TwoMLPHead(
        #     in_channels=self.config.in_c * resolution * resolution,
        #     representation_size=self.config.hidden_dim,
        # )
        #
        # # Box Predictor锛坈ls + reg锛?
        # box_predictor = FastRCNNPredictor(
        #     in_channels=self.config.hidden_dim,
        #     num_classes=self.config.num_classes + 1,  # include background
        # )
        #
        # # RoI Heads (roi align + box head + box predictor)
        # self.roi_heads = RoIHeads(
        #     box_roi_pool=self.roi_align_h2p,
        #     box_head=box_head,
        #     box_predictor=box_predictor,
        #
        #     fg_iou_thresh=self.config.det_fg_iou_thresh,
        #     bg_iou_thresh=self.config.det_bg_iou_thresh,
        #     batch_size_per_image=self.config.det_batch_size_per_image,
        #     positive_fraction=self.config.det_positive_fraction,
        #
        #     bbox_reg_weights=None,
        #     score_thresh=self.config.det_score_thresh,
        #     nms_thresh=self.config.det_nms_thresh,
        #     detections_per_img=self.config.detections_per_img,
        # )

    def _get_dense_proposals(
        self,
        high_feature_maps : torch.Tensor,   # high-level feature maps, [B, C_h, H_h, W_h]
        image_sizes: List[Tuple[int, int]],  # [(H_img, W_img), ...]
        images_tensor: Optional[torch.Tensor] = None,  # images.tensors
        rpn_targets: List[Dict[str, torch.Tensor]] = None,  # RPN targets for training
    )-> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        ''':return dense_proposals : List[Tensor], len=R, each Tensor is [num_proposals, 4]'''
        self.rpn.train()

        # prepare img_list
        img_list = ImageList(tensors=images_tensor, image_sizes=image_sizes)

        features = {"0": high_feature_maps}  # RPN features format, Dict[str, tensor]

        dense_proposals_list, rpn_losses_dict = self.rpn(img_list, features, rpn_targets)

        return dense_proposals_list, rpn_losses_dict

    @torch.no_grad()
    def _build_rpn_targets(
        self,
        wboxes: torch.Tensor,
        wb_labels: torch.Tensor,
        sam3_targets: List[Dict[str, torch.Tensor]],
        score_thresh: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Filter SAM3 targets for RPN supervision.

        Returned format is List[Dict] with keys:
        - "box": Tensor [N, 4]
        - "score": Tensor [N]
        - "class_id": Tensor [N], range [1, C]
        """
        if sam3_targets is None:
            return []

        device = wboxes.device
        num_classes = int(self.config.num_classes)
        rpn_targets: List[Dict[str, torch.Tensor]] = []

        for b, sam_target in enumerate(sam3_targets):
            sam_boxes = sam_target.get("boxes", None)
            sam_scores = sam_target.get("scores", None)

            if sam_boxes is None or sam_boxes.numel() == 0:
                rpn_targets.append({
                    "box": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "score": torch.zeros((0,), device=device, dtype=torch.float32),
                    "class_id": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            sam_boxes = sam_boxes.to(device=device, dtype=torch.float32)
            if sam_scores is None:
                sam_scores = torch.ones((sam_boxes.shape[0],), device=device, dtype=torch.float32)
            else:
                sam_scores = sam_scores.to(device=device, dtype=torch.float32)

            wb_mask = (wboxes[:, 0].long() == b)
            wb_boxes_b = wboxes[wb_mask, 1:5].to(device=device, dtype=torch.float32)
            wb_labels_b = wb_labels[wb_mask].to(device=device, dtype=torch.float32)

            if wb_boxes_b.numel() == 0:
                rpn_targets.append({
                    "box": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "score": torch.zeros((0,), device=device, dtype=torch.float32),
                    "class_id": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            inside = (
                (sam_boxes[:, None, 0] >= wb_boxes_b[None, :, 0]) &
                (sam_boxes[:, None, 1] >= wb_boxes_b[None, :, 1]) &
                (sam_boxes[:, None, 2] <= wb_boxes_b[None, :, 2]) &
                (sam_boxes[:, None, 3] <= wb_boxes_b[None, :, 3])
            )
            score_keep = sam_scores > score_thresh
            wb_areas = (
                (wb_boxes_b[:, 2] - wb_boxes_b[:, 0]).clamp(min=0.0) *
                (wb_boxes_b[:, 3] - wb_boxes_b[:, 1]).clamp(min=0.0)
            )

            keep_boxes: List[torch.Tensor] = []
            keep_scores: List[torch.Tensor] = []
            keep_class_ids: List[int] = []

            for sam_idx in range(sam_boxes.shape[0]):
                if not bool(score_keep[sam_idx]):
                    continue

                valid_wb = torch.nonzero(inside[sam_idx], as_tuple=False).squeeze(1)
                if valid_wb.numel() == 0:
                    continue

                chosen_wb = valid_wb[torch.argmin(wb_areas[valid_wb])]
                class_id = int(torch.argmax(wb_labels_b[chosen_wb]).item()) + 1
                class_id = max(1, min(class_id, num_classes))

                keep_boxes.append(sam_boxes[sam_idx])
                keep_scores.append(sam_scores[sam_idx])
                keep_class_ids.append(class_id)

            if len(keep_boxes) == 0:
                rpn_targets.append({
                    "box": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "score": torch.zeros((0,), device=device, dtype=torch.float32),
                    "class_id": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            rpn_targets.append({
                "box": torch.stack(keep_boxes, dim=0),
                "score": torch.stack(keep_scores, dim=0),
                "class_id": torch.tensor(keep_class_ids, device=device, dtype=torch.long),
            })

        return rpn_targets

    # @torch.no_grad()
    # def _get_dense_proposals(
    #     self,
    #     high_feature_maps : torch.Tensor,   # high-level feature maps, [B, C_h, H_h, W_h]
    #     image_sizes: List[Tuple[int, int]],  # [(H_img, W_img), ...] 鐪熷疄灏哄
    #     images_tensor: Optional[torch.Tensor] = None,  # images.tensors
    #     rpn_module: Optional[nn.Module] = None
    # )-> List[torch.Tensor]:
    #     ''':return dense_proposals : List[Tensor], len=R, each Tensor is [num_proposals, 4]'''
    #     # self.rpn.eval()
    #     if rpn_module is None:
    #         rpn_module = self.rpn
    #
    #     was_training = rpn_module.training
    #     rpn_module.eval()
    #
    #     # prepare img_list
    #     img_list = ImageList(tensors=images_tensor, image_sizes=image_sizes)
    #
    #     features = {"0": high_feature_maps}  # RPN features format, Dict[str, tensor]
    #     # dense_proposals_list, _ = self.rpn(img_list, features)
    #     dense_proposals_list, _ = rpn_module(img_list, features)
    #
    #     if was_training:
    #         rpn_module.train()
    #
    #     return dense_proposals_list

    # @torch.no_grad()
    # def _get_multi_bboxes(
    #     self,
    #     box_xyxy: torch.Tensor,  # [4] or [N,4], image coords (xyxy)
    #     img_hw: Tuple[int, int],  # (Himg, Wimg)
    #     do_aug: bool = True,
    #     delta_aug: float = 0.10,  # [0.05, 0.15]
    #     num_aug: int = 4,  # generate how many jitter boxes per base box
    #     min_box_size: int = 2,  # avoid degenerate tiny boxes
    #     keep_at_least_one: bool = True,  # NEW: never return empty if input non-empty
    # ) -> torch.Tensor:
    #     """
    #     Seed Proposal Augmentation (box jittering) with fallback:
    #       - If a box is smaller than min_box_size in width/height, expand it to min_box_size
    #         around its center (then clamp to image bounds).
    #       - Optionally guarantee at least one output box (keep_at_least_one=True).
    #
    #     Return:
    #       out_boxes: Tensor [M, 4] in image coords (xyxy), float32
    #                 M is roughly N*(1+num_aug), after duplicate removal.
    #     """
    #     device = box_xyxy.device
    #     dtype = torch.float32
    #     Himg, Wimg = int(img_hw[0]), int(img_hw[1])
    #
    #     if box_xyxy is None or box_xyxy.numel() == 0:
    #         return torch.zeros((0, 4), device=device, dtype=dtype)
    #
    #     # ---- make [N,4] float boxes
    #     if box_xyxy.dim() == 1:
    #         boxes = box_xyxy.view(1, 4).to(device=device, dtype=dtype)
    #     else:
    #         boxes = box_xyxy.to(device=device, dtype=dtype)
    #
    #     # ---- sanitize ordering (x1<=x2, y1<=y2)
    #     x1 = torch.min(boxes[:, 0], boxes[:, 2])
    #     y1 = torch.min(boxes[:, 1], boxes[:, 3])
    #     x2 = torch.max(boxes[:, 0], boxes[:, 2])
    #     y2 = torch.max(boxes[:, 1], boxes[:, 3])
    #     boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    #
    #     # ---- clamp to image bounds (float)
    #     # use [0, W-1]/[0, H-1] to match your original convention
    #     boxes[:, 0] = boxes[:, 0].clamp(0, Wimg - 1)
    #     boxes[:, 2] = boxes[:, 2].clamp(0, Wimg - 1)
    #     boxes[:, 1] = boxes[:, 1].clamp(0, Himg - 1)
    #     boxes[:, 3] = boxes[:, 3].clamp(0, Himg - 1)
    #
    #     # ---- drop NaN/Inf boxes early (robustness)
    #     finite = torch.isfinite(boxes).all(dim=1)
    #     boxes = boxes[finite]
    #     if boxes.numel() == 0:
    #         return torch.zeros((0, 4), device=device, dtype=dtype)
    #
    #     # ---- remove truly invalid (non-positive size) boxes
    #     raw_w = boxes[:, 2] - boxes[:, 0]
    #     raw_h = boxes[:, 3] - boxes[:, 1]
    #     pos = (raw_w > 0.0) & (raw_h > 0.0)
    #     boxes = boxes[pos]
    #     if boxes.numel() == 0:
    #         return torch.zeros((0, 4), device=device, dtype=dtype)
    #
    #     # -------------------------------------------------------------------------
    #     # Fallback expansion: ensure EACH original box is at least min_box_size
    #     # (center-preserving), then clamp and re-sanitize.
    #     # -------------------------------------------------------------------------
    #     minsz = float(min_box_size)
    #
    #     cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    #     cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    #     w = (boxes[:, 2] - boxes[:, 0])
    #     h = (boxes[:, 3] - boxes[:, 1])
    #
    #     w2 = torch.clamp(w, min=minsz)
    #     h2 = torch.clamp(h, min=minsz)
    #
    #     bx1 = cx - 0.5 * w2
    #     by1 = cy - 0.5 * h2
    #     bx2 = cx + 0.5 * w2
    #     by2 = cy + 0.5 * h2
    #
    #     # clamp to bounds
    #     bx1 = bx1.clamp(0, Wimg - 1)
    #     bx2 = bx2.clamp(0, Wimg - 1)
    #     by1 = by1.clamp(0, Himg - 1)
    #     by2 = by2.clamp(0, Himg - 1)
    #
    #     # re-sanitize after clamp
    #     ax1 = torch.min(bx1, bx2)
    #     ax2 = torch.max(bx1, bx2)
    #     ay1 = torch.min(by1, by2)
    #     ay2 = torch.max(by1, by2)
    #     boxes = torch.stack([ax1, ay1, ax2, ay2], dim=-1)
    #
    #     # If clamping near borders caused width/height to collapse again,
    #     # do a second-pass 鈥減ush inside bounds鈥?to enforce min size as much as possible.
    #     # (This matters when cx is extremely close to the border.)
    #     w3 = boxes[:, 2] - boxes[:, 0]
    #     h3 = boxes[:, 3] - boxes[:, 1]
    #
    #     need_w = w3 < minsz
    #     need_h = h3 < minsz
    #
    #     if need_w.any():
    #         # try to set [x1, x2] = [cx - min/2, cx + min/2] then shift into [0, W-1]
    #         cxn = (boxes[:, 0] + boxes[:, 2]) * 0.5
    #         nx1 = cxn - 0.5 * minsz
    #         nx2 = cxn + 0.5 * minsz
    #         # shift if out of bounds
    #         shift_l = (0.0 - nx1).clamp(min=0.0)
    #         shift_r = (nx2 - (Wimg - 1)).clamp(min=0.0)
    #         nx1 = nx1 + shift_l - shift_r
    #         nx2 = nx2 + shift_l - shift_r
    #         # clamp final
    #         nx1 = nx1.clamp(0, Wimg - 1)
    #         nx2 = nx2.clamp(0, Wimg - 1)
    #         boxes[need_w, 0] = nx1[need_w]
    #         boxes[need_w, 2] = nx2[need_w]
    #
    #     if need_h.any():
    #         cyn = (boxes[:, 1] + boxes[:, 3]) * 0.5
    #         ny1 = cyn - 0.5 * minsz
    #         ny2 = cyn + 0.5 * minsz
    #         shift_t = (0.0 - ny1).clamp(min=0.0)
    #         shift_b = (ny2 - (Himg - 1)).clamp(min=0.0)
    #         ny1 = ny1 + shift_t - shift_b
    #         ny2 = ny2 + shift_t - shift_b
    #         ny1 = ny1.clamp(0, Himg - 1)
    #         ny2 = ny2.clamp(0, Himg - 1)
    #         boxes[need_h, 1] = ny1[need_h]
    #         boxes[need_h, 3] = ny2[need_h]
    #
    #     # final validity check (after fallback expansion)
    #     fw = boxes[:, 2] - boxes[:, 0]
    #     fh = boxes[:, 3] - boxes[:, 1]
    #     valid_final = (fw > 0.0) & (fh > 0.0)
    #     boxes = boxes[valid_final]
    #
    #     if boxes.numel() == 0:
    #         # extremely rare (only possible if Himg/Wimg are tiny or coords are pathological)
    #         if keep_at_least_one:
    #             return torch.zeros((1, 4), device=device, dtype=dtype)
    #         return torch.zeros((0, 4), device=device, dtype=dtype)
    #
    #     # always keep originals (after expansion)
    #     out = [boxes]
    #
    #     if (not do_aug) or (num_aug <= 0):
    #         return boxes
    #
    #     # ---- xyxy -> cxcywh for jitter
    #     cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    #     cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    #     w = (boxes[:, 2] - boxes[:, 0]).clamp(min=minsz)
    #     h = (boxes[:, 3] - boxes[:, 1]).clamp(min=minsz)
    #
    #     # ---- sample eps for each aug and each box
    #     for _ in range(int(num_aug)):
    #         eps = (2.0 * torch.rand((boxes.size(0), 4), device=device, dtype=dtype) - 1.0) * float(delta_aug)
    #         epsx, epsy, epsw, epsh = eps[:, 0], eps[:, 1], eps[:, 2], eps[:, 3]
    #
    #         # random sign per component
    #         sgn = torch.where(
    #             torch.rand((boxes.size(0), 4), device=device) > 0.5,
    #             torch.ones((boxes.size(0), 4), device=device, dtype=dtype),
    #             -torch.ones((boxes.size(0), 4), device=device, dtype=dtype),
    #         )
    #         sx, sy, sw, sh = sgn[:, 0], sgn[:, 1], sgn[:, 2], sgn[:, 3]
    #
    #         cx2 = cx * (1.0 + sx * epsx)
    #         cy2 = cy * (1.0 + sy * epsy)
    #         w2 = w * (1.0 + sw * epsw)
    #         h2 = h * (1.0 + sh * epsh)
    #
    #         # keep positive size (and at least min size)
    #         w2 = torch.clamp(w2, min=minsz)
    #         h2 = torch.clamp(h2, min=minsz)
    #
    #         nx1 = cx2 - 0.5 * w2
    #         ny1 = cy2 - 0.5 * h2
    #         nx2 = cx2 + 0.5 * w2
    #         ny2 = cy2 + 0.5 * h2
    #
    #         # clamp to image bounds
    #         nx1 = nx1.clamp(0, Wimg - 1)
    #         nx2 = nx2.clamp(0, Wimg - 1)
    #         ny1 = ny1.clamp(0, Himg - 1)
    #         ny2 = ny2.clamp(0, Himg - 1)
    #
    #         # sanitize ordering after clamp
    #         ax1 = torch.min(nx1, nx2)
    #         ax2 = torch.max(nx1, nx2)
    #         ay1 = torch.min(ny1, ny2)
    #         ay2 = torch.max(ny1, ny2)
    #
    #         aug = torch.stack([ax1, ay1, ax2, ay2], dim=-1)
    #
    #         # filter tiny (strict) for augmented boxes
    #         aw = aug[:, 2] - aug[:, 0]
    #         ah = aug[:, 3] - aug[:, 1]
    #         aug = aug[(aw >= minsz) & (ah >= minsz)]
    #
    #         if aug.numel() > 0:
    #             out.append(aug)
    #
    #     out_boxes = torch.cat(out, dim=0)  # [M,4]
    #
    #     # ---- remove duplicates (quantize to ints for hashing)
    #     # keep at least one if input non-empty
    #     q = torch.round(out_boxes).to(torch.int64)
    #     uniq = torch.unique(q, dim=0, sorted=False)
    #
    #     if uniq.numel() == 0 and keep_at_least_one:
    #         # fallback: keep the first (expanded) original box
    #         uniq = torch.round(boxes[:1]).to(torch.int64)
    #
    #     return uniq.to(dtype=dtype)

    def _map_boxes_roi_to_image_xyxy(
        self,
        boxes_local: Union[List[List[float]], torch.Tensor],    # list of [lx1, ly1, lx2, ly2] in ccam pixel coords, or Tensor [N, 4]
        wbox: torch.Tensor,     # Tensor [5] = [batch_idx, x1, y1, x2, y2] in image coords
        cam_hw: Tuple[int, int],    # (Hc, Wc) spatial size of ccam
        img_hw: Tuple[int, int],    # (Himg, Wimg) spatial size of original image
    ) -> torch.Tensor:
        """
        Map boxes from ROI-local coordinate system (ccam space) to image coordinate system.
        :returns: boxes_global: Tensor [N, 4] in image coords (xyxy), float32
        """
        device = wbox.device
        dtype = torch.float32

        if isinstance(boxes_local, list):
            if len(boxes_local) == 0:
                return torch.zeros((0, 4), device=device, dtype=dtype)
            boxes_local = torch.tensor(boxes_local, device=device, dtype=dtype)
        else:
            boxes_local = boxes_local.to(device=device, dtype=dtype)

        # Unpack
        _, x1, y1, x2, y2 = wbox.to(dtype=dtype)
        Hc, Wc = cam_hw
        Himg, Wimg = img_hw

        # Avoid division by zero
        Wc = max(int(Wc), 1)
        Hc = max(int(Hc), 1)

        roi_w = (x2 - x1).clamp(min=1e-6)
        roi_h = (y2 - y1).clamp(min=1e-6)

        # Scale from cam pixels to image pixels within ROI
        sx = roi_w / float(Wc)
        sy = roi_h / float(Hc)

        # Map
        gx1 = x1 + boxes_local[:, 0] * sx
        gy1 = y1 + boxes_local[:, 1] * sy
        gx2 = x1 + boxes_local[:, 2] * sx
        gy2 = y1 + boxes_local[:, 3] * sy

        boxes_global = torch.stack([gx1, gy1, gx2, gy2], dim=-1)

        # Clamp to image bounds
        boxes_global[:, 0].clamp_(0, Wimg - 1)
        boxes_global[:, 2].clamp_(0, Wimg - 1)
        boxes_global[:, 1].clamp_(0, Himg - 1)
        boxes_global[:, 3].clamp_(0, Himg - 1)

        # Ensure x1<=x2, y1<=y2 (robustness)
        x_min = torch.min(boxes_global[:, 0], boxes_global[:, 2])
        x_max = torch.max(boxes_global[:, 0], boxes_global[:, 2])
        y_min = torch.min(boxes_global[:, 1], boxes_global[:, 3])
        y_max = torch.max(boxes_global[:, 1], boxes_global[:, 3])
        boxes_global = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        return boxes_global


    def _map_boxes_image_to_roi_xyxy(
        self,
        boxes_img_xyxy: torch.Tensor,  # [N,4] in image coords
        wbox_xyxy: torch.Tensor,  # [4]   in image coords
        cam_hw: tuple,  # (Hc, Wc)
    ) -> torch.Tensor:
        """
        Map image-space boxes (xyxy) to ROI-local CCAM pixel coords (xyxy in [0..Wc-1]/[0..Hc-1]).
        :return: cam_boxes: [N,4] in RoI coords
        """
        device = boxes_img_xyxy.device
        boxes = boxes_img_xyxy.to(device=device, dtype=torch.float32)

        x1, y1, x2, y2 = wbox_xyxy.to(device=device, dtype=torch.float32)
        Hc, Wc = int(cam_hw[0]), int(cam_hw[1])
        roi_w = (x2 - x1).clamp(min=1e-6)
        roi_h = (y2 - y1).clamp(min=1e-6)

        # normalize to ROI [0,1]
        nx1 = (boxes[:, 0] - x1) / roi_w
        ny1 = (boxes[:, 1] - y1) / roi_h
        nx2 = (boxes[:, 2] - x1) / roi_w
        ny2 = (boxes[:, 3] - y1) / roi_h

        # to cam pixels
        cx1 = (nx1 * (Wc - 1)).round()
        cy1 = (ny1 * (Hc - 1)).round()
        cx2 = (nx2 * (Wc - 1)).round()
        cy2 = (ny2 * (Hc - 1)).round()

        cam_boxes = torch.stack([cx1, cy1, cx2, cy2], dim=-1)

        # clamp + sanitize
        cam_boxes[:, 0].clamp_(0, Wc - 1)
        cam_boxes[:, 2].clamp_(0, Wc - 1)
        cam_boxes[:, 1].clamp_(0, Hc - 1)
        cam_boxes[:, 3].clamp_(0, Hc - 1)

        x_min = torch.min(cam_boxes[:, 0], cam_boxes[:, 2])
        x_max = torch.max(cam_boxes[:, 0], cam_boxes[:, 2])
        y_min = torch.min(cam_boxes[:, 1], cam_boxes[:, 3])
        y_max = torch.max(cam_boxes[:, 1], cam_boxes[:, 3])

        cam_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1).to(dtype=torch.long)
        return cam_boxes


    def _ccam_fg_bg_score_for_boxes(
        self,
        ccam_i: torch.Tensor,  # [Hc, Wc] float
        cam_boxes_xyxy: torch.Tensor,  # [N, 4] long in CCAM coords
        top_p: float = 0.2,  # Top-p% mean
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each box, compute:
          fg_score = mean(top p% CCAM values inside box)
          bg_score = mean(1-CCAM inside box)  (consistent with your original definition)
        Return fg, bg: [N] float

        Notes:
        - top_p in (0,1]; if top_p==1, it degenerates to plain mean.
        - For very small patches, always take at least 1 pixel.
        """
        device = self.config.device
        ccam_i = ccam_i.to(device=device, dtype=torch.float32)

        # clamp for safety in case CCAM is not strictly within [0,1]
        ccam_i = ccam_i.clamp(0.0, 1.0)

        N = cam_boxes_xyxy.shape[0]
        if N == 0:
            z = torch.zeros((0,), device=device, dtype=torch.float32)
            return z, z

        # sanitize top_p
        top_p = float(top_p)
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1], got {top_p}")

        fg = torch.zeros((N,), device=device, dtype=torch.float32)
        bg = torch.zeros((N,), device=device, dtype=torch.float32)

        for n in range(N):
            x1, y1, x2, y2 = cam_boxes_xyxy[n].tolist()

            # ensure valid slicing bounds (in case of any rounding / clipping issues)
            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)
            x2 = min(int(x2), ccam_i.shape[1] - 1)
            y2 = min(int(y2), ccam_i.shape[0] - 1)

            if x2 < x1 or y2 < y1:
                fg[n] = 0.0
                bg[n] = 1.0
                continue

            patch = ccam_i[y1:y2 + 1, x1:x2 + 1]  # [h,w]
            numel = patch.numel()
            if numel == 0:
                fg[n] = 0.0
                bg[n] = 1.0
                continue

            flat = patch.reshape(-1)

            # take top-k where k = ceil(p * numel), but at least 1
            k = int((top_p * numel) + 0.999999)  # ceil without importing math
            k = max(1, min(k, numel))

            # topk is stable and fast for moderate sizes
            top_vals = torch.topk(flat, k=k, largest=True, sorted=False).values
            m_top = top_vals.mean()

            fg[n] = m_top
            bg[n] = 1.0 - m_top  # keep your original bg definition

        # numeric safety
        fg = fg.clamp(0.0, 1.0)
        bg = bg.clamp(0.0, 1.0)

        return fg, bg


    def _xyxy_to_cxcywh(self, xyxy: torch.Tensor) -> torch.Tensor:
        # xyxy: [N,4]
        x1, y1, x2, y2 = xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        return torch.stack([cx, cy, w, h], dim=-1)


    def _compute_fg_scores(
        self,
        sparse_props_ccam_scores : torch.Tensor,  # [total_num_sparse_proposals], range [0, 1]
        prototypes_embeddings : torch.Tensor,   # Normalized, shape [num_classes, D]
        bg_prototype_embedding : torch.Tensor,  # Normalized, shape [D]
        sparse_proposal_embeddings : torch.Tensor,  # Normalized, shape [total_num_sparse_proposals, D]
        sparse_proposal_labels : torch.Tensor,  # [total_num_sparse_proposals], range [1, num_classes] (0 reserved for background)
        sparse_proposal_boxes : torch.Tensor,  # [total_num_sparse_proposals, 5], each row = [batch_idx, x1, y1, x2, y2]
        sparse_proposal_object_scores : torch.Tensor,  # [total_num_sparse_proposals], range [0, 1]
        w_prototype_sim : float,
        w_ccam_score: float,
        w_obj_score: float,
        tau : float = 0.1
    )-> torch.Tensor:
        """
        :return: fg_scores: [total_num_sparse_proposals], range [0,1]
        """
        z = sparse_proposal_embeddings
        bg = bg_prototype_embedding
        s_1 = sparse_props_ccam_scores
        s_2 = sparse_proposal_object_scores
        labels = sparse_proposal_labels
        C = prototypes_embeddings.shape[0]
        N = z.shape[0]

        # gather class prototype for each proposal
        # shift to [0, C-1]
        cls_idx = (labels - 1).clamp(min=0, max=C - 1)  # [total_num_sparse_proposals], range [0, C-1]
        proto_k = prototypes_embeddings[cls_idx]  # [total_num_sparse_proposals, D]

        # compute
        # sim_k = (z * proto_k).sum(dim=1) / tau  # [total_num_sparse_proposals]
        sim_k = (z * proto_k).sum(dim=1)  # [total_num_sparse_proposals]
        # logits = float(w_prototype_sim) * sim_k + float(w_ccam_score) * s_1 + float(w_obj_score) * s_2  # [total_num_sparse_proposals]
        margin_k = (z * proto_k - z * bg).sum(dim=1)  # [total_num_sparse_proposals]
        logits = float(w_prototype_sim) * margin_k + float(w_ccam_score) * s_1 + float(w_obj_score) * s_2  # [total_num_sparse_proposals]

        # # for debug
        # with open("debug_data.txt", "w") as f:
        #     for i in range(logits.size(0)):
        #         # f.write(f"Proposal {i}: sim_k={sim_k[i].item()}, ccam_score={s_1[i].item()}, obj_score={s_2[i].item()}, logit={logits[i].item()}\n")
        #         f.write(f"Proposal {i}: sim_k={sim_k[i].item()}, margin_k={margin_k[i].item()}, ccam_score={s_1[i].item()}, obj_score={s_2[i].item()}, logit={logits[i].item()}\n")

        fg_scores_0 = torch.sigmoid(logits)  # [total_num_sparse_proposals]

        # # grouped softmax by class
        # fg_scores = torch.zeros((N,), device=self.config.device, dtype=torch.float32)
        # valid_mask = (labels >= 1) & (labels <= C)
        # if valid_mask.any():
        #     # process each class separately
        #     for c in range(1, C + 1):
        #         mask_c = valid_mask & (labels == c)
        #         if not mask_c.any():
        #             continue
        #         logits_c = logits[mask_c]  # [n_c]
        #         # temperature-scaled softmax within the class group
        #         logits_c = logits_c / max(float(tau), 1e-6)
        #         logits_c = logits_c - logits_c.max()
        #         fg_scores[mask_c] = torch.softmax(logits_c, dim=0)

        # ---- 3) grouped softmax by (image, class) ----
        fg_scores = torch.zeros((N,), device=logits.device, dtype=torch.float32)

        valid_mask = (labels >= 1) & (labels <= C)
        if not valid_mask.any():
            return fg_scores

        # group key: (batch_idx, class_label)
        # batch_idx is sparse_proposal_boxes[:,0]
        batch_idx = sparse_proposal_boxes[:, 0].long()  # [N]
        group_id = batch_idx * (C + 1) + labels.long()  # [N], unique per (img, cls) since labels in [1..C]

        # Iterate each group; each group does a softmax over its proposals
        unique_gids = torch.unique(group_id[valid_mask])
        for gid in unique_gids:
            m = valid_mask & (group_id == gid)
            if not m.any():
                continue
            l = logits[m] / tau
            l = l - l.max()  # numerical stability
            fg_scores[m] = torch.softmax(l, dim=0)

        fg_scores_final = fg_scores_0 + fg_scores  # combine original sigmoid score with grouped softmax

        return fg_scores_final


    def _build_rpn_pseudo_labels(
        self,
        proposals: List[Dict],
        batch_size: int,
        image_sizes: List[Tuple[int, int]],  # [(Himg, Wimg), ...]
        min_box_size: float = 4.0,
        nms_iou_thr: Optional[float] = 0.3,
        keep_empty: bool = False,
        contain_thr: float = 0.80,
        contain_score_ratio_thr: float = 0.90,
        center_dist_rel_thr: float = 0.30,
        center_score_ratio_thr: float = 0.90,
        do_global_dedup: bool = False,
        global_nms_iou_thr: float = 0.15,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build RPN pseudo labels with:
          1) sanitization
          2) per-class NMS
          3) containment filter
          4) center-distance suppression
          5) optional global dedup

        Return list length B, each:
          {
            "boxes":  FloatTensor [Nb,4] in image coords (xyxy),
            "scores": FloatTensor [Nb],
            "labels": LongTensor  [Nb] in [1..C]  (0 reserved for background)
          }
        """

        device = self.config.device
        pseudo: List[Dict[str, torch.Tensor]] = []

        # =========================================================
        # helper functions
        # =========================================================
        def _box_area(boxes: torch.Tensor) -> torch.Tensor:
            w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
            h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
            return w * h

        def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
            """
            boxes1: [N,4], boxes2: [M,4]
            return: [N,M]
            """
            area1 = _box_area(boxes1)  # [N]
            area2 = _box_area(boxes2)  # [M]

            lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
            rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]

            wh = (rb - lt).clamp(min=0.0)  # [N,M,2]
            inter = wh[..., 0] * wh[..., 1]  # [N,M]

            union = area1[:, None] + area2[None, :] - inter
            iou = inter / union.clamp(min=1e-12)
            return iou

        def _pairwise_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
            """
            boxes1: [N,4], boxes2: [M,4]
            return: [N,M] 浜ら潰绉?
            """
            lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
            rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
            wh = (rb - lt).clamp(min=0.0)
            inter = wh[..., 0] * wh[..., 1]
            return inter

        def _containment_filter_single_class(
                boxes: torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor,
                contain_thr: float,
                contain_score_ratio_thr: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            澶勭悊鈥滃ぇ妗嗗灏忔鈥濅絾 IoU 涓嶉珮鐨勬儏鍐点€?
            瀵瑰悓绫绘锛岃嫢锛?
              inter / area(smaller) >= contain_thr
            涓斿皬妗嗗垎鏁版病鏈夋樉钁楅珮浜庡ぇ妗嗭紝
            鍒欏垹鎺夎緝宸殑閭ｄ釜锛堥粯璁ゅ€惧悜淇濈暀楂樺垎妗嗭級銆?
            """
            if boxes.size(0) <= 1:
                return boxes, scores, labels

            # 鎸夊垎鏁伴檷搴忥紝浼樺厛淇濈暀楂樺垎妗?
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

            areas = _box_area(boxes)  # [N]
            inter = _pairwise_intersection(boxes, boxes)  # [N,N]

            keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)

            N = boxes.size(0)
            for i in range(N):
                if not keep[i]:
                    continue
                for j in range(i + 1, N):
                    if not keep[j]:
                        continue

                    # 鍚岀被鎵嶅鐞嗭紙铏界劧杩欓噷鏈韩灏辨槸鍗曠被瀛愰泦锛屼粛淇濈暀淇濇姢锛?
                    if labels[i] != labels[j]:
                        continue

                    ai = areas[i]
                    aj = areas[j]
                    inter_ij = inter[i, j]

                    if ai <= 0 or aj <= 0 or inter_ij <= 0:
                        continue

                    smaller = torch.minimum(ai, aj)
                    contain_ratio = inter_ij / smaller.clamp(min=1e-12)

                    # 涓€鏂瑰ぇ閮ㄥ垎琚彟涓€鏂瑰寘鍚?
                    if contain_ratio >= contain_thr:
                        # 鍒嗘暟浣庣殑鑻ユ病鏈夋槑鏄句紭鍔匡紝鍒欏垹鎺?
                        # 鍥犱负褰撳墠鏄寜鍒嗘暟闄嶅簭锛岄粯璁ゅ垹 j 鏇村悎鐞?
                        if scores[j] <= scores[i] / contain_score_ratio_thr:
                            keep[j] = False
                        else:
                            # 鏋佸皯瑙侊細鍚庨潰鐨勫垎鏁版樉钁楁洿楂橈紝鍒欏垹鍓嶉潰
                            keep[i] = False
                            break

            return boxes[keep], scores[keep], labels[keep]

        def _center_distance_filter_single_class(
                boxes: torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor,
                center_dist_rel_thr: float,
                center_score_ratio_thr: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            澶勭悊鈥滀腑蹇冨嚑涔庝竴鑷翠絾灏哄/闀垮鐣ユ湁宸紓鈥濈殑閲嶅妗嗐€?
            鍒ゅ畾鎬濊矾锛?
              1) 涓ゆ涓績璺濈瓒冲杩?
              2) 璺濈鐩稿闃堝€肩敤杈冨皬妗嗗瑙掔嚎褰掍竴鍖?
              3) 鍒嗘暟宸窛涓嶅ぇ鏃讹紝淇濈暀鏇撮珮鍒嗘
            """
            if boxes.size(0) <= 1:
                return boxes, scores, labels

            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
            h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
            diag = torch.sqrt(w * w + h * h)  # [N]

            keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)
            N = boxes.size(0)

            for i in range(N):
                if not keep[i]:
                    continue
                for j in range(i + 1, N):
                    if not keep[j]:
                        continue

                    if labels[i] != labels[j]:
                        continue

                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = torch.sqrt(dx * dx + dy * dy)

                    ref_diag = torch.minimum(diag[i], diag[j]).clamp(min=1e-6)
                    rel_dist = dist / ref_diag

                    if rel_dist <= center_dist_rel_thr:
                        # 涓績闈炲父杩戯紝鑻ュ悗鑰呭垎鏁版病鏈夋槑鏄句紭鍔匡紝鍒欏垹鎺夊悗鑰?
                        if scores[j] <= scores[i] / center_score_ratio_thr:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break

            return boxes[keep], scores[keep], labels[keep]

        def _process_one_class(
                boxes_c: torch.Tensor,
                scores_c: torch.Tensor,
                labels_c: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            瀵瑰崟涓被鍒殑涓€缁勬渚濇鍋氾細
              1) NMS
              2) containment filter
              3) center-distance suppression
            """
            if boxes_c.numel() == 0:
                return boxes_c, scores_c, labels_c

            # 1) 绫诲唴 NMS
            iou_thr = 0.3 if nms_iou_thr is None else float(nms_iou_thr)
            keep = nms(boxes_c, scores_c, iou_thr)
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]
            labels_c = labels_c[keep]

            # 2) containment filter
            boxes_c, scores_c, labels_c = _containment_filter_single_class(
                boxes_c, scores_c, labels_c,
                contain_thr=contain_thr,
                contain_score_ratio_thr=contain_score_ratio_thr,
            )

            # 3) center-distance suppression
            boxes_c, scores_c, labels_c = _center_distance_filter_single_class(
                boxes_c, scores_c, labels_c,
                center_dist_rel_thr=center_dist_rel_thr,
                center_score_ratio_thr=center_score_ratio_thr,
            )

            return boxes_c, scores_c, labels_c

        # =========================================================
        # init empty containers per image
        # =========================================================
        boxes_list = [[] for _ in range(batch_size)]
        scores_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        # =========================================================
        # collect raw boxes
        # =========================================================
        for p in proposals:
            box5 = p["box"]  # [5] = (batch_idx,x1,y1,x2,y2)
            b = int(box5[0].item())
            if b < 0 or b >= batch_size:
                continue

            xyxy = box5[1:5].view(1, 4)
            c = int(p["class_id"])
            s = p["score"]

            boxes_list[b].append(xyxy)
            scores_list[b].append(s)
            labels_list[b].append(c)

        # =========================================================
        # per-image sanitize
        # =========================================================
        eps = 1e-6
        minsz = float(min_box_size)

        for b in range(batch_size):
            Himg, Wimg = int(image_sizes[b][0]), int(image_sizes[b][1])

            if len(boxes_list[b]) == 0:
                pseudo.append({
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            boxes = torch.cat(boxes_list[b], dim=0).to(device=device, dtype=torch.float32)
            scores = torch.tensor(scores_list[b], device=device, dtype=torch.float32).view(-1)
            labels = torch.tensor(labels_list[b], device=device, dtype=torch.long).view(-1)

            # 1) drop non-finite
            finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            boxes, scores, labels = boxes[finite], scores[finite], labels[finite]

            if boxes.numel() == 0:
                pseudo.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
                continue

            # 2) enforce ordering
            x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
            y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
            x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
            y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            # 3) clamp to image bounds
            boxes[:, 0].clamp_(0.0, max(Wimg - 1, 0))
            boxes[:, 2].clamp_(0.0, max(Wimg - 1, 0))
            boxes[:, 1].clamp_(0.0, max(Himg - 1, 0))
            boxes[:, 3].clamp_(0.0, max(Himg - 1, 0))

            # 4) remove non-positive size
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            pos = (w > eps) & (h > eps)
            boxes, scores, labels = boxes[pos], scores[pos], labels[pos]

            if boxes.numel() == 0:
                pseudo.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
                continue

            # 5) expand too-small boxes to min_box_size
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w = (boxes[:, 2] - boxes[:, 0]).clamp(min=eps)
            h = (boxes[:, 3] - boxes[:, 1]).clamp(min=eps)

            w2 = torch.clamp(w, min=minsz)
            h2 = torch.clamp(h, min=minsz)

            nx1 = cx - 0.5 * w2
            ny1 = cy - 0.5 * h2
            nx2 = cx + 0.5 * w2
            ny2 = cy + 0.5 * h2

            nx1 = nx1.clamp(0.0, max(Wimg - 1, 0))
            nx2 = nx2.clamp(0.0, max(Wimg - 1, 0))
            ny1 = ny1.clamp(0.0, max(Himg - 1, 0))
            ny2 = ny2.clamp(0.0, max(Himg - 1, 0))

            ax1 = torch.minimum(nx1, nx2)
            ay1 = torch.minimum(ny1, ny2)
            ax2 = torch.maximum(nx1, nx2)
            ay2 = torch.maximum(ny1, ny2)
            boxes = torch.stack([ax1, ay1, ax2, ay2], dim=-1)

            # 6) final strict validity
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            valid = (w > eps) & (h > eps) & torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]

            if boxes.numel() == 0:
                if (not keep_empty):
                    boxes = torch.tensor([[0.0, 0.0, minsz, minsz]], device=device, dtype=torch.float32)
                    boxes[:, 2].clamp_(0.0, max(Wimg - 1, 0))
                    boxes[:, 3].clamp_(0.0, max(Himg - 1, 0))
                    scores = torch.tensor([1.0], device=device, dtype=torch.float32)
                    labels = torch.tensor([1], device=device, dtype=torch.long)

                pseudo.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
                continue

            # =====================================================
            # 7) Per-class process:
            #    NMS + containment filter + center suppression
            # =====================================================
            out_boxes = []
            out_scores = []
            out_labels = []

            unique_classes = torch.unique(labels).tolist()
            for c in unique_classes:
                idx_c = torch.nonzero(labels == c, as_tuple=False).squeeze(1)
                if idx_c.numel() == 0:
                    continue

                boxes_c = boxes[idx_c]
                scores_c = scores[idx_c]
                labels_c = labels[idx_c]

                boxes_c, scores_c, labels_c = _process_one_class(
                    boxes_c, scores_c, labels_c
                )

                if boxes_c.numel() > 0:
                    out_boxes.append(boxes_c)
                    out_scores.append(scores_c)
                    out_labels.append(labels_c)

            if len(out_boxes) > 0:
                boxes = torch.cat(out_boxes, dim=0)
                scores = torch.cat(out_scores, dim=0)
                labels = torch.cat(out_labels, dim=0)

                # 鍙€夛細璺ㄧ被鍏ㄥ眬鍘婚噸
                if do_global_dedup and boxes.size(0) > 1:
                    keep = nms(boxes, scores, global_nms_iou_thr)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                order = torch.argsort(scores, descending=True)
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]
            else:
                boxes = boxes[:0]
                scores = scores[:0]
                labels = labels[:0]

            # 8) optional non-empty fallback
            if (not keep_empty) and boxes.numel() == 0:
                boxes = torch.tensor([[0.0, 0.0, minsz, minsz]], device=device, dtype=torch.float32)
                boxes[:, 2].clamp_(0.0, max(Wimg - 1, 0))
                boxes[:, 3].clamp_(0.0, max(Himg - 1, 0))
                scores = torch.tensor([1.0], device=device, dtype=torch.float32)
                labels = torch.tensor([1], device=device, dtype=torch.long)

            pseudo.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        return pseudo


    def _evaluate_pseudo_labels(self,
        pseudo_labels: List[Dict[str, torch.Tensor]],  # pseudo labels
        gt_targets: List[Dict[str, torch.Tensor]]  # ground-truth targets for evaluate pseudo labels
    )-> float:
        """
        Evaluate pseudo labels using ground-truth targets.
        """
        preds = []
        for p in pseudo_labels:
            # copy tensors (avoid in-place side effects)
            boxes = p["boxes"].detach()
            scores = p["scores"].detach()
            labels = p["labels"].detach()

            # drop background (labels <= 0)
            keep = labels > 0
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        targets = []
        for t in gt_targets:
            boxes = t["boxes"].detach()
            labels = t["labels"].detach()

            labels = labels + 1     # shift to [1..C]

            targets.append({
                "boxes": boxes,
                "labels": labels,
            })

        map_metric = MeanAveragePrecision(
            iou_type="bbox",
            # iou_thresholds=torch.arange(0.5, 0.96, 0.05).tolist(),  # mAP@[0.50 : 0.95]
            iou_thresholds=[0.5],  # mAP@[0.50]
            max_detection_thresholds=[1, 10, 100],
        ).to(self.config.device)

        map_metric.update(preds, targets)
        result = map_metric.compute()

        return float(result["map"])


    def forward(
        self,
        # train_phase: str,  # "warmup" or "main"
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor,  # class label for weak boxes, one-hot, [R, num_classes], range [0, num_classes - 1]
        bg_prototype: Dict[str, torch.Tensor],  # background prototype
        gt_targets: List[Dict[str, torch.Tensor]] = None,  # ground-truth targets for evaluate pseudo labels
        sam3_targets: List[Dict[str, torch.Tensor]] = None,
        # rpn_teacher: Optional[nn.Module] = None,  # teacher RPN for no-grad pseudo labels
        vis_dir: str = None,  # for debug : visualization output dir
        epoch: int = 0,  # for debug : current epoch
        it: int = 0  # for debug : current iteration
    ) -> Dict[str, Any]:
        """
        :return:
        out : Dict[str, Any], for warmpu phase, contains:
        - loss_ccam: torch.Tensor, CCAM loss
        - batch_bg_prototype: torch.Tensor, batch background prototype for EMA update, shape [D]
        - constrain_losses_dict: Dict[str, torch.Tensor], constrain losses
                                {
                                    "loss_constrain" : total_loss,
                                    "loss_proto" : loss_proto,
                                    "loss_pull" : loss_pull,
                                    "loss_push" : loss_push,
                                }
        - loss_obj: torch.Tensor, sparse proposal objectness loss
        for main phase, also contains:
        - rpn_losses_dict: Dict[str, torch.Tensor], RPN losses
                            {
                                "loss_objectness": loss_objectness,
                                "loss_rpn_box_reg": loss_rpn_box_reg,
                            }
        - pseudo_labels_mAP: float, mAP of pseudo labels
        """
        out : Dict[str, Any] = {}   # output container

        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]

        # -----get weak box feature maps-----
        wb_feature_maps = self.roi_align_h2wb(high_feature_maps, wboxes)  # [R, C_h, H_wb, W_wb]

        # -----Branch 1: get CCAM for proposal fg & bg scores-----
        # get CCAM and loss_ccam
        ccam, loss_ccam = self.ccam_generator(wb_feature_maps)  # [R, 1, H_wb, W_wb]

        out.update({
            "loss_ccam" : loss_ccam
        })

        # -----Branch 2: get seed proposals-----
        # 1) get dense proposals
        img_size = (imgs.shape[-1], imgs.shape[-2])  # (H_img, W_img)
        img_size_list = [img_size for _ in range(imgs.shape[0])]

        rpn_targets = self._build_rpn_targets(
            wboxes=wboxes,
            wb_labels=wb_labels,
            sam3_targets=sam3_targets,
        )
        # # for debug
        # with open("debug_data.txt", "w") as f:
        #     for b in range(imgs.shape[0]):
        #         f.write(f"Image {b}:\n")
        #         f.write(f"  RPN Targets:\n")
        #         t = rpn_targets[b]
        #         for i in range(t["box"].shape[0]):
        #             box = t["box"][i].tolist()
        #             label = t["class_id"][i].item()
        #             score = t["score"][i].item()
        #             f.write(f"    Box: {box}, score: {score}, Label: {label}\n")


        rpn_targets_for_train = [
            {
                "boxes": target["box"],
                "labels": target["class_id"],
            }
            for target in rpn_targets
        ]

        dense_proposals_list, rpn_losses_dict = self._get_dense_proposals(
            high_feature_maps=high_feature_maps,
            image_sizes=img_size_list,
            images_tensor=imgs,
            rpn_targets=rpn_targets_for_train,
        )  # List[Tensor], len=B, each Tensor is [num_proposals, 4]

        out.update({
            "rpn_losses_dict": rpn_losses_dict
        })

        # 2) split dense proposals into sparse proposals and background proposals based on wboxes
        B = imgs.shape[0]
        sparse_proposals_list = []  # len=B, each Tensor is [num_props, 4]
        sparse_proposal_labels_list = []  # len=B, each Tensor is [num_props]
        bg_proposals_list = []  # len=B, each Tensor is [num_bg_props, 4]

        for b in range(B):
            # all proposals in this image
            props_xyxy = dense_proposals_list[b]    # [num_props, 4]

            # all weak boxes in this image
            wb_mask = (wboxes[:, 0].long() == b)    # bool mask for wboxes in this image, shape [R]
            wbs = wboxes[wb_mask, 1:5]  # [num_wbs, 4]
            wbl = wb_labels[wb_mask]  # one-hot class labels for wboxes in this image, shape [num_wbs, num_classes], range [0, num_classes - 1]

            # completely inside wbox
            # broadcast
            p = props_xyxy[:, None, :]  # [num_props, 1, 4]
            w = wbs[None, :, :]  # [1, num_wbs, 4]
            # inside matrix, shape [num_props, num_wbs]
            inside = (p[..., 0] >= w[..., 0]) & (p[..., 1] >= w[..., 1]) & \
                     (p[..., 2] <= w[..., 2]) & (p[..., 3] <= w[..., 3])
            # iou matrix for proposals and wboxes, shape [num_props, num_wbs]
            ious = box_iou(props_xyxy, wbs)

            # initial sparse mask: any inside
            sparse_mask = inside.any(dim=1)  # [num_props]

            # no proposals completely inside any wbox, keep at least one proposal
            if not sparse_mask.any():
                max_iou_per_prop, _ = ious.max(dim=1)  # [num_props]
                # keep proposals whose max IoU with any wbox >= iou threshold
                iou_keep = max_iou_per_prop >= float(self.config.keep_iou_thr)  # [num_props]
                if iou_keep.any():
                    sparse_mask = iou_keep
                else:   # no proposals whose max IoU with any wbox >= iou threshold
                    # keep the proposal with highest IoU with any wbox
                    best_idx = torch.argmax(max_iou_per_prop)
                    sparse_mask = torch.zeros_like(max_iou_per_prop, dtype=torch.bool)
                    sparse_mask[best_idx] = True

            # assign class labels for sparse proposals based on wboxes
            # if proposal inside multiple wboxes: pick max IoU among inside ones for assign label
            big_neg = torch.finfo(ious.dtype).min   # large negative value for masking
            # for each proposal, get the max IoU with wboxes that it is inside of; if not inside any wbox, assign big negative
            masked_ious = torch.where(inside, ious, torch.tensor(big_neg, device=ious.device, dtype=ious.dtype))    # [num_props, num_wbs]
            wbox_best_inside = torch.argmax(masked_ious, dim=1)  # [num_props]
            wbox_best_iou = torch.argmax(ious, dim=1)  # [num_props]

            # for sparse proposals: choose best_inside if truly inside; else best_iou
            use_inside = inside.any(dim=1)
            wbox_best = torch.where(use_inside, wbox_best_inside, wbox_best_iou)  # index of best wbox for each proposal, shape [num_props]

            # assign labels for sparse proposals
            assigned_wbl = wbl[wbox_best]   # [num_props, num_classes]
            prop_labels_all = torch.argmax(assigned_wbl, dim=1).long()  # [num_props], range [0, num_classes - 1]
            prop_labels_all = prop_labels_all + 1  # shift to range [1, num_classes], where 0 is reserved for background

            # split sparse / bg
            sparse_props = props_xyxy[sparse_mask]
            sparse_labels = prop_labels_all[sparse_mask]
            bg_props = props_xyxy[~sparse_mask]

            sparse_proposals_list.append(sparse_props)
            sparse_proposal_labels_list.append(sparse_labels)
            bg_proposals_list.append(bg_props)


        # 3) get CCAM foreground scores of sparse proposals
        R, _, Hc, Wc = ccam.shape
        sparse_props_ccam_scores_per_img: List[Optional[torch.Tensor]] = [None for _ in range(B)]

        # invert ccam
        # ccam = 1 - ccam

        for b in range(B):
            wboxes_in_img : Dict[int, torch.Tensor] = {}  # {idx : wbox_xyxy(shape [4], image coords)}
            for i in range(R):
                if int(wboxes[i, 0].item()) == b:
                    wboxes_in_img[i] = wboxes[i, 1:5]

            props_xyxy = sparse_proposals_list[b]  # [num_props, 4], image coords
            num_props = props_xyxy.shape[0]

            # match sparse proposal with wbox
            wbox_global_indices = list(wboxes_in_img.keys())
            wbs = torch.stack([wboxes_in_img[idx] for idx in wbox_global_indices], dim=0)  # [num_wbs, 4]
            inside = (
                    (props_xyxy[:, None, 0] >= wbs[None, :, 0]) &
                    (props_xyxy[:, None, 1] >= wbs[None, :, 1]) &
                    (props_xyxy[:, None, 2] <= wbs[None, :, 2]) &
                    (props_xyxy[:, None, 3] <= wbs[None, :, 3])
            )   # [num_props, num_wbs]
            ious = box_iou(props_xyxy, wbs)     # [num_props, num_wbs]
            big_neg = torch.finfo(ious.dtype).min
            masked_ious = torch.where(inside, ious, torch.tensor(big_neg, device=ious.device, dtype=ious.dtype))
            # if one sparse proposal match multiple wboxes, pick the one with highest IoU among inside ones
            wbox_best_inside = torch.argmax(masked_ious, dim=1)  # [num_props]
            # if one sparse proposal not inside any wbox, pick the wbox with highest IoU
            wbox_best_iou = torch.argmax(ious, dim=1)  # [num_props]
            use_inside = inside.any(dim=1)  # [num_props]
            chosen_local_widx = torch.where(
                use_inside,
                wbox_best_inside,
                wbox_best_iou
            )  # [num_props]

            # initialize scores in this img
            img_scores = torch.zeros(
                (num_props,), device=self.config.device, dtype=torch.float32
            )

            # for each sparse proposal, get ccam fg score
            for local_j, global_widx in enumerate(wbox_global_indices):
                prop_mask = (chosen_local_widx == local_j)  # [num_props]
                props_for_this_wbox = props_xyxy[prop_mask]  # [num_assigned, 4]
                wbox_xyxy = wboxes[global_widx, 1:5]  # [4], image coords
                ccam_i = ccam[global_widx, 0]  # [Hc, Wc]
                # image coords -> roi coords in this weak box's CCAM
                ccam_boxes = self._map_boxes_image_to_roi_xyxy(
                    boxes_img_xyxy=props_for_this_wbox,
                    wbox_xyxy=wbox_xyxy,
                    cam_hw=(Hc, Wc),
                )  # [num_assigned, 4], roi coords
                fg_scores, bg_scores = self._ccam_fg_bg_score_for_boxes(
                    ccam_i=ccam_i,
                    cam_boxes_xyxy=ccam_boxes,
                )  # [num_assigned], [num_assigned]

                scores = fg_scores - bg_scores  # [num_assigned]
                img_scores[prop_mask] = scores

            sparse_props_ccam_scores_per_img[b] = img_scores

        sparse_props_ccam_scores_list = []
        for b in range(B):
            sparse_props_ccam_scores_list.append(sparse_props_ccam_scores_per_img[b])

        sparse_props_ccam_scores = torch.cat(sparse_props_ccam_scores_list, dim=0)  # [total_num_sparse_proposals]

        # # for debug
        # with open("debug_data.txt", 'w') as f:
        #     for score in sparse_props_ccam_scores.cpu().tolist():
        #         f.write(f"{score}\n")

        # get sparse proposal & background proposal features
        high_feature_maps_dict = {"0" : high_feature_maps}
        image_shapes = [(imgs.shape[-2], imgs.shape[-1]) for _ in range(B)]
        sparse_proposal_features = self.roi_align_h2p(
            high_feature_maps_dict,
            sparse_proposals_list,
            image_shapes
        )    # [total_num_sparse_proposals, C_h, H_p, W_p]
        bg_proposal_features = self.roi_align_h2p(
            high_feature_maps_dict,
            bg_proposals_list,
            image_shapes
        )    # [total_num_bg_proposals, C_h, H_p, W_p]

        sparse_proposal_boxes_list = []
        bg_proposal_boxes_list = []
        for b in range(B):
            props_xyxy = sparse_proposals_list[b]  # [num_props, 4]
            bg_props_xyxy = bg_proposals_list[b]  # [num_bg_props, 4]

            batch_idx = torch.full((props_xyxy.shape[0],), b, device=self.config.device, dtype=torch.long)
            batch_idx_bg = torch.full((bg_props_xyxy.shape[0],), b, device=self.config.device, dtype=torch.long)

            props_boxes = torch.cat([batch_idx.unsqueeze(1), props_xyxy], dim=1)
            bg_props_boxes = torch.cat([batch_idx_bg.unsqueeze(1), bg_props_xyxy], dim=1)

            sparse_proposal_boxes_list.append(props_boxes)
            bg_proposal_boxes_list.append(bg_props_boxes)

        sparse_proposal_boxes = torch.cat(sparse_proposal_boxes_list, dim=0)  # [total_num_sparse_proposals, 5], each props is [batch_idx, x1, y1, x2, y2]
        # bg_proposal_boxes = torch.cat(bg_proposal_boxes_list, dim=0)  # [total_num_bg_proposals, 5], each props is [batch_idx, x1, y1, x2, y2]

        sparse_proposal_labels = torch.cat(sparse_proposal_labels_list, dim=0)  # [total_num_sparse_proposals], range [1, num_classes]

        # 4) embed sparse proposal features, background proposal features and prototypes
        # sparse proposal features to embeddings
        sparse_proposal_features = self.gap(sparse_proposal_features)  # [total_num_sparse_proposals, C_h, 1, 1]
        sparse_proposal_features = sparse_proposal_features.view(sparse_proposal_features.shape[0], -1)    # [total_num_sparse_proposals, C_h]
        sparse_proposal_embeddings = self.proj(sparse_proposal_features)  # [total_num_sparse_proposals, D = embed_dim]
        # background proposal features to embeddings
        bg_proposal_features = self.gap(bg_proposal_features)  # [total_num_bg_proposals, C_h, 1, 1]
        bg_proposal_features = bg_proposal_features.view(bg_proposal_features.shape[0], -1)    # [total_num_bg_proposals, C_h]
        bg_proposal_embeddings = self.proj(bg_proposal_features)  # [total_num_bg_proposals, D = embed_dim]
        # build batch background prototype: LogSumExp pooling of background proposal features
        N = bg_proposal_features.shape[0]
        batch_bg_prototype = torch.logsumexp(bg_proposal_features, dim=0) - torch.log(torch.tensor(float(N), device=self.config.device))  # [D]
        key = 'bg'
        if key not in bg_prototype:  # first time initialization
            bg_prototype[key] = batch_bg_prototype

        out.update({
            "batch_bg_prototype" : batch_bg_prototype
        })

        # prototypes to embeddings
        prototypes = torch.stack([self.dataset_mps[k] for k in range(self.config.num_classes)], dim=0)  # [num_classes, D_p = C_h]
        prototypes_embeddings = self.proj(prototypes)  # [num_classes, D]
        prototypes_embeddings_norm = F.normalize(prototypes_embeddings, dim=1)  # [num_classes, D]
        # background prototype to embedding
        bg_prototype_embedding = self.proj(bg_prototype['bg'].unsqueeze(0))  # [1, D]
        bg_prototype_embedding = bg_prototype_embedding.squeeze(0)  # [D]
        # L2 norm
        bg_prototype_embedding_norm = F.normalize(bg_prototype_embedding, dim=0)  # [D]

        # 5) compute foreground scores for sparse proposals
        # L2 norm
        sparse_proposal_embeddings_norm = F.normalize(sparse_proposal_embeddings, dim=1)  # [total_num_sparse_proposals, D]
        bg_proposal_embeddings_norm = F.normalize(bg_proposal_embeddings, dim=1)  # [total_num_bg_proposals, D]
        # get object scores
        sparse_proposal_object_logits = self.obj_classifier(sparse_proposal_embeddings)  # [total_num_sparse_proposals, num_classes + 1]
        sparse_proposal_object_scores = F.softmax(sparse_proposal_object_logits, dim=1)  # [total_num_sparse_proposals, num_classes + 1]
        bg_proposal_object_logits = self.obj_classifier(bg_proposal_embeddings)  # [total_num_bg_proposals, num_classes + 1]
        sparse_proposal_object_scores_list = []
        for i in range(sparse_proposal_labels.shape[0]):
            c = int(sparse_proposal_labels[i].item())
            sparse_proposal_object_scores_list.append(sparse_proposal_object_scores[i, c])
        sparse_proposal_object_scores = torch.stack(sparse_proposal_object_scores_list, dim=0)  # [total_num_sparse_proposals]

        sparse_proposal_fg_scores = self._compute_fg_scores(
            sparse_props_ccam_scores=sparse_props_ccam_scores,
            sparse_proposal_embeddings=sparse_proposal_embeddings_norm,
            sparse_proposal_labels=sparse_proposal_labels,
            sparse_proposal_boxes=sparse_proposal_boxes,
            prototypes_embeddings=prototypes_embeddings_norm,
            bg_prototype_embedding=bg_prototype_embedding_norm,
            sparse_proposal_object_scores=sparse_proposal_object_scores,
            w_prototype_sim=self.config.w_prototype_sim,
            w_ccam_score=self.config.w_ccam_score,
            w_obj_score=self.config.w_obj_score
        )       # [total_num_sparse_proposals]

        # # for debug
        # with open("debug_data.txt", "a", encoding="utf-8") as f:
        #     f.write("Sparse proposal foreground scores:\n")
        #     for i in range(sparse_proposal_fg_scores.shape[0]):
        #         score = sparse_proposal_fg_scores[i].item()
        #         f.write(f"Proposal {i}: Score: {score}\n")

        # compute loss_constrain
        constrain_losses_dict = get_constrain_loss(
            proposal_embeddings=sparse_proposal_embeddings_norm,
            prototypes_embeddings=prototypes_embeddings_norm,
            bg_prototype_embedding=bg_prototype_embedding_norm,
            labels=sparse_proposal_labels,
            fg_scores=sparse_proposal_fg_scores,
            w_proto_loss=self.config.w_proto_loss,
            w_pull_loss=self.config.w_pull_loss,
            w_push_loss=self.config.w_push_loss
        )
        out.update({
            "constrain_losses_dict" : constrain_losses_dict
        })

        # compute loss_obj
        loss_obj = get_object_loss(
            proposal_obj_logits=sparse_proposal_object_logits,
            bg_proposal_obj_logits=bg_proposal_object_logits,
            proposal_labels=sparse_proposal_labels,
            fg_scores=sparse_proposal_fg_scores
        )
        out.update({
            "loss_obj": loss_obj
        })

        # if train_phase == "warmup":
        #     return out

        # 6) Object Discovery
        # select Top-Scoring proposals for each class
        top_scoring_proposal_embeddings_norm = torch.zeros(
            (self.config.num_classes, sparse_proposal_embeddings.shape[1]),
            device=self.config.device
        )
        for c in range(1, self.config.num_classes + 1):
            cls_mask = (sparse_proposal_labels == c)    # bool mask for proposals of class c, shape [total_num_sparse_proposals]
            if cls_mask.any():
                fg_scores = sparse_proposal_fg_scores[cls_mask]  # [num_props_in_cls]
                indices = torch.nonzero(cls_mask).squeeze(1)    # [num_props_in_cls]

                # select Top-Scoring proposal
                top_idx_in_cls = indices[fg_scores.argmax()]
                top_scoring_proposal_embeddings_norm[c - 1] = sparse_proposal_embeddings_norm[top_idx_in_cls]

        # get threshold for each class
        thresholds = F.cosine_similarity(top_scoring_proposal_embeddings_norm, prototypes_embeddings_norm, dim=1)  # [num_classes]
        sims_all = torch.matmul(sparse_proposal_embeddings_norm, prototypes_embeddings_norm.t())  # [total_num_sparse_proposals, num_classes]

        # # for debug
        # sims_with_bg_all = torch.matmul(sparse_proposal_embeddings_norm, torch.cat([bg_prototype_embedding_norm.unsqueeze(0), prototypes_embeddings_norm], dim=0).t())  # [total_num_sparse_proposals, num_classes + 1]
        # with open("debug_data.txt", "a", encoding="utf-8") as f:
        #     f.write("\nSim for all proposal:\n")
        #     for i in range(sims_with_bg_all.shape[0]):
        #         sim_values = sims_with_bg_all[i].cpu().detach().numpy().tolist()
        #         f.write(f"Proposal {i}: label: {sparse_proposal_labels[i]}, score: {sparse_proposal_fg_scores[i]}, " + f", ".join([f"Class {c}: {sim:.4f}" for c, sim in enumerate(sim_values)]) + "\n")

        # get seed proposals
        labels = sparse_proposal_labels.to(dtype=torch.long) - 1    # shift to range [0, num_classes - 1]
        sim_for_label = sims_all.gather(1, labels.view(-1, 1)).squeeze(1)  # [total_num_sparse_proposals]
        thr_for_label = thresholds[labels]  # [total_num_sparse_proposals]
        # keep only if its class-specific similarity passes its class threshold
        keep_mask = sim_for_label > thr_for_label  # [total_num_sparse_proposals], dtype = bool

        # with open("debug_data.txt", "w", encoding="utf-8") as f:
        #     f.write("Sim per proposal for its assigned class:\n")
        #     for i in range(sim_for_label.shape[0]):
        #         sim = sim_for_label[i].item()
        #         thr = thr_for_label[i].item()
        #         keep = keep_mask[i].item()
        #         f.write(f"Proposal {i}: Sim: {sim:.4f}, Thr: {thr:.4f}, Keep: {keep}\n")

        # keep at least one proposal for per-image
        batch_idx = sparse_proposal_boxes[:, 0].long()  # [N], image index
        for b in range(B):
            idx_b = torch.where(batch_idx == b)[0]

            # if no proposal kept for this image -> force keep the highest-score proposal
            if not keep_mask[idx_b].any():
                scores_b = sparse_proposal_fg_scores[idx_b]
                top_idx_in_b = torch.argmax(scores_b).item()
                keep_mask[idx_b[top_idx_in_b]] = True

        keep_indices = torch.where(keep_mask)[0]  # [N_keep]
        seed_proposals: List[Dict] = []
        for i in keep_indices:
            c = int(sparse_proposal_labels[i].item())
            seed_proposals.append({
                "box": sparse_proposal_boxes[i],  # [5] = (batch_idx, x1, y1, x2, y2)
                "class_id": c,  # class label, range [1, num_classes]
                "score": sparse_proposal_fg_scores[i],  # scalar, foreground score
                "logit": sparse_proposal_object_logits[i, c],  # scalar, logit of class c
            })

        # # one-to-one match
        # match_losses_dict, matched_indices = match_seed_proposals_with_sam3_targets(
        #     seed_proposals=seed_proposals,
        #     rpn_targets=rpn_targets,
        #     imgs=imgs,
        #     cost_class=self.config.cost_class,
        #     cost_bbox=self.config.cost_bbox,
        #     cost_giou=self.config.cost_giou,
        #     match_focal_alpha=self.config.match_focal_alpha,
        #     match_focal_gamma=self.config.match_focal_gamma,
        #     lambda_match_cls=self.config.lambda_match_cls,
        #     lambda_match_l1=self.config.lambda_match_l1,
        #     lambda_match_giou=self.config.lambda_match_giou,
        # )
        #
        # out.update({
        #     "match_losses_dict" : match_losses_dict
        # })
        #
        # seed_proposals = [seed_proposals[i] for i in matched_indices]


        # # for debug
        # with open("debug_data.txt", "w") as f:
        #     seed_proposals_per_img = [[] for _ in range(B)]
        #     for p in seed_proposals:
        #         b = int(p["box"][0].item())
        #         seed_proposals_per_img[b].append(p)
        #     f.write("Seed proposals:\n")
        #     for b in range(B):
        #         f.write(f"Image {b}:\n")
        #         for p in seed_proposals_per_img[b]:
        #             box = p["box"][1:5].cpu().numpy().tolist()
        #             score = p["score"].cpu().item()
        #             class_id = p["class_id"]
        #             f.write(f"  Box: {box}, Score: {score}, Class ID: {class_id}\n")

        # # get aug seed proposals
        # Himg, Wimg = imgs.shape[-2], imgs.shape[-1]
        # aug_seed_proposals: List[Dict] = []
        # for p in seed_proposals:
        #     box5 = p["box"]  # [5] = (batch_idx, x1, y1, x2, y2)
        #     b = int(box5[0].item())
        #     xyxy = box5[1:5]  # [4]
        #
        #     aug_xyxy = self._get_multi_bboxes(
        #         box_xyxy=xyxy,
        #         img_hw=(Himg, Wimg),
        #         do_aug=True
        #     )   # [M = (origin + aug), 5]
        #
        #     batch_idx = torch.full((aug_xyxy.size(0), 1), b, device=aug_xyxy.device, dtype=aug_xyxy.dtype)
        #     aug_box5 = torch.cat([batch_idx, aug_xyxy], dim=1)  # [M, 5]
        #
        #     for j in range(aug_box5.size(0)):
        #         aug_seed_proposals.append({
        #             "box": aug_box5[j],  # [5] = (batch_idx, x1, y1, x2, y2)
        #             "class_id": p["class_id"],
        #             "score": p["score"],
        #             "logit": p["logit"],
        #         })

        # with open("debug_data.txt", "a") as f:
        #     aug_seed_proposals_per_img = [[] for _ in range(B)]
        #     for p in aug_seed_proposals:
        #         b = int(p["box"][0].item())
        #         aug_seed_proposals_per_img[b].append(p)
        #     f.write("Augmented Seed proposals:\n")
        #     for b in range(B):
        #         f.write(f"Image {b}:\n")
        #         for p in aug_seed_proposals_per_img[b]:
        #             box = p["box"][1:5].cpu().numpy().tolist()
        #             score = p["score"].cpu().item()
        #             class_id = p["class_id"]
        #             f.write(f"  Box: {box}, Score: {score}, Class ID: {class_id}\n")

        # 7) get pseudo labels for RPN training
        # pseudo_labels: list length B, each element:
        #                 {
        #                     "boxes": Tensor [num_boxes, 4] in image coords (xyxy),
        #                     "scores": Tensor [num_boxes],
        #                     "labels": Tensor [num_boxes] in range [1, C] (0 reserved for background),
        #                 }
        image_sizes = [(imgs.shape[-2], imgs.shape[-1]) for _ in range(B)]  # [(Himg,Wimg),...]
        # pseudo_labels: list length B, each:
        #               {
        #                   "boxes":  FloatTensor [Nb,4] in image coords (xyxy),
        #                   "scores": FloatTensor [Nb],
        #                   "labels": LongTensor  [Nb] in [1..C]  (0 reserved for background)
        #               }
        # pseudo_labels = self._build_rpn_pseudo_labels(
        #     proposals=seed_proposals,
        #     batch_size=B,
        #     image_sizes=image_sizes,
        # )
        pseudo_labels = self.rpn_pseudo_label_generator(
            proposals=seed_proposals,
            batch_size=B,
            image_sizes=image_sizes,
        )

        # for debug
        with open("debug_data.txt", "w") as f:
            f.write("\nPseudo Labels:\n")
            for b in range(B):
                f.write(f"Image {b}:\n")
                boxes = pseudo_labels[b]["boxes"].cpu().numpy().tolist()
                scores = pseudo_labels[b]["scores"].cpu()
                labels = pseudo_labels[b]["labels"].cpu()
                for box, score, label in zip(boxes, scores, labels):
                    f.write(f"  Box: {box}, Score: {score.item()}, Label: {label.item()}\n")

        # # -----get proposals-----
        # self.rpn.train()
        # B, _, Himg, Wimg = imgs.shape
        #
        # # build ImageList
        # image_sizes = [(Himg, Wimg) for _ in range(B)]
        # img_list = ImageList(tensors=imgs, image_sizes=image_sizes)
        #
        # # build targets in the format expected by RPN: List[Dict], each requires "boxes"
        # rpn_targets: List[Dict[str, torch.Tensor]] = []
        # for b in range(B):
        #     boxes = pseudo_labels[b]["boxes"]  # [Nb,4] xyxy in image coords
        #     # safety checks
        #     # clamp to image bounds
        #     boxes[:, 0].clamp_(0, Wimg - 1)
        #     boxes[:, 2].clamp_(0, Wimg - 1)
        #     boxes[:, 1].clamp_(0, Himg - 1)
        #     boxes[:, 3].clamp_(0, Himg - 1)
        #     # ensure x1<=x2, y1<=y2
        #     x1 = torch.min(boxes[:, 0], boxes[:, 2])
        #     x2 = torch.max(boxes[:, 0], boxes[:, 2])
        #     y1 = torch.min(boxes[:, 1], boxes[:, 3])
        #     y2 = torch.max(boxes[:, 1], boxes[:, 3])
        #     boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        #
        #     rpn_targets.append({"boxes": boxes})
        #
        # # build features dict for RPN
        # features = {"0": high_feature_maps}
        # # get proposals
        # # proposals : List[Tensor], len=B, each Tensor is [num_proposals, 4] in image coords
        # # rpn_losses_dict : Dict[str, Tensor], RPN losses
        # #                    {
        # #                        "loss_objectness": loss_objectness,
        # #                        "loss_rpn_box_reg": loss_rpn_box_reg,
        # #                    }
        # proposals, rpn_losses_dict = self.rpn(img_list, features, rpn_targets)
        #
        # out.update({
        #     "rpn_losses_dict": rpn_losses_dict
        # })


        # # -----R-CNN Head-----
        # det_targets = [
        #     {
        #         "boxes": pseudo_labels[b]["boxes"],
        #         "labels": pseudo_labels[b]["labels"],
        #     }
        #     for b in range(B)
        # ]
        # detections : List[Dict[str, torch.Tensor]], len=B, each Dict has
        #              {
        #                   "boxes" : [N_boxes, 4],
        #                   "labels" : [N_boxes], range [1, C], 0 reserved for background
        #                   "scores" : [N_boxes], softmax confidence
        #              }
        # det_losses_dict: Dict[str, torch.Tensor], Detection losses
        #                  {
        #                       "loss_classifier": loss_classifier,
        #                       "loss_box_reg": loss_box_reg
        #                  }
        # self.roi_heads.train()
        # _, det_losses_dict = self.roi_heads(
        #     features={"0": high_feature_maps},
        #     proposals=proposals,
        #     image_shapes=image_sizes,
        #     targets=det_targets,
        # )
        # with torch.no_grad():
        #     self.roi_heads.eval()
        #     detections = self.roi_heads(
        #         features={"0": high_feature_maps},
        #         proposals=proposals,
        #         image_shapes=image_sizes
        #     )
        # # get Top-k scoring detections
        # det_topk_ratio = 0.5
        # topk_detections: List[Dict] = []
        # for b in range(B):
        #     det_b = detections[b]
        #     boxes = det_b["boxes"]
        #     scores = det_b["scores"]
        #     labels = det_b["labels"]
        #
        #     N = boxes.shape[0]
        #     k = int(max(1, round(det_topk_ratio * N)))
        #     topk_indices = torch.topk(scores, k=k, largest=True).indices
        #     topk_detections.append({
        #         "boxes": boxes[topk_indices],
        #         "scores": scores[topk_indices],
        #         "labels": labels[topk_indices],
        #     })
        #
        # # -----one-to-one match-----
        # matched_pairs, unmatched_dets = self._match_with_hungarian(
        #     topk_detections=topk_detections,
        #     aug_seed_proposals=aug_seed_proposals,
        #     num_classes=self.config.num_classes,
        # )
        #
        # # get final detections
        # # final_detections : List[Dict[str, torch.Tensor]], len=B, each Dict has
        # #              {
        # #                   "boxes" : [N_boxes, 4],
        # #                   "labels" : [N_boxes], range [1, C], 0 reserved for background
        # #                   "scores" : [N_boxes], softmax confidence
        # #              }
        # final_detections : List[Dict] = []
        # boxes_list = [[] for _ in range(B)]
        # labels_list = [[] for _ in range(B)]
        # scores_list = [[] for _ in range(B)]
        #
        # for det, seed in matched_pairs:
        #     b = int(det["box"][0].item())
        #
        #     det_box = det["box"][1:5]
        #     det_score = det["score"]
        #     det_label = seed["class_id"] + 1  # shift to range [1, C]
        #
        #     boxes_list[b].append(det_box.view(1, 4))
        #     labels_list[b].append(torch.tensor([det_label], device=device, dtype=torch.long))
        #     scores_list[b].append(det_score.view(1))
        #
        # for b in range(B):
        #     final_detections.append({
        #         "boxes": torch.cat(boxes_list[b], dim=0),  # [N_boxes, 4]
        #         "labels": torch.cat(labels_list[b], dim=0),  # [N_boxes]
        #         "scores": torch.cat(scores_list[b], dim=0),  # [N_boxes]
        #     })
        #
        # # compute match loss
        # match_losses_dict = compute_match_loss(
        #     matched_pairs=matched_pairs,
        #     unmatched_dets=unmatched_dets,
        #     imgs=imgs,
        #     num_classes=self.config.num_classes,
        #     match_focal_alpha=self.config.match_focal_alpha,
        #     match_focal_gamma=self.config.match_focal_gamma,
        #     lambda_match_cls=self.config.lambda_match_cls,
        #     lambda_match_l1=self.config.lambda_match_l1,
        #     lambda_match_giou=self.config.lambda_match_giou
        # )

        # evaluate pseudo labels
        pseudo_labels_mAP = self._evaluate_pseudo_labels(pseudo_labels, gt_targets)


        out.update({
            "pseudo_labels_mAP" : pseudo_labels_mAP
        })

        # for debug
        rpn_targets_for_eval = [
            {
                "boxes": rpn_targets[b]["box"],
                "labels": rpn_targets[b]["class_id"],
                "scores": rpn_targets[b]["score"],
            }
            for b in range(B)
        ]
        # pseudo_labels_mAP_1 = self._evaluate_pseudo_labels(rpn_targets_for_eval, gt_targets)
        visualize_stage2_debug_batch(
            imgs=imgs,  # [B,3,H,W]
            wboxes=wboxes,  # [R,5]
            ccam=ccam,  # [R,1,Hc,Wc]
            # final_detections=final_detections,
            # final_detections=detections[0],
            final_detections=rpn_targets_for_eval,
            # final_detections=sam3_targets,
            pseudo_labels=pseudo_labels,
            gt_targets=gt_targets,
            out_dir=vis_dir,
            step_tag=f"e{epoch}_it{it}",
        )

        return out

# For backbone verification
class LinearProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16 with aligned weights
        config : LinearProbConfig       # LinearProb configuration
    )->None:
        super(LinearProb, self).__init__()

        self.encoder = backbone
        # freeze backbone weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.in_size, self.config.in_size))

        in_dim = self.config.in_c * self.config.in_size * self.config.in_size
        self.classifier_0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.classifier_6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=self.config.out_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m)->None:
        """
        Initialize weights for Linear
        :param m: Module to initialize
        """
        if isinstance(m, nn.Linear):  # Check if the module is a Linear layer
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
            if m.bias is not None:  # Initialize bias to zero if it exists
                nn.init.constant_(m.bias, 0)

    def forward(self, x : torch.Tensor)-> Dict[str, torch.Tensor]:
        '''
        :param x: input images, [B, C, H, W]
        :return: logits for 7 classes, {class_id: logit}
        '''

        self.encoder[-1] = nn.Identity()
        feature_maps = self.encoder(x)
        pooled_features = self.avgpool(feature_maps)  # [B, C, 1, 1]

        logits = {}
        logits['0'] = self.classifier_0(pooled_features)
        logits['1'] = self.classifier_1(pooled_features)
        logits['2'] = self.classifier_2(pooled_features)
        logits['3'] = self.classifier_3(pooled_features)
        logits['4'] = self.classifier_4(pooled_features)
        logits['5'] = self.classifier_5(pooled_features)
        logits['6'] = self.classifier_6(pooled_features)

        return logits

# For prototypes verification
class PrototypeChecker(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16 with aligned weights
        patch_embed : PatchEmbed,       # Patch Embed layer from Stage1
        config : PrototypeCheckerConfig       # PrototypeChecker configuration
    )-> None:
        super(PrototypeChecker, self).__init__()

        self.encoder = backbone
        self.config = config

        self.roi_align = RoIAlign(
            output_size=config.roi_out_size,
            spatial_scale=config.spatial_scale,
            sampling_ratio=config.sampling_ratio,
            aligned=config.aligned,
        )

        self.patch_embed = patch_embed

    def forward(
        self,
        x: torch.Tensor,        # input images, [B, C, H, W]
        boxes: torch.Tensor,        # GT boxes for RoI Align, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        boxes_labels : torch.Tensor,        # class labels for GT boxes, [R, num_classes]
        prototypes : Dict[int, torch.Tensor],        # Normalized, {class_id, prototype tensor}
        return_details: bool = True,
        lse_alpha : float = 10.0,
        lse_eps : float = 1e-6
    ) -> Dict[str, Any]:
        '''
        :return: Similarity of GT and prototypes, {class_id : average_similarity}
        '''
        # get GT patch features
        self.encoder[-1] = nn.Identity()
        feature_maps = self.encoder(x)
        roi_features = self.roi_align(feature_maps, boxes)  # [R, C, H, W]
        patch_features = self.patch_embed(roi_features)  # [R, num_patches, D]

        # sort by class_id
        class_ids = sorted(list(prototypes.keys()))
        proto_mat = torch.stack([prototypes[k] for k in class_ids], dim=0)  # [num_classes, D]

        # statistics锛歴um and cnt
        R, num_classes = boxes_labels.shape
        sims_sum = {k: 0.0 for k in range(num_classes)}
        sims_cnt = {k: 0 for k in range(num_classes)}

        # statistics: results details
        details = {
            "pos": {k: [] for k in range(num_classes)},  # sim_pos of each GT
            "neg_max": {k: [] for k in range(num_classes)},  # max sim_neg of each GT
            "margin": {k: [] for k in range(num_classes)},  # margin = sim_pos - max sim_neg
        }

        for i in range(R):
            gt_label = torch.argmax(boxes_labels[i]).item()

            gt_patch = patch_features[i]  # [num_patches, D]
            gt_patch = F.normalize(gt_patch, dim=1)

            # # mean pooling
            # gt_vec = gt_patch.mean(dim=0, keepdim=True)  # [1, D]
            # gt_vec = F.normalize(gt_vec, dim=1)  # [1, D]

            # LogSumExp pooling
            m = gt_patch.max(dim=0, keepdim=True).values  # [1, D]
            lse = m + torch.log(torch.exp(lse_alpha * (gt_patch - m)).mean(dim=0, keepdim=True) + lse_eps) / lse_alpha
            gt_vec = F.normalize(lse, dim=1, eps=lse_eps)  # [1, D]

            # similarity of GT and prototypes(all classes)
            sims_all = torch.matmul(gt_vec, proto_mat.t()).squeeze(0)  # [num_classes]

            sim_pos = sims_all[gt_label].item()     # positive class similarity

            # max negative class similarity
            if num_classes > 1:
                mask = torch.ones(num_classes, dtype=torch.bool, device=sims_all.device)
                mask[gt_label] = False
                sim_neg_max = sims_all[mask].max().item()
            else:
                sim_neg_max = float("-inf")

            margin = sim_pos - sim_neg_max if sim_neg_max != float("-inf") else float("inf")

            sims_sum[gt_label] += sim_pos
            sims_cnt[gt_label] += 1

            if return_details:
                details["pos"][gt_label].append(sim_pos)
                details["neg_max"][gt_label].append(sim_neg_max)
                details["margin"][gt_label].append(margin)

        out = {
            "sum": sims_sum,
            "cnt": sims_cnt,
        }
        if return_details:
            out["details"] = details

        return out


def build_Stage1_model(
    stage1_config: Stage1Config,        # Stage1 configuration
)-> nn.Module:
    backbone, hook = build_vgg16_backbone_with_hook(stage1_config.layer_indices)

    # mp_generator
    mp_generator = MorphologicalPrototypeGenerator(
        num_classes=stage1_config.num_classes,
        in_c=stage1_config.in_c,
        patch_size=stage1_config.patch_size,
        embed_dim=stage1_config.embed_dim,
        components_range=stage1_config.components_range,
        random_state=stage1_config.random_state,
        max_iter=stage1_config.max_iter,
        roi_out_size=stage1_config.roi_out_size_mid,
        spatial_scale=stage1_config.spatial_scale_mid,
        sampling_ratio=stage1_config.sampling_ratio
    )

    model = Stage1(
        backbone=backbone,
        hook=hook,
        mp_generator=mp_generator,
        config=stage1_config
    )

    return model

def build_Stage2_model(
    stage2_config: Stage2Config,        # Stage2 configuration
    backbone_weights_path: str,  # Aligned backbone weights path
    dataset_mps_path: str       # Morphological Prototype path
)-> nn.Module:
    backbone = vgg16(pretrained=False).features
    backbone.load_state_dict(torch.load(backbone_weights_path))
    # backbone = vgg16(pretrained=True).features

    # load morphological prototypes
    dataset_mps = torch.load(dataset_mps_path)  # Dict[int, Tensor]

    model = Stage2(
        backbone=backbone,
        dataset_mps=dataset_mps,
        config=stage2_config
    )

    return model

def build_LinearProb_model(
    linear_prob_config: LinearProbConfig,       # LinearProb configuration
    backbone_weights_path: str  # Aligned backbone weights path
)-> nn.Module:
    backbone = vgg16(pretrained=False).features
    backbone.load_state_dict(torch.load(backbone_weights_path))
    # backbone = vgg16(pretrained=True).features

    model = LinearProb(
        backbone=backbone,
        config=linear_prob_config
    )

    return model

def build_PrototypeChecker_model(
    prototypeChecker_config : PrototypeCheckerConfig,
    backbone_weights_path : str,  # Aligned backbone weights path
    patch_embed_weights_path : str      # Patch Embed weights path
)-> nn.Module:
    backbone = vgg16(pretrained=False).features
    backbone.load_state_dict(torch.load(backbone_weights_path))

    patch_embed = PatchEmbed(
        img_size=prototypeChecker_config.roi_out_size,
        patch_size=prototypeChecker_config.patch_size,
        in_chans=prototypeChecker_config.in_c,
        embed_dim=prototypeChecker_config.embed_dim,
    )
    patch_embed.load_state_dict(torch.load(patch_embed_weights_path))

    model = PrototypeChecker(
        backbone=backbone,
        patch_embed=patch_embed,
        config=prototypeChecker_config
    )

    return model


