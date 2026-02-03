# Morphological Prototype R-CNN
import os.path

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Union
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
from torchvision.ops.boxes import generalized_box_iou
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import MultiScaleRoIAlign

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
    # for proto loss
    proto_loss_tau: float  # temperature tau for proto loss
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
    # for RoI Head
    det_fg_iou_thresh: float  # foreground IoU threshold
    det_bg_iou_thresh: float  # background IoU threshold
    det_batch_size_per_image: int  # detection batch size per image
    det_positive_fraction: float  # detection positive fraction
    det_score_thresh: float  # detection score threshold
    det_nms_thresh: float   # detection NMS threshold
    detections_per_img: int # number of detections per image
    # for RoI Align
    roi_out_size_h2wb: Tuple[int]  # output size for high feature maps to weak box features
    spatial_scale_h2wb: float  # spatial scale for high feature maps to weak box features
    roi_out_size_wb2p: Tuple[int]  # output size for weak box features to proposal box features
    spatial_scale_wb2p: float  # spatial scale
    roi_out_size_h2p: Tuple[int]  # output size for high feature maps to proposal box features
    spatial_scale_h2p: float    # spatial scale for high feature maps to proposal box features
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True    # aligned flag

@dataclass
class Stage2CCAMConfig:
    device: torch.device  # device
    freeze_backbone: bool  # freeze backbone weights
    in_c: int  # input channels
    # for CCAM
    ccam_threshold: float  # CCAM threshold
    # for RoI Align
    roi_out_size_h2wb: Tuple[int, int]  # output size for high feature maps to weak box features
    spatial_scale_h2wb: float  # spatial scale for high feature maps to weak box features
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag

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

        # (train)RoI Align for high-level feature maps to weak box feature maps
        self.roi_align_h2wb = RoIAlign(
            output_size=self.config.roi_out_size_h2wb,
            spatial_scale=self.config.spatial_scale_h2wb,
            sampling_ratio=self.config.sampling_ratio,
            aligned=self.config.aligned,
        )
        # (train)RoI Align for weak box feature maps to proposal box feature maps
        self.roi_align_wb2p = RoIAlign(
            output_size=self.config.roi_out_size_wb2p,
            spatial_scale=self.config.spatial_scale_wb2p,
            sampling_ratio=self.config.sampling_ratio,
            aligned=self.config.aligned,
        )
        # (train and inference) RoI Align for high-level feature maps to proposal box feature maps
        # self.roi_align_h2p = RoIAlign(
        #     output_size=self.config.roi_out_size_h2p,
        #     spatial_scale=self.config.spatial_scale_h2p,
        #     sampling_ratio=self.config.sampling_ratio,
        #     aligned=self.config.aligned,
        # )
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

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Projection Head
        hidden_dim_in = self.config.in_c
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.embed_dim)
        )

        # Object Classifier
        self.obj_classifier = nn.Linear(config.embed_dim, self.config.num_classes + 1)  # num_classes + 1(for background class)

        # CCAM Generator
        self.ccam_generator = CCAMGenerator(in_c=self.config.in_c)

        # One-to-One Matcher
        self.matcher = HungarianMatcher(
            cost_class=self.config.cost_class,
            cost_bbox=self.config.cost_bbox,
            cost_giou=self.config.cost_giou
        )

        # Box Head
        resolution = self.config.roi_out_size_h2p[0]
        box_head = TwoMLPHead(
            in_channels=self.config.in_c * resolution * resolution,
            representation_size=self.config.hidden_dim,
        )

        # Box Predictor（cls + reg）
        box_predictor = FastRCNNPredictor(
            in_channels=self.config.hidden_dim,
            num_classes=self.config.num_classes + 1,  # include background
        )

        # RoI Heads (roi align + box head + box predictor)
        self.roi_heads = RoIHeads(
            box_roi_pool=self.roi_align_h2p,
            box_head=box_head,
            box_predictor=box_predictor,

            fg_iou_thresh=self.config.det_fg_iou_thresh,
            bg_iou_thresh=self.config.det_bg_iou_thresh,
            batch_size_per_image=self.config.det_batch_size_per_image,
            positive_fraction=self.config.det_positive_fraction,

            bbox_reg_weights=None,
            score_thresh=self.config.det_score_thresh,
            nms_thresh=self.config.det_nms_thresh,
            detections_per_img=self.config.detections_per_img,
        )

    @torch.no_grad()
    def _get_dense_proposals(
        self,
        wb_feature_maps : torch.Tensor,   # weak box feature maps, [R, C_h, H_wb, W_wb]
        wboxes: torch.Tensor  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
    )-> List[torch.Tensor]:
        ''':return dense_proposals : List[Tensor], len=R, each Tensor is [num_proposals, 4]'''
        self.rpn.eval()

        device = wb_feature_maps.device
        R = wb_feature_maps.shape[0]

        # prepare img_list
        image_sizes = []
        for i in range(R):
            _, x1, y1, x2, y2 = wboxes[i]

            # weak box 的真实尺寸（原图坐标系）
            h = int(y2 - y1)
            w = int(x2 - x1)

            image_sizes.append((h, w))
        dummy_images = torch.zeros((R, 3, 1, 1), device=device)     # 实际不使用，仅占位
        img_list = ImageList(tensors=dummy_images, image_sizes=image_sizes)

        features = {"0": wb_feature_maps}  # RPN features format, Dict[str, tensor]
        dense_proposals_list, _ = self.rpn(img_list, features)

        return dense_proposals_list

    def _get_proto_loss(
        self,
        dense_proposal_embeddings: torch.Tensor,  # Normalized, [N_D, D]
        prototypes_embeddings: torch.Tensor,  # Normalized, [C, D]
        best_cls: torch.Tensor,   # [N_D] long, 0~C-1
        keep_mask: torch.Tensor,  # [N_D] bool
        tau: float,
        eps: float = 1e-8,
    ):
        """
        仅对 keep_mask == True 的“可靠 proposals”计算。
        :return:
        """
        device = dense_proposal_embeddings.device
        if dense_proposal_embeddings.numel() == 0:
            return torch.tensor(0.0, device=device)

        if keep_mask is None or keep_mask.numel() == 0 or (keep_mask.sum() == 0):
            # 没有可靠 proposal，则不施加该约束（避免噪声）
            return torch.tensor(0.0, device=device)

        z = dense_proposal_embeddings[keep_mask]  # [N_keep, D]
        y = best_cls[keep_mask].long()  # [N_keep]
        p = prototypes_embeddings  # [C, D]

        # logits: [N_keep, C]
        logits = (z @ p.t()) / max(tau, eps)

        # InfoNCE with single positive == CE over prototype logits
        loss = F.cross_entropy(logits, y)

        return loss

    def _get_multi_bboxes(
        self,
        cam : np.ndarray,     # [h, w, 1]
        cam_thr : float = 0.2,        # threshold, [0, 1]
        area_ratio : float =0.5
    )-> List[List[int]]:
        """
        Copy from : https://github.com/MingXiangL/SPE
        :return: estimated bounding box: len(contours), each is [x1, y1, x2, y2]
        """
        # with open("debug_data.txt", "a", encoding="utf-8") as f:
        #     f.write(f"CCAM: (shape: {cam.shape})\n")
        #     for line in cam:
        #         f.write(" ".join([f"{val:.4f}" for val in line.flatten().tolist()]) + "\n")
        # cam = 1 - cam
        cam = (cam * 255.).astype(np.uint8)
        map_thr = cam_thr * np.max(cam)


        _, thr_gray_heatmap = cv2.threshold(cam,
                                            int(map_thr), 255,
                                            cv2.THRESH_TOZERO)
        # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)
        # for debug visualization
        # cv2.imwrite("debug_thr_gray_heatmap_1.png", thr_gray_heatmap)

        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            estimated_bbox = []
            areas = list(map(cv2.contourArea, contours))
            area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
            for idx in area_idx:
                if areas[idx] >= areas[area_idx[0]] * area_ratio:
                    c = contours[idx]
                    x, y, w, h = cv2.boundingRect(c)
                    estimated_bbox.append([x, y, x + w, y + h])
            # areas1 = sorted(areas, reverse=True)

            # pdb.set_trace()

            # estimated_bbox = [x, y, x + w, y + h]
        else:
            estimated_bbox = [[0, 0, 1, 1]]

        return estimated_bbox  # , thr_gray_heatmap, len(contours)

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

    def _xyxy_to_cxcywh(self, xyxy: torch.Tensor) -> torch.Tensor:
        # xyxy: [N,4]
        x1, y1, x2, y2 = xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        return torch.stack([cx, cy, w, h], dim=-1)

    def _match_sparse_seed_with_hungarian(
        self,
        sparse_proposals: List[Dict],
        seed_proposals: List[Dict],
        num_classes: int,
        other_class_logit: float = -20.0,
    ) -> Tuple[List[Tuple[Dict, Dict]], List[Dict]]:
        """
        :return:
          - matched_pairs: List[(sparse_dict, seed_dict)]
          - unmatched_sparse: List[sparse_dict]  (treated as background for cls focal)
        """
        if len(sparse_proposals) == 0 or len(seed_proposals) == 0:
            return [], list(sparse_proposals)

        device = sparse_proposals[0]["box"].device
        matched_pairs: List[Tuple[Dict, Dict]] = []
        unmatched_sparse: List[Dict] = []

        # all batch ids
        batch_ids = sorted(set(
            [int(p["box"][0].item()) for p in sparse_proposals] +
            [int(t["box"][0].item()) for t in seed_proposals]
        ))

        for b in batch_ids:
            sparse_b = [p for p in sparse_proposals if int(p["box"][0].item()) == b]
            seed_b = [t for t in seed_proposals if int(t["box"][0].item()) == b]

            if len(sparse_b) == 0:
                continue
            if len(seed_b) == 0:
                # no target => all sparse are unmatched(background)
                unmatched_sparse.extend(sparse_b)
                continue

            Ns = len(sparse_b)

            # boxes for matcher
            pred_boxes_xyxy = torch.stack([p["box"][1:5] for p in sparse_b], dim=0).to(device=device,
                                                                                       dtype=torch.float32)
            pred_boxes = self._xyxy_to_cxcywh(pred_boxes_xyxy)

            # pred_logits for matcher: use only foreground C dims (exclude background dim)
            pred_logits = torch.full((1, Ns, num_classes), other_class_logit, device=device, dtype=torch.float32)
            for i, p in enumerate(sparse_b):
                cls = int(p["class_id"])
                if "cls_logits" in p:
                    # take foreground logit
                    fg_logit = p["cls_logits"][cls].to(device=device, dtype=torch.float32)
                else:
                    logit = p.get("logit", None)
                    if logit is None:
                        score = p.get("score", 0.5)
                        score = float(score) if not torch.is_tensor(score) else float(score.item())
                        fg_logit = torch.log(
                            torch.tensor(score, device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6))
                    else:
                        fg_logit = logit if torch.is_tensor(logit) else torch.tensor(float(logit), device=device)
                        fg_logit = fg_logit.to(device=device, dtype=torch.float32)
                pred_logits[0, i, cls] = fg_logit

            outputs = {
                "pred_logits": pred_logits,  # [1, Ns, C]
                "pred_boxes": pred_boxes.unsqueeze(0),  # [1, Ns, 4]
            }

            # targets
            tgt_labels = torch.tensor([int(t["class_id"]) for t in seed_b], device=device, dtype=torch.long)
            tgt_boxes_xyxy = torch.stack([t["box"][1:5] for t in seed_b], dim=0).to(device=device, dtype=torch.float32)
            tgt_boxes = self._xyxy_to_cxcywh(tgt_boxes_xyxy)
            targets = [{"labels": tgt_labels, "boxes": tgt_boxes}]

            # hungarian
            indices = self.matcher(outputs, targets)
            src_idx, tgt_idx = indices[0]

            matched_src = set(src_idx.tolist())
            for si, ti in zip(src_idx.tolist(), tgt_idx.tolist()):
                matched_pairs.append((sparse_b[si], seed_b[ti]))

            # collect unmatched
            for i in range(Ns):
                if i not in matched_src:
                    unmatched_sparse.append(sparse_b[i])

        return matched_pairs, unmatched_sparse

    def _softmax_focal_loss(
        self,
        logits: torch.Tensor,  # [N, C+1]
        targets: torch.Tensor,  # [N] in [0..C] (C=background)
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)  # [N, C+1]
        probs = torch.softmax(logits, dim=-1)  # [N, C+1]

        idx = torch.arange(logits.size(0), device=logits.device)
        pt = probs[idx, targets]  # [N]
        logpt = log_probs[idx, targets]  # [N]

        loss = - (alpha * (1.0 - pt).pow(gamma) * logpt)  # [N]
        return loss.mean()

    def _compute_loss_match(
        self,
        sparse_proposals: List[Dict],  # all sparse (matched+unmatched)
        matched_pairs: List[Tuple[Dict, Dict]],  # matched pairs
        imgs: torch.Tensor,  # [B, C, H, W]
    ) -> Dict[str, torch.Tensor]:

        device = imgs.device
        C = self.config.num_classes
        bg_id = C  # background index in [0..C]

        N = len(sparse_proposals)
        if N == 0:
            z = torch.tensor(0.0, device=device)
            return {"loss_match": z, "loss_match_cls": z, "loss_match_l1": z, "loss_match_giou": z}

        # ---------- (1) classification targets for ALL sparse ----------
        # default all background
        cls_targets = torch.full((N,), bg_id, device=device, dtype=torch.long)

        # matched -> set to seed class_id (or sparse class_id; usually一致)
        matched_sp_idx = []
        for sp, sd in matched_pairs:
            matched_sp_idx.append(int(sp["sp_idx"]))
            cls_targets[int(sp["sp_idx"])] = int(sd["class_id"])  # safer: use seed label as GT

        # logits [N, C+1]
        cls_logits = torch.stack([p["cls_logits"].to(device=device, dtype=torch.float32) for p in sparse_proposals],
                                 dim=0)

        alpha = self.config.match_focal_alpha
        gamma = self.config.match_focal_gamma
        loss_cls = self._softmax_focal_loss(cls_logits, cls_targets, alpha=alpha, gamma=gamma)

        # ---------- (2) regression only for matched ----------
        K = len(matched_pairs)
        if K == 0:
            loss_l1 = torch.tensor(0.0, device=device)
            loss_giou = torch.tensor(0.0, device=device)
        else:
            matched_sparse_boxes = torch.stack([p[0]["box"] for p in matched_pairs], dim=0).to(device=device)
            matched_seed_boxes = torch.stack([p[1]["box"] for p in matched_pairs], dim=0).to(device=device)

            sparse_xyxy = matched_sparse_boxes[:, 1:5].float()
            seed_xyxy = matched_seed_boxes[:, 1:5].float()

            # normalize L1 by image size
            H, W = imgs.shape[-2], imgs.shape[-1]
            scale = torch.tensor([W, H, W, H], device=device, dtype=torch.float32).unsqueeze(0)
            loss_l1 = F.l1_loss(sparse_xyxy / scale, seed_xyxy / scale, reduction="none").sum(dim=1).mean()

            giou = generalized_box_iou(sparse_xyxy, seed_xyxy)
            loss_giou = (1.0 - giou.diag()).mean()

        lam_cls = self.config.lambda_match_cls
        lam_l1 = self.config.lambda_match_l1
        lam_giou = self.config.lambda_match_giou

        loss_match = lam_cls * loss_cls + lam_l1 * loss_l1 + lam_giou * loss_giou

        return {
            "loss_match": loss_match,
            "loss_match_cls": loss_cls,
            "loss_match_l1": loss_l1,
            "loss_match_giou": loss_giou,
        }

    def _build_rpn_pseudo_labels_from_matched_sparse(
        self,
        matched_sparse_boxes: torch.Tensor,  # [K,5] (batch_idx,x1,y1,x2,y2)
        matched_sparse_scores: torch.Tensor,  # [K]
        matched_labels: torch.Tensor,  # [K]
        batch_size: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """:return:  list length B, each element: {"boxes": [Nb,4] xyxy in image coords, "scores": [Nb], "labels": [Nb]  (1..C, 0 reserved for background)}"""
        device = matched_sparse_boxes.device
        pseudo = []

        if matched_sparse_boxes.numel() == 0:
            for _ in range(batch_size):
                pseudo.append({
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                })
            return pseudo

        batch_idx = matched_sparse_boxes[:, 0].long()  # [K]
        boxes_xyxy = matched_sparse_boxes[:, 1:5].float()  # [K,4]
        scores = matched_sparse_scores.float()  # [K]

        # IMPORTANT:
        # RoIHeads expects labels in [1..C], with 0 reserved for background
        labels = matched_labels.long() + 1  # [K]

        for b in range(batch_size):
            mask = (batch_idx == b)
            pseudo.append({
                "boxes": boxes_xyxy[mask],
                "scores": scores[mask],
                "labels": labels[mask],
            })

        return pseudo

    def _refine_pseudo_labels_nms_topk(
        self,
        pseudo_by_img: List[Dict[str, torch.Tensor]],
        iou_thr: float = 0.7,
        topk: int = 200
    ) -> List[Dict[str, torch.Tensor]]:
        refined = []
        for item in pseudo_by_img:
            boxes = item["boxes"]
            scores = item["scores"]
            labels = item["labels"]

            if boxes.numel() == 0:
                refined.append(item)
                continue

            # topk
            if scores.numel() > topk:
                idx = torch.topk(scores, k=topk, largest=True).indices
                boxes = boxes[idx]
                scores = scores[idx]

            # nms
            keep = nms(boxes, scores, iou_thr)
            refined.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return refined

    def _evaluate_pseudo_labels(self,
        pseudo_labels: List[Dict[str, torch.Tensor]],  # pseudo labels
        gt_targets: List[Dict[str, torch.Tensor]]  # ground-truth targets for evaluate pseudo labels
    )-> float:
        """
        Evaluate pseudo labels using ground-truth targets.
        """
        # Copy from torchvision references
        map_metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=torch.arange(0.5, 0.96, 0.05).tolist(),  # 0.50 : 0.05 : 0.95
            max_detection_thresholds=[1, 10, 100],
        ).to(self.config.device)

        map_metric.update(pseudo_labels, gt_targets)
        result = map_metric.compute()

        return float(result["map"])

    def _forward_train(
        self,
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor,  # class label for weak boxes, [R, num_classes]
        gt_targets: List[Dict[str, torch.Tensor]] = None  # ground-truth targets for evaluate pseudo labels
    )-> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, List[Dict[str, torch.Tensor]]]:
        """
        :return:
        - loss_proto: torch.Tensor, Prototype loss
        - loss_ccam: torch.Tensor, CCAM loss
        - match_losses_dict: Dict[str, torch.Tensor], One-to-One Match losses
                            {
                                "loss_match": loss_match,
                                "loss_match_cls": loss_cls,
                                "loss_match_l1": loss_l1,
                                "loss_match_giou": loss_giou,
                            }
        - rpn_losses_dict: Dict[str, torch.Tensor], RPN losses
                            {
                                "loss_objectness": loss_objectness,
                                "loss_rpn_box_reg": loss_rpn_box_reg,
                            }
        - det_losses_dict: Dict[str, torch.Tensor], Detection losses
                            {
                                "loss_classifier": loss_classifier,
                                "loss_box_reg": loss_box_reg
                            }
        - pseudo_labels_mAP: float, mAP of pseudo labels
        - detections: List[Dict[str, torch.Tensor]], length B, each element:
                            {
                                "boxes": Tensor [Nd, 4] in image coords (xyxy),
                                "labels": Tensor [Nd] in [1..C] (0 reserved for background),
                                "scores": Tensor [Nd],
                            }
        """
        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]

        # -----get weak box feature maps-----
        wb_feature_maps = self.roi_align_h2wb(high_feature_maps, wboxes)  # [R, C_h, H_wb, W_wb]

        # -----Branch 1: get sparse proposals-----
        # get dense proposals
        dense_proposals_list = self._get_dense_proposals(wb_feature_maps, wboxes)  # List[Tensor], len=R, each Tensor is [num_proposals, 4]
        wbox_batch_idx = wboxes[:, 0].to(device=wb_feature_maps.device)  # [R]
        wbox_cls_ids = wb_labels.argmax(dim=1).to(device=wb_feature_maps.device)  # [R], 0~C-1
        dense_proposal_boxes_list = []
        dense_proposal_labels_list = []
        for i, props_xyxy in enumerate(dense_proposals_list):
            # props_xyxy: [num_proposals_i, 4]
            num_pi = props_xyxy.shape[0]

            bi = wbox_batch_idx[i].expand(props_xyxy.shape[0], 1).to(dtype=props_xyxy.dtype)  # [num_proposals, 1]
            props_5 = torch.cat([bi, props_xyxy], dim=1)  # [num_proposals, 5]
            dense_proposal_boxes_list.append(props_5)

            li = wbox_cls_ids[i].expand(num_pi)  # [num_pi]
            dense_proposal_labels_list.append(li)

        dense_proposal_boxes = torch.cat(dense_proposal_boxes_list, dim=0)  # [N_D = num_proposals, 5]
        dense_proposal_labels = torch.cat(dense_proposal_labels_list, dim=0).long()  # [N_D], 0~C-1

        # get proposal features
        dense_proposal_features = self.roi_align_wb2p(wb_feature_maps, dense_proposals_list)    # [N_D = total_num_proposals, C_h, H_p, W_p]

        # Object Discovery
        # Dense proposal features to embeddings
        dense_proposal_features = self.gap(dense_proposal_features)  # [N_D, C_h, 1, 1]
        dense_proposal_features = dense_proposal_features.view(dense_proposal_features.shape[0], -1)    # [N_D, C_h]
        dense_proposal_embeddings = self.proj(dense_proposal_features)  # [N_D, D = embed_dim]

        # get object scores
        dense_proposal_object_logits = self.obj_classifier(dense_proposal_embeddings)  # [N_D, num_classes + 1]
        dense_proposal_object_scores = F.softmax(dense_proposal_object_logits, dim=1)  # [N_D, num_classes + 1]

        # get prototypes embeddings
        prototypes = torch.stack([self.dataset_mps[k] for k in range(self.config.num_classes)], dim=0)  # [num_classes, D_p = C_h]
        prototypes_embeddings = self.proj(prototypes)  # [num_classes, D]

        # select Top-Scoring proposals for each class
        top_scoring_proposal_indices = dense_proposal_object_scores.argmax(dim=0)  # [num_classes + 1]
        top_scoring_proposal_embeddings = dense_proposal_embeddings[top_scoring_proposal_indices[:-1]]  # [num_classes, D]
        # top_scoring_proposal_boxes = dense_proposal_boxes[top_scoring_proposal_indices[:-1]]  # [num_classes, 5]

        # L2 Normalize
        dense_proposal_embeddings = F.normalize(dense_proposal_embeddings, dim=1)  # [N_D, D]
        prototypes_embeddings = F.normalize(prototypes_embeddings, dim=1)  # [num_classes, D]
        top_scoring_proposal_embeddings = F.normalize(top_scoring_proposal_embeddings, dim=1)  # [num_classes, D]

        # get threshold for each class
        thresholds = F.cosine_similarity(top_scoring_proposal_embeddings, prototypes_embeddings, dim=1)  # [num_classes]
        sims_all = torch.matmul(dense_proposal_embeddings, prototypes_embeddings.t())  # [N_D, num_classes]

        # get sparse proposals
        # N_D, C = sims_all.shape  # [N_D, C]
        labels = dense_proposal_labels.long()  # [N_D], 0~C-1  (each proposal belongs to its wbox class)
        sim_for_label = sims_all.gather(1, labels.view(-1, 1)).squeeze(1)  # [N_D]
        thr_for_label = thresholds[labels]  # [N_D]
        # keep only if its class-specific similarity passes its class threshold
        keep_mask = sim_for_label > thr_for_label  # [N_D] bool
        keep_indices = torch.where(keep_mask)[0]  # [N_keep]
        best_cls = labels
        best_sim = sim_for_label
        sparse_proposals: List[Dict] = []
        for i in keep_indices:
            c = int(labels[i].item())
            sparse_proposals.append({
                "sp_idx": len(sparse_proposals),  # global index for background padding
                "box": dense_proposal_boxes[i],  # [5], (batch_idx, x1, y1, x2, y2)
                "class_id": c,  # chosen unique class for this box
                "score": dense_proposal_object_scores[i, c],  # scalar
                "logit": dense_proposal_object_logits[i, c],  # scalar
                "cls_logits": dense_proposal_object_logits[i],  # [C+1] include background
                # for debugging/analysis
                "sim": best_sim[i].detach(),
                "dense_idx": int(i.item()),
            })
        with open("debug_data.txt", "w", encoding="utf-8") as f:
            f.write("Thresholds for each class:\n")
            for t in thresholds:
                f.write(f"{t.item()} ")
            f.write("\n")
            f.write(f"Similarity Matrix(shape: {sims_all.shape}):\n")
            for sims in sims_all:
                f.write(f"{sims}\n")
            f.write(f"Sparse proposals(number:{len(sparse_proposals)}):\n")
            for p in sparse_proposals:
                for k, v in p.items():
                    f.write(f"{k}: {v} \n")


        # get proto loss
        tau = self.config.proto_loss_tau
        loss_proto = self._get_proto_loss(
            dense_proposal_embeddings=dense_proposal_embeddings,
            prototypes_embeddings=prototypes_embeddings,
            best_cls=best_cls,
            keep_mask=keep_mask,
            tau=tau
        )

        # -----Branch 2: get seed proposals-----

        # get CCAM and loss_ccam
        ccam, loss_ccam = self.ccam_generator(wb_feature_maps)      # [R, 1, H_wb, W_wb]

        # get augmented seed proposals from CCAM
        seed_proposals : List[Dict] = []
        R, _, _, _ = ccam.shape
        for i in range(R):
            ccam_i = ccam[i].detach().cpu().numpy().transpose(1, 2, 0)      # [h, w, 1]

            # for debug: Visualize CCAM
            # imgs: torch.Tensor [B,C,H,W] (一般是RGB归一化)
            b = int(wboxes[i, 0].item())
            x1, y1, x2, y2 = wboxes[i, 1:5].detach().cpu().numpy().tolist()
            img_chw = imgs[b].detach().cpu().numpy()
            img_chw = denorm_imagenet(img_chw)  # -> [0,1]
            img_np = np.transpose(img_chw, (1, 2, 0))  # HWC
            img_u8 = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            overlay = overlay_ccam_on_image(
                img_bgr=img_bgr,
                ccam_hw1=ccam_i,  # (Hc,Wc,1)
                wbox_xyxy=(x1, y1, x2, y2),
                alpha=0.45,
                normalize_ccam=True
            )
            cv2.imwrite(f"./results/visualizations/ccam/debug_ccam_overlay_b{b}_r{i}.png", overlay)

            local_boxes_list = self._get_multi_bboxes(ccam_i, self.config.ccam_threshold)
            # 过滤“无 contours”的默认返回框
            if len(local_boxes_list) == 1 and local_boxes_list[0] == [0, 0, 1, 1]:
                continue
            # map local boxes to image boxes
            wbox_i = wboxes[i]  # [5]
            Himg, Wimg = imgs.shape[2], imgs.shape[3]
            global_boxes_xyxy = self._map_boxes_roi_to_image_xyxy(
                boxes_local=local_boxes_list,
                wbox=wbox_i,
                cam_hw=(ccam_i.shape[0], ccam_i.shape[1]),
                img_hw=(Himg, Wimg)
            )   # Tensor [N, 4]
            batch_idx = wbox_i[0].to(global_boxes_xyxy).view(1, 1).repeat(global_boxes_xyxy.size(0), 1)
            global_boxes_wbatch = torch.cat([batch_idx, global_boxes_xyxy], dim=1)      # [N, 5]

            class_id = wb_labels[i].argmax().item()
            for j in range(global_boxes_wbatch.size(0)):
                seed_proposals.append({
                    "box" : global_boxes_wbatch[j],  # Tensor [5], (batch_idx, x1, y1, x2, y2)
                    "class_id" : class_id,
                })
        with open("debug_data.txt", "a", encoding="utf-8") as f:
            f.write(f"Seed proposals(number:{len(seed_proposals)}):\n")
            for p in seed_proposals:
                for k, v in p.items():
                    f.write(f"{k}: {v} \n")

        # ----- one-to-one match-----
        matched_pairs, unmatched_sparse = self._match_sparse_seed_with_hungarian(
            sparse_proposals=sparse_proposals,
            seed_proposals=seed_proposals,
            num_classes=self.config.num_classes,
        )
        # debug : visualize matched pairs
        visualize_hungarian_matches(
            imgs=imgs,
            sparse_proposals=sparse_proposals,
            seed_proposals=seed_proposals,
            matched_pairs=matched_pairs,
            wboxes=wboxes,
            gt_targets=gt_targets,
        )

        # split matched pairs
        if len(matched_pairs) > 0:
            matched_sparse_boxes = torch.stack(
                [p[0]["box"] for p in matched_pairs], dim=0
            )  # [K = matched boxes, 5]
            # matched_seed_boxes = torch.stack(
            #     [p[1]["box"] for p in matched_pairs], dim=0
            # )  # [K, 5]
            matched_labels = torch.tensor(
                [p[1]["class_id"] for p in matched_pairs],
                device=matched_sparse_boxes.device,
                dtype=torch.long
            )
            matched_sparse_scores = torch.tensor(
                [float(p[0].get("score", 1.0)) if not torch.is_tensor(p[0].get("score", 1.0))
                 else float(p[0]["score"].detach().cpu().item())
                 for p in matched_pairs],
                device=matched_sparse_boxes.device,
                dtype=torch.float32
            )  # [K]
        else:
            matched_sparse_boxes = torch.zeros((0, 5), device=imgs.device)
            # matched_seed_boxes = torch.zeros((0, 5), device=imgs.device)
            matched_labels = torch.zeros((0,), device=imgs.device, dtype=torch.long)
            matched_sparse_scores = torch.zeros((0,), device=imgs.device, dtype=torch.float32)

        # compute match loss
        match_losses_dict = self._compute_loss_match(
            sparse_proposals=sparse_proposals,  # ALL sparse (matched+unmatched)
            matched_pairs=matched_pairs,
            imgs=imgs,
        )

        # -----build pseudo labels-----
        # pseudo_labels : List[Dict[str, torch.Tensor]], list length B, each element:
        #               {
        #                   "boxes": [Nb,4] xyxy in image coords,
        #                   "scores": [Nb],
        #                   "labels": [Nb]  (1..C, 0 reserved for background)
        #               }
        pseudo_labels = self._build_rpn_pseudo_labels_from_matched_sparse(
            matched_sparse_boxes=matched_sparse_boxes,
            matched_sparse_scores=matched_sparse_scores,
            matched_labels=matched_labels,
            batch_size=imgs.shape[0],
        )
        # NMS + Top-K Refinement
        pseudo_labels = self._refine_pseudo_labels_nms_topk(
            pseudo_labels,
            iou_thr=self.config.rpn_pseudo_nms_thr,
            topk=self.config.rpn_pseudo_topk
        )

        # -----get proposals-----
        self.rpn.train()
        B, _, Himg, Wimg = imgs.shape
        device = imgs.device

        # build ImageList
        image_sizes = [(Himg, Wimg) for _ in range(B)]
        dummy_images = torch.zeros((B, 3, 1, 1), device=device)  # placeholder only
        img_list = ImageList(tensors=dummy_images, image_sizes=image_sizes)

        # build targets in the format expected by RPN: List[Dict], each requires "boxes"
        rpn_targets: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            boxes = pseudo_labels[b]["boxes"]  # [Nb,4] xyxy in image coords
            if boxes.numel() == 0:
                # keep empty, RPN will treat as no GT for this image
                rpn_targets.append({"boxes": boxes.to(device=device, dtype=torch.float32)})
            else:
                # safety: ensure float32 + valid boxes
                boxes = boxes.to(device=device, dtype=torch.float32)
                # clamp to image bounds
                boxes[:, 0].clamp_(0, Wimg - 1)
                boxes[:, 2].clamp_(0, Wimg - 1)
                boxes[:, 1].clamp_(0, Himg - 1)
                boxes[:, 3].clamp_(0, Himg - 1)
                # ensure x1<=x2, y1<=y2
                x1 = torch.min(boxes[:, 0], boxes[:, 2])
                x2 = torch.max(boxes[:, 0], boxes[:, 2])
                y1 = torch.min(boxes[:, 1], boxes[:, 3])
                y2 = torch.max(boxes[:, 1], boxes[:, 3])
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                rpn_targets.append({"boxes": boxes})

        # build features dict for RPN
        features = {"0": high_feature_maps}
        # get proposals
        proposals, rpn_losses_dict = self.rpn(img_list, features, rpn_targets)

        # -----R-CNN Head-----
        batch_size = imgs.shape[0]
        det_targets = [
            {
                "boxes": pseudo_labels[b]["boxes"],
                "labels": pseudo_labels[b]["labels"],
            }
            for b in range(batch_size)
        ]
        # detections : List[Dict[str, torch.Tensor]], len=B, each Dict has
        #              {
        #                   "boxes" : [N_boxes, 4],
        #                   "labels" : [N_boxes](1...C), 0 reserved for background
        #                   "scores" : [N_boxes], softmax confidence
        #              }
        detections, det_losses_dict = self.roi_heads(
            features={"0": high_feature_maps},
            proposals=proposals,
            image_shapes=image_sizes,
            targets=det_targets,
        )

        pseudo_labels_mAP = self._evaluate_pseudo_labels(pseudo_labels, gt_targets)

        return loss_ccam, loss_proto, match_losses_dict, rpn_losses_dict, det_losses_dict, pseudo_labels_mAP, detections

    def _forward_inference(
        self,
        imgs: torch.Tensor,  # input images, [B, C, H, W]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        :return:
        - detections: List[Dict[str, torch.Tensor]], length B, each element:
                            {
                                "boxes": Tensor [Nd, 4] in image coords (xyxy),
                                "labels": Tensor [Nd] in [1..C] (0 reserved for background),
                                "scores": Tensor [Nd],
                            }
        """
        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]

        # -----RPN-----
        self.rpn.eval()
        B, _, Himg, Wimg = imgs.shape
        device = imgs.device
        # build ImageList
        image_sizes = [(Himg, Wimg) for _ in range(B)]
        dummy_images = torch.zeros((B, 3, 1, 1), device=device)
        img_list = ImageList(tensors=dummy_images, image_sizes=image_sizes)
        # build features dict for RPN
        features = {"0": high_feature_maps}
        # get proposals
        proposals, _ = self.rpn(img_list, features, None)

        # -----R-CNN Head-----
        self.roi_heads.eval()
        detections, _ = self.roi_heads(
            features={"0": high_feature_maps},
            proposals=proposals,
            image_shapes=image_sizes,
            targets=None,
        )

        return detections

    def forward(
        self,
        mode: str,  # "train" or "inference"
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor = None,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels : torch.Tensor = None,     # class label for weak boxes, [R, num_classes]
        gt_targets : List[Dict[str, torch.Tensor]] = None   # ground-truth targets for evaluate pseudo labels
    ):
        if mode == "train":
            return self._forward_train(imgs, wboxes, wb_labels, gt_targets)
        elif mode == "inference":
            return self._forward_inference(imgs)

# CCAM verification
class Stage2_ccam(nn.Module):
    def __init__(
        self,
        config : Stage2CCAMConfig,       # Stage2 CCAM configuration
        backbone : nn.Module,  # Default VGG-16 with aligned weights
    )-> None:
        super(Stage2_ccam, self).__init__()

        self.encoder = backbone
        if config.freeze_backbone:     # freeze backbone weights
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.config = config

        # (train)RoI Align for high-level feature maps to weak box feature maps
        self.roi_align_h2wb = RoIAlign(
            output_size=self.config.roi_out_size_h2wb,
            spatial_scale=self.config.spatial_scale_h2wb,
            sampling_ratio=self.config.sampling_ratio,
            aligned=self.config.aligned,
        )

        # CCAM Generator
        self.ccam_generator = CCAMGenerator(in_c=self.config.in_c)

    def _get_multi_bboxes(
        self,
        cam : np.ndarray,     # [h, w, 1]
        cam_thr : float = 0.2,        # threshold, [0, 1]
        area_ratio : float =0.5,
        do_aug: bool = True,
        delta_aug : float = 0.10,  # jitter strength, e.g. 0.05~0.15
        num_aug: int = 4,  # how many jittered boxes per base box
        min_box_size: int = 2  # avoid degenerate tiny boxes
    )-> List[List[int]]:
        """
        Copy from : https://github.com/MingXiangL/SPE
        :return: estimated bounding box: len(contours), each is [x1, y1, x2, y2]
        """
        # with open("debug_data.txt", "a", encoding="utf-8") as f:
        #     f.write(f"CCAM: (shape: {cam.shape})\n")
        #     for line in cam:
        #         f.write(" ".join([f"{val:.4f}" for val in line.flatten().tolist()]) + "\n")
        assert cam.ndim == 3 and cam.shape[2] == 1, f"Expect [H,W,1], got {cam.shape}"
        Hc, Wc, _ = cam.shape

        # ---- (1) thresholding + find contours ----
        cam_u8 = (cam * 255.).astype(np.uint8)
        map_thr = cam_thr * float(np.max(cam_u8))

        _, thr_gray_heatmap = cv2.threshold(
            cam_u8, int(map_thr), 255, cv2.THRESH_TOZERO
        )

        contours, _ = cv2.findContours(
            thr_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) != 0:
            estimated_bbox: List[List[int]] = []
            areas = list(map(cv2.contourArea, contours))
            area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)

            for idx in area_idx:
                if areas[idx] >= areas[area_idx[0]] * area_ratio:
                    c = contours[idx]
                    x, y, w, h = cv2.boundingRect(c)
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    # clamp + sanitize
                    x1 = max(0, min(x1, Wc - 1))
                    y1 = max(0, min(y1, Hc - 1))
                    x2 = max(0, min(x2, Wc - 1))
                    y2 = max(0, min(y2, Hc - 1))
                    if x2 < x1: x1, x2 = x2, x1
                    if y2 < y1: y1, y2 = y2, y1

                    # filter too small
                    if (x2 - x1) >= min_box_size and (y2 - y1) >= min_box_size:
                        estimated_bbox.append([x1, y1, x2, y2])
        else:
            estimated_bbox = [[0, 0, 1, 1]]

        # ---- (2) Seed Proposal Augmentation (box jittering in CCAM coords) ----
        # REF_11 idea: for each seed box, generate multiple jittered boxes in its neighborhood.
        if (not do_aug) or (len(estimated_bbox) == 0):
            return estimated_bbox

        # if it's the default dummy box, do not augment
        if len(estimated_bbox) == 1 and estimated_bbox[0] == [0, 0, 1, 1]:
            return estimated_bbox

        aug_boxes: List[List[int]] = []
        # deterministic randomness if you want reproducibility:
        # rng = np.random.RandomState(0)
        rng = np.random

        for (x1, y1, x2, y2) in estimated_bbox:
            w = max(float(x2 - x1), 1.0)
            h = max(float(y2 - y1), 1.0)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            for _ in range(int(num_aug)):
                # eps ~ U(-delta_aug, +delta_aug)
                ex = float(rng.uniform(-delta_aug, delta_aug))
                ey = float(rng.uniform(-delta_aug, delta_aug))
                ew = float(rng.uniform(-delta_aug, delta_aug))
                eh = float(rng.uniform(-delta_aug, delta_aug))

                # jitter in neighborhood (relative)
                cx2 = cx * (1.0 + ex)
                cy2 = cy * (1.0 + ey)
                w2 = w * (1.0 + ew)
                h2 = h * (1.0 + eh)

                # keep positive size
                w2 = max(w2, float(min_box_size))
                h2 = max(h2, float(min_box_size))

                nx1 = cx2 - 0.5 * w2
                ny1 = cy2 - 0.5 * h2
                nx2 = cx2 + 0.5 * w2
                ny2 = cy2 + 0.5 * h2

                # clamp to CCAM bounds
                nx1 = max(0.0, min(nx1, Wc - 1.0))
                ny1 = max(0.0, min(ny1, Hc - 1.0))
                nx2 = max(0.0, min(nx2, Wc - 1.0))
                ny2 = max(0.0, min(ny2, Hc - 1.0))

                # sanitize ordering
                if nx2 < nx1: nx1, nx2 = nx2, nx1
                if ny2 < ny1: ny1, ny2 = ny2, ny1

                # filter too small after clamp
                if (nx2 - nx1) < min_box_size or (ny2 - ny1) < min_box_size:
                    continue

                aug_boxes.append([int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))])

        # Optional: remove duplicates (common after rounding)
        if len(aug_boxes) > 0:
            uniq = []
            seen = set()
            for b in aug_boxes:
                t = tuple(b)
                if t not in seen:
                    seen.add(t)
                    uniq.append(b)
            aug_boxes = uniq

        return estimated_bbox + aug_boxes

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

    def _ccam_box_score(
            self,
            ccam_hw1: np.ndarray,  # (Hc, Wc, 1) float in [0,1] (or any float)
            box_xyxy: List[int],  # [x1, y1, x2, y2] in CCAM coords
            mode: str = "topk_mean",  # "max" | "mean" | "topk_mean"
            topk_ratio: float = 0.1,  # top 10% pixels
            topk_min: int = 20,  # at least 20 pixels
    ) -> float:
        """
        Compute a confidence score for a seed box based on CCAM activation inside the box.
        Return a python float (recommended in [0,1] if ccam is normalized).
        """
        assert ccam_hw1.ndim == 3 and ccam_hw1.shape[2] == 1, f"Expect (H,W,1), got {ccam_hw1.shape}"

        Hc, Wc, _ = ccam_hw1.shape
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]

        # clamp to valid range
        x1 = max(0, min(x1, Wc - 1))
        y1 = max(0, min(y1, Hc - 1))
        x2 = max(0, min(x2, Wc - 1))
        y2 = max(0, min(y2, Hc - 1))

        # ensure x2>=x1, y2>=y1
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        # slicing: use inclusive->exclusive
        patch = ccam_hw1[y1:y2 + 1, x1:x2 + 1, 0]  # (h, w)
        if patch.size == 0:
            return 0.0

        v = patch.reshape(-1).astype(np.float32)

        if mode == "max":
            score = float(v.max())
        elif mode == "mean":
            score = float(v.mean())
        elif mode == "topk_mean":
            k = int(max(topk_min, round(topk_ratio * v.size)))
            k = min(k, v.size)
            # top-k mean
            # np.partition is O(n)
            topk = np.partition(v, -k)[-k:]
            score = float(topk.mean())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # optional: clamp to [0,1] if your CCAM is normalized
        if np.isfinite(score):
            score = max(0.0, min(1.0, score))
        else:
            score = 0.0
        return score

    def _get_mAP(self,
        seed_proposals: List[Dict[str, torch.Tensor]],  # seed proposals
        gt_targets: List[Dict[str, torch.Tensor]]  # ground-truth targets for evaluate pseudo labels
    )-> float:
        """
        Evaluate pseudo labels using ground-truth targets.
        """
        # Copy from torchvision references
        map_metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=torch.arange(0.5, 0.96, 0.05).tolist(),  # 0.50 : 0.05 : 0.95
            max_detection_thresholds=[1, 10, 100],
        ).to(self.config.device)

        map_metric.update(seed_proposals, gt_targets)
        result = map_metric.compute()

        return float(result["map"])

    def forward(
        self,
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor = None,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor = None,  # class label for weak boxes, [R, num_classes]
        gt_targets: List[Dict[str, torch.Tensor]] = None,  # ground-truth targets for evaluate pseudo labels
        vis_dir : str = None,        # visualization directory
        should_invert : bool = True     # whether to invert CCAM
    )-> Tuple[torch.Tensor, float]:
        """
        :return:
        - loss_ccam: CCAM loss
        - seed_mAP: mAP of seed boxes
        """
        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]

        # -----get weak box feature maps-----
        wb_feature_maps = self.roi_align_h2wb(high_feature_maps, wboxes)  # [R, C_h, H_wb, W_wb]

        # get CCAM and loss_ccam
        ccam, loss_ccam = self.ccam_generator(wb_feature_maps)  # [R, 1, H_wb, W_wb]
        if should_invert:
            ccam = 1 - ccam

        # get seed proposals from CCAM
        seed_proposals: List[Dict] = []
        R, _, _, _ = ccam.shape
        for i in range(R):
            ccam_i = ccam[i].detach().cpu().numpy().transpose(1, 2, 0)  # [h, w, 1]

            local_boxes_list = self._get_multi_bboxes(ccam_i, self.config.ccam_threshold)
            # 过滤“无 contours”的默认返回框
            if len(local_boxes_list) == 1 and local_boxes_list[0] == [0, 0, 1, 1]:
                continue
            # map local boxes to image boxes
            wbox_i = wboxes[i]  # [5]
            Himg, Wimg = imgs.shape[2], imgs.shape[3]
            global_boxes_xyxy = self._map_boxes_roi_to_image_xyxy(
                boxes_local=local_boxes_list,
                wbox=wbox_i,
                cam_hw=(ccam_i.shape[0], ccam_i.shape[1]),
                img_hw=(Himg, Wimg)
            )  # Tensor [N, 4]
            batch_idx = wbox_i[0].to(global_boxes_xyxy).view(1, 1).repeat(global_boxes_xyxy.size(0), 1)
            global_boxes_wbatch = torch.cat([batch_idx, global_boxes_xyxy], dim=1)  # [N, 5]

            # class_id = wb_labels[i].argmax().item()
            # for j in range(global_boxes_wbatch.size(0)):
            #     seed_proposals.append({
            #         "box": global_boxes_wbatch[j],  # Tensor [5], (batch_idx, x1, y1, x2, y2)
            #         "class_id": class_id,   # [0, C-1]
            #     })

            class_id = wb_labels[i].argmax().item()
            # local_boxes_list: CCAM coords boxes (same order as mapping)
            # global_boxes_wbatch: mapped image coords boxes with batch idx
            for j in range(global_boxes_wbatch.size(0)):
                local_box = local_boxes_list[j]  # [x1,y1,x2,y2] in CCAM coords

                # score from CCAM activation inside local box
                score = self._ccam_box_score(
                    ccam_hw1=ccam_i,  # (Hc,Wc,1)
                    box_xyxy=local_box,
                    mode="topk_mean",  # or "max"
                    topk_ratio=0.1,
                    topk_min=20,
                )

                seed_proposals.append({
                    "box": global_boxes_wbatch[j],  # Tensor [5] (batch_idx,x1,y1,x2,y2)
                    "class_id": class_id,
                    "score": float(score),  # python float
                })

        # for debug: Visualize CCAM and decide whether to invert
        # imgs: torch.Tensor [B,C,H,W] (一般是RGB归一化)
        R, _, _, _ = ccam.shape
        for i in range(R):
            ccam_i = ccam[i].detach().cpu().numpy().transpose(1, 2, 0)  # [h, w, 1]

            b = int(wboxes[i, 0].item())
            x1, y1, x2, y2 = wboxes[i, 1:5].detach().cpu().numpy().tolist()
            img_chw = imgs[b].detach().cpu().numpy()
            img_chw = denorm_imagenet(img_chw)  # -> [0,1]
            img_np = np.transpose(img_chw, (1, 2, 0))  # HWC
            img_u8 = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

            gt_boxes_b = None
            gt_labels_b = None
            if gt_targets is not None:
                gt_boxes_b = gt_targets[b]["boxes"]  # Tensor [Ng,4]
                gt_labels_b = gt_targets[b].get("labels")

            seed_boxes_b = [
                p for p in seed_proposals
                if int(p["box"][0].item()) == b
            ]

            vis_img = overlay_ccam_on_image(
                img_bgr=img_bgr,
                ccam_hw1=ccam_i,
                wbox_xyxy=(x1, y1, x2, y2),
                alpha=0.45,
                gt_boxes=gt_boxes_b,
                gt_labels=gt_labels_b,
                seed_boxes=seed_boxes_b,
                draw_wbox=True,
            )
            ccam_vis_path = os.path.join(vis_dir, f"debug_ccam_overlay_b{b}_r{i}.png")
            cv2.imwrite(ccam_vis_path, vis_img)


        B = imgs.shape[0]
        # build seed boxes for mAP evaluation
        seed_boxes_by_img: List[Dict[str, torch.Tensor]] = []
        # for b in range(B):
        #     boxes_list = []
        #     labels_list = []
        #     for p in seed_proposals:
        #         box = p["box"]
        #         batch_idx = int(box[0].item())
        #         if batch_idx == b:
        #             boxes_list.append(box[1:5].unsqueeze(0))
        #             labels_list.append(p["class_id"])
        #
        #     if len(boxes_list) > 0:
        #         boxes_tensor = torch.cat(boxes_list, dim=0)
        #         labels_tensor = torch.tensor(labels_list, device=boxes_tensor.device, dtype=torch.long)
        #     else:
        #         boxes_tensor = torch.zeros((0, 4), device=imgs.device, dtype=torch.float32)
        #         labels_tensor = torch.zeros((0,), device=imgs.device, dtype=torch.long)
        #
        #     seed_boxes_by_img.append({
        #         "boxes": boxes_tensor,    # [N_boxes, 4]
        #         "labels": labels_tensor,  # [N_boxes] in [0..C-1]
        #     })
        for b in range(B):
            boxes_list = []
            labels_list = []
            scores_list = []

            for p in seed_proposals:
                box = p["box"]
                batch_idx = int(box[0].item())
                if batch_idx == b:
                    boxes_list.append(box[1:5].unsqueeze(0))
                    labels_list.append(p["class_id"])
                    scores_list.append(float(p.get("score", 1.0)))  # fallback

            if len(boxes_list) > 0:
                boxes_tensor = torch.cat(boxes_list, dim=0)
                labels_tensor = torch.tensor(labels_list, device=boxes_tensor.device, dtype=torch.long)
                scores_tensor = torch.tensor(scores_list, device=boxes_tensor.device, dtype=torch.float32)
            else:
                boxes_tensor = torch.zeros((0, 4), device=imgs.device, dtype=torch.float32)
                labels_tensor = torch.zeros((0,), device=imgs.device, dtype=torch.long)
                scores_tensor = torch.zeros((0,), device=imgs.device, dtype=torch.float32)

            seed_boxes_by_img.append({
                "boxes": boxes_tensor,  # [N,4]
                "labels": labels_tensor,  # [N]
                "scores": scores_tensor,  # [N]
            })

        seed_mAP = self._get_mAP(seed_boxes_by_img, gt_targets)

        return loss_ccam, seed_mAP



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

        # statistics：sum and cnt
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

def build_Stage2CCAM_model(
    stage2CCAM_config : Stage2CCAMConfig,
    backbone_weights_path: str,  # Aligned backbone weights path
)-> nn.Module:
    backbone = vgg16(pretrained=False).features
    backbone.load_state_dict(torch.load(backbone_weights_path))
    # backbone = vgg16(pretrained=True).features

    model = Stage2_ccam(
        backbone=backbone,
        config=stage2CCAM_config
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