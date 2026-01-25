# Morphological Prototype R-CNN
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from utils import random_masking, add_gaussian_noise, MorphologicalPrototypeGenerator, FeatureHook, build_vgg16_backbone_with_hook, vgg_layer_out_c_maps, CCAMGenerator
from torchvision.ops import RoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models import vgg16
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
import numpy as np
import cv2

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
    img_size: Tuple[int, int]  # input image size
    num_classes: int  # number of classes
    in_c : int  # high-level feature maps channels
    dataset_mps : Dict[int, torch.Tensor]  # {class_id : prototype tensor}
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
    # for RoI Align
    roi_out_size_h2wb: Tuple[int, int]  # output size for high feature maps to weak box features
    spatial_scale_h2wb: float  # spatial scale for high feature maps to weak box features
    roi_out_size_wb2p: Tuple[int, int]  # output size for weak box features to proposal box features
    spatial_scale_wb2p: float  # spatial scale
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
    )-> None:
        super(Stage2, self).__init__()

        self.encoder = backbone
        # # freeze backbone weights
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.config = config

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
        # (inference) RoI Align for high-level feature maps to proposal box feature maps
        self.roi_align_h2p = RoIAlign(
            output_size=self.config.roi_out_size_h2p,
            spatial_scale=self.config.spatial_scale_h2p,
            sampling_ratio=self.config.sampling_ratio,
            aligned=self.config.aligned,
        )

        # RPN
        anchor_generator = AnchorGenerator(
            sizes=self.config.rpn_anchor_sizes,
            aspect_ratios=self.config.rpn_anchor_aspect_ratios
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
        cam = (cam * 255.).astype(np.uint8)
        map_thr = cam_thr * np.max(cam)

        _, thr_gray_heatmap = cv2.threshold(cam,
                                            int(map_thr), 255,
                                            cv2.THRESH_TOZERO)
        # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

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

    def _forward_train(
        self,
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor  # class label for weak boxes, [R, num_classes]
    ):
        # -----get high-level feature maps-----
        self.encoder[-1] = nn.Identity()
        high_feature_maps = self.encoder(imgs)  # [B, C_h, H_h, W_h]

        # -----get weak box feature maps-----
        wb_feature_maps = self.roi_align_h2wb(high_feature_maps, wboxes)  # [R, C_h, H_wb, W_wb]

        # -----Branch 1: get sparse proposals-----
        # get dense proposals
        dense_proposals_list = self._get_dense_proposals(wb_feature_maps, wboxes)  # List[Tensor], len=R, each Tensor is [num_proposals, 4]
        wbox_batch_idx = wboxes[:, 0].to(device=wb_feature_maps.device)  # [R]
        dense_proposal_boxes_list = []
        for i, props_xyxy in enumerate(dense_proposals_list):
            # props_xyxy: [num_proposals, 4]
            bi = wbox_batch_idx[i].expand(props_xyxy.shape[0], 1).to(dtype=props_xyxy.dtype)  # [num_proposals, 1]
            props_5 = torch.cat([bi, props_xyxy], dim=1)  # [num_proposals, 5]
            dense_proposal_boxes_list.append(props_5)
        dense_proposal_boxes = torch.cat(dense_proposal_boxes_list, dim=0)  # [N_D = num_proposals, 5]

        # get proposal features
        dense_proposal_features = self.roi_align_wb2p(wb_feature_maps, dense_proposals_list)    # [N_D = total_num_proposals, C_h, H_p, W_p]

        # Object Discovery
        # Dense proposal features to embeddings
        dense_proposal_features = self.gap(dense_proposal_features)  # [N_D, C_h, 1, 1]
        dense_proposal_features = dense_proposal_features.view(dense_proposal_features.shape[0], -1)    # [N_D, C_h]
        dense_proposal_embeddings = self.proj(dense_proposal_features)  # [N_D, D = embed_dim]

        # get object scores
        dense_proposal_object_scores = self.obj_classifier(dense_proposal_embeddings)  # [N_D, num_classes + 1]
        dense_proposal_object_scores = F.softmax(dense_proposal_object_scores, dim=1)  # [N_D, num_classes + 1]

        # get prototypes embeddings
        prototypes = torch.stack([self.config.dataset_mps[k] for k in range(self.config.num_classes)], dim=0)  # [num_classes, D_p = C_h]
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
        sparse_mask = sims_all > thresholds.unsqueeze(0)  # [N_D, num_classes]
        sparse_proposals : List[Dict] = []
        for c in range(self.config.num_classes):
            indices = torch.where(sparse_mask[:, c])[0]
            for i in indices:
                sparse_proposals.append({
                    "box" : dense_proposal_boxes[i],  # [5], (batch_idx, x1, y1, x2, y2)
                    "class_id" : c,
                    "score": dense_proposal_object_scores[i, c]
                })

        # -----Branch 2: get seed proposals-----
        # get CCAM and loss_ccam
        ccam, loss_ccam = self.ccam_generator(wb_feature_maps)      # [R, 1, H_wb, W_wb]

        # get augmented seed proposals from CCAM
        seed_proposals : List[Dict] = []
        R, _, _, _ = ccam.shape
        for i in range(R):
            ccam_i = ccam[i].detach().cpu().numpy().transpose(1, 2, 0)      # [h, w, 1]
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

        # ----- one-to-one match-----











    def forward(
        self,
        mode: str,  # "train" or "inference"
        imgs: torch.Tensor,  # input images, [B, C, H, W]
        wboxes: torch.Tensor = None,  # weak boxes for training, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels : torch.Tensor = None     # class label for weak boxes, [R, num_classes]
    ):
        if mode == "train":
            return self._forward_train(imgs, wboxes, wb_labels)
        elif mode == "inference":
            return self._forward_inference(imgs)

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

