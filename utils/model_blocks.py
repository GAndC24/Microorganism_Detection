'''
ref. from https://github.com/CVI-SZU/CCAM
ref. from https://github.com/zxhuang1698/interpretability-by-parts
ref. from https://github.com/Sierkinhane/ORNet
ref. from https://github.com/facebookresearch/detr
ref. from https://github.com/fundamentalvision/Deformable-DETR
modified by
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, List, Dict
from torchvision.models import vgg16
from torchvision.ops import RoIAlign
import fvcore.nn.weight_init as weight_init
from timm.models.vision_transformer import PatchEmbed
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import SimMaxLoss, SimMinLoss
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area

# Feature Hook to extract Multi-level features of backbone
class FeatureHook:
    def __init__(self)-> None:
        self.outputs = OrderedDict()
        self.handles = []

    def _hook(self, name : str)-> callable:
        """
        :param name: feature name, such as 'low, mid, high'
        :return: function fn
        """
        def fn(module : nn.Module, inp : Tuple, out : torch.Tensor)-> None:
            """
            :param module: 当前层，如 nn.relu
            :param inp: 当前层的输入
            :param out: 当前层的输出
            """
            self.outputs[name] = out
        return fn

    def register(self, module: nn.Module, name: str)-> None:
        """
        :param module: 当前层，如 nn.relu
        :param name:  feature name, such as 'low, mid, high'
        """
        # 当 module.forward() 执行完毕后，自动调用 hook 作为附加监听器
        handle = module.register_forward_hook(self._hook(name))
        self.handles.append(handle)

    def clear(self):
        """
        每次 forward 后，清空 hook
        """
        self.outputs.clear()

    def remove(self):
        """
        训练结束后，移除 hook
        """
        for h in self.handles:
            h.remove()
        self.handles = []

# Class Activation Map Head, generate CAMs and compute CAM loss
class CAMHead(nn.Module):
    def __init__(
        self,
        num_classes: int, # number of classes
        in_channels: int  # input channels
    )-> None:
        super(CAMHead, self).__init__()

        self.num_classes = num_classes

        self.CE_loss = nn.CrossEntropyLoss()
        self.cam_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

        weight_init.c2_msra_fill(self.cam_conv)

    def forward(self, x : torch.Tensor, y : torch.Tensor)-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param x: RoI feature maps, [R, C, H, W]
        :param y: weak box labels, [R, num_classes]
        :return:
        - CAMs, [R, num_classes, H, W]
        - CAM loss, {loss_name: loss}
        """
        # get CAMs
        x = self.cam_conv(x)

        # get class logits
        logits = F.avg_pool2d(x, (x.size(2), x.size(3)))
        logits = logits.view(-1, self.num_classes)

        # compute CE Loss
        target = torch.argmax(y, dim=1)
        loss_cam = self.CE_loss(logits, target)

        return x, loss_cam

# Morphological Prototype Generator
class MorphologicalPrototypeGenerator(nn.Module):
    def __init__(
        self,
        num_classes : int,  # number of classes
        in_c : int,        # input channels
        # Patch Embed parameters
        patch_size : int,       # patch size
        embed_dim: int,  # embedding dimension
        # GMM parameters
        components_range : List,   # list of number of components for GMM
        random_state : int,     # random state for GMM(seed)
        max_iter : int,     # max iteration for EM
        # RoI Align parameters
        roi_out_size: Tuple[int, int],  # output size
        spatial_scale : float = 1/16,  # spatial scale
        sampling_ratio : int = 2,       # sampling ratio
        aligned : bool = True,            # aligned flag
    )->None:
        super(MorphologicalPrototypeGenerator, self).__init__()

        self.num_classes = num_classes
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.num_prototypes = num_classes
        self.patch_size = patch_size
        self.components_range = components_range
        self.random_state = random_state
        self.max_iter = max_iter

        self.roi_align = RoIAlign(
            output_size=roi_out_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )

        self.cam_head = CAMHead(
            num_classes=num_classes,
            in_channels=in_c,
        )

        self.patch_embed = PatchEmbed(
            img_size=roi_out_size,
            patch_size=patch_size,
            in_chans=in_c,
            embed_dim=embed_dim,
        )

        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.cls_head.apply(self._init_weights)

    def _init_weights(self, m)->None:
        """
        Initialize weights for Linear and BatchNorm layers.
        :param m: Module to initialize
        """
        if isinstance(m, nn.Linear):  # Check if the module is a Linear layer
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
            if m.bias is not None:  # Initialize bias to zero if it exists
                nn.init.constant_(m.bias, 0)

    def _cam_to_patch_fg_bg_scores(self, cams : torch.Tensor,wb_labels : torch.Tensor)-> torch.Tensor:
        """
        :param cams: CAMs, [R, K = num_classes, H, W]
        :param wb_labels: weak box labels, [R, K = num_classes]
        :return: patch_fg_bg_scores : fg/bg scores for each patch, [R, Np = num_patches, 2]
        """

        R, K, H, W = cams.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "RoI output size must be divisible by patch_size"

        # get class ids for each weak box
        cls_ids = torch.argmax(wb_labels, dim=1)  # [R]

        # gather class-specific CAM: [R, H, W]
        cam_cls = cams[torch.arange(R, device=cams.device), cls_ids]  # [R, H, W]

        # get probability maps from CAMs
        cam_prob = torch.sigmoid(cam_cls).unsqueeze(1)  # [R, 1, H, W]

        fg_map = F.avg_pool2d(
            cam_prob,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )       # [R, 1, Hp, Wp]
        fg_score = fg_map.flatten(1)        # [R, N]
        bg_score = 1 - fg_score     # [R, N]

        patch_fg_bg_scores = torch.stack([fg_score, bg_score], dim=-1)  # [R, Np, 2]

        return patch_fg_bg_scores

    def _get_anchor_features(self, patch_features : torch.Tensor, patch_fg_bg_scores : torch.Tensor)->torch.Tensor:
        """
        :param patch_features: for each weak box, [Np, D]
        :param patch_fg_bg_scores: for each weak box, [Np, 2]
        :return: anchor_feature: the anchor feature of weak box, [D]
        """
        x = patch_features.cpu().detach().numpy()

        # Norm x
        x = x / np.linalg.norm(x, axis=1, keepdims=True)

        # Use BIC to select number of components
        bic_scores : List = []
        for k in self.components_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                init_params='kmeans',  # K-means 初始化
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            gmm.fit(x)
            bic_scores.append(gmm.bic(x))

        best_k = self.components_range[bic_scores.index(min(bic_scores))]

        # GMM clustering
        gmm = GaussianMixture(
            n_components=best_k,
            covariance_type='full',
            init_params='kmeans',
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        gmm.fit(x)

        # get responsibility of each component
        responsibilities = gmm.predict_proba(x)     # [Np, best_k]

        # get fg scores for each component
        obj_scores = patch_fg_bg_scores[:, 0].cpu().detach().numpy()  # [Np, ]
        obj_scores = obj_scores.reshape(-1, 1)  # [Np, 1]
        fg_scores = (responsibilities * obj_scores).sum(axis=0)

        # select the component with highest fg score as anchor
        idx = np.argmax(fg_scores)
        anchor_feature = gmm.means_[idx]  # [D]

        return torch.Tensor(anchor_feature).to(patch_features.device)

    def _get_morphological_prototypes(
        self,
        patch_features : torch.Tensor,      # [R, Np, D]
        weights : torch.Tensor,    # [R, Np]
        wb_labels : torch.Tensor,       # [R, num_classes]
        eps: float = 1e-6,
        normalize_proto: bool = True
    )-> Dict[int, torch.Tensor]:
        """
        :return: prototypes, {class_id: prototype tensor}
        """
        # 扩维以便广播：
        y_ = wb_labels[:, :, None, None]  # [R, num_classes, 1, 1]
        w_ = weights[:, None, :, None]  # [R, 1, Np, 1]
        f_ = patch_features[:, None, :, :]  # [R, 1, Np, D]

        numerator = (y_ * w_ * f_).sum(dim=(0, 2))  # sum over r and p => [num_classes, D]
        denom = (y_ * w_).sum(dim=(0, 2))  # [num_classes, 1]

        prototypes = numerator / (denom + eps)  # [num_classes, D]
        if normalize_proto:
            prototypes = F.normalize(prototypes, p=2, dim=-1)

        proto_dict : Dict[int, torch.Tensor] = {}
        for class_id in range(self.num_classes):
            proto_dict[class_id] = prototypes[class_id]

        return proto_dict


    def forward(
        self,
        x : torch.Tensor,       # middle feature maps, [B, C, H, W]
        wboxes : torch.Tensor,    # weak boxes, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels : torch.Tensor,    # class label for weak boxes, [R, num_classes]
        lse_alpha : float = 10.0    # LSE alpha
    )-> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        :return:
        - CAM loss
        - prototypes, {class_id: prototype tensor}
        - patch logits, [R * Np, D]
        - patch features for SupCon, [R, view = 1, D]
        """
        # RoI Align to get weak box features
        roi_features = self.roi_align(x, wboxes)       # [R = num_wboxes, C, H, W]

        # get CAMs
        cams, loss_cam = self.cam_head(roi_features, wb_labels)        # cams, [R, num_classes, H, W]

        # patch embedding
        roi_features_detached = roi_features.detach()
        patch_features = self.patch_embed(roi_features_detached)     # [R, Np = num_patches, D]
        R, Np, D = patch_features.shape

        # get fg/bg scores for each patch
        patch_fg_bg_scores = self._cam_to_patch_fg_bg_scores(cams, wb_labels)      # [R, Np, 2], fg_scores = [:, :, 0], bg_scores = [:, :, 1]

        # select top-k patches based on fg scores
        top_k_ratio = 0.25      # [0.2, 0.3]
        k = int(max(1, Np * top_k_ratio))
        topk_fg_scores, topk_indices = torch.topk(patch_fg_bg_scores[:, :, 0], k=k, dim=1)    # [R, k]
        patch_indices = torch.arange(R, device=x.device).unsqueeze(1).expand(-1, k)    # [R, k]
        topk_patch_features = patch_features[patch_indices, topk_indices]    # [R, k, D]

        # get patch logits for classification
        patch_logits = self.cls_head(topk_patch_features.reshape(R * k, D))      # [R * k, num_classes]

        # LogSumExp for topk_patch_features
        x_lse = lse_alpha * topk_patch_features  # [R, k, D]
        contrast_patch_features = torch.logsumexp(x_lse, dim=1) / lse_alpha   # [R, D]
        contrast_patch_features = F.normalize(contrast_patch_features, p=2, dim=-1)  # [R, D]
        contrast_patch_features = contrast_patch_features.view(R, 1, D)     # for SupCon format, [R, view = 1, D]

        # get anchor feature of each weak box
        anchor_features_list : List[torch.Tensor] = []
        for i in range(R):
            patch_feature = patch_features[i]
            patch_fg_bg_score = patch_fg_bg_scores[i]
            anchor_feature = self._get_anchor_features(patch_feature, patch_fg_bg_score)
            anchor_features_list.append(anchor_feature)
        anchor_features = torch.stack(anchor_features_list, dim=0)      # [R, D]

        # compute patch weights(similarity)
        patch_features = F.normalize(patch_features, p=2, dim=-1)
        anchor_features = F.normalize(anchor_features, p=2, dim=-1)
        weights = (patch_features * anchor_features.unsqueeze(1)).sum(dim=-1)  # [R, Np]
        weights = F.relu(weights)  # ReLU, remove negative value
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # normalize to sum to 1

        # get morphological prototypes
        prototypes = self._get_morphological_prototypes(patch_features, weights, wb_labels)        # {class_id : prototype tensor}

        return loss_cam, prototypes, patch_logits, contrast_patch_features

# CCAM Generator, generate CCAMs and compute CCAM loss
class CCAMGenerator(nn.Module):
    def __init__(
        self,
        in_c : int,     # input channels
        alpha : float = 0.05
    )-> None:
        super(CCAMGenerator, self).__init__()

        self.activation_head = nn.Conv2d(in_c, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.criterion = [
            SimMaxLoss(metric='cos', alpha=alpha), # BG-BG positive contrast
            SimMinLoss(metric='cos'),   # BG-FG negative contrast
            SimMaxLoss(metric='cos', alpha=alpha)   # FG-FG positive contrast
        ]

    def forward(self, x : torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input feature maps, [N, C, H, W]
        :return:
        - ccam: class activation map, [N, 1, H, W]
        - loss_ccam: CCAM loss
        """
        N, C, H, W = x.size()

        ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]

        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        fg_feats = fg_feats.reshape(x.size(0), -1)      # [N, C]
        bg_feats = bg_feats.reshape(x.size(0), -1)      # [N, C]
        for loss in self.criterion:
            loss.to(x.device)

        loss_bg_bg = self.criterion[0](bg_feats)
        loss_bg_fg = self.criterion[1](bg_feats, fg_feats)
        loss_fg_fg = self.criterion[2](fg_feats)
        loss_ccam = loss_bg_bg + loss_bg_fg + loss_fg_fg

        return ccam, loss_ccam

# One-to-One matcher

def _box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def _box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def _generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if boxes1.shape[1] < 4:
        print(f'boxes1.shape:{boxes1.shape}')
    if boxes2.shape[1] < 4:
        print(f'boxes2.shape[1]:{boxes2.shape[1]}')
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), boxes1
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), boxes2
    iou, union = _box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, match_ratio: int = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.match_ratio = match_ratio
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -_generalized_box_iou(_box_cxcywh_to_xyxy(out_bbox), _box_cxcywh_to_xyxy(tgt_bbox))
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        with open("./debug_data.txt", "a", encoding="utf-8") as f:
            f.write(f"Cost matrix shape: {C.shape}\n")
            f.write(f"Cost matrix max value: {C.max()}\n")

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# Build backbone hook
def build_backbone_hook(backbone : nn.Module, indices : List[int]) -> FeatureHook:
    """
    :param backbone: Default VGG-16 backbone
    :param indices: feature layer indices
    :return: FeatureHook object
    """
    hook = FeatureHook()
    for idx, tag in zip(indices, ['low', 'mid', 'high']):
        module = backbone[idx]
        hook.register(module, tag)
    return hook

# Build VGG-16 backbone with hook
def build_vgg16_backbone_with_hook(indices : List[int]) -> Tuple[nn.Module, FeatureHook]:
    """
    :return: VGG-16 backbone and FeatureHook object
    """
    # Load default VGG-16 backbone
    backbone = vgg16(pretrained=True).features
    # Build backbone hooker
    hook = build_backbone_hook(backbone, indices)
    return backbone, hook