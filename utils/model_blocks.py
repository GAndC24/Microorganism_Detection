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
from torchvision.models import vgg16
from torchvision.ops import RoIAlign
import fvcore.nn.weight_init as weight_init
from timm.models.vision_transformer import PatchEmbed
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import SimMaxLoss, SimMinLoss
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
from dataclasses import dataclass
from typing import Dict, List, Tuple
from torchvision.ops import nms

# -----Feature Hook-----
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
            :param module: 褰撳墠灞傦紝濡?nn.relu
            :param inp: 褰撳墠灞傜殑杈撳叆
            :param out: 褰撳墠灞傜殑杈撳嚭
            """
            self.outputs[name] = out
        return fn

    def register(self, module: nn.Module, name: str)-> None:
        """
        :param module: 褰撳墠灞傦紝濡?nn.relu
        :param name:  feature name, such as 'low, mid, high'
        """
        # 褰?module.forward() 鎵ц瀹屾瘯鍚庯紝鑷姩璋冪敤 hook 浣滀负闄勫姞鐩戝惉鍣?
        handle = module.register_forward_hook(self._hook(name))
        self.handles.append(handle)

    def clear(self):
        """
        姣忔 forward 鍚庯紝娓呯┖ hook
        """
        self.outputs.clear()

    def remove(self):
        """
        璁粌缁撴潫鍚庯紝绉婚櫎 hook
        """
        for h in self.handles:
            h.remove()
        self.handles = []

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
                init_params='kmeans',  # K-means 鍒濆鍖?
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
        # 鎵╃淮浠ヤ究骞挎挱锛?
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

# -----One-to-One matcher-----
def _box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def _box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        (x1 - x0),
        (y1 - y0),
    ]
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
        # with open("./debug_data.txt", "a", encoding="utf-8") as f:
        #     f.write(f"Cost matrix shape: {C.shape}\n")
        #     f.write(f"Cost matrix max value: {C.max()}\n")

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def match_seed_proposals_with_sam3_targets(
    seed_proposals: List[Dict],
    rpn_targets: List[Dict[str, torch.Tensor]],
    imgs: torch.Tensor = None,
    matcher: HungarianMatcher = None,
    cost_class: float = 1.0,
    cost_bbox: float = 1.0,
    cost_giou: float = 1.0,
    match_focal_alpha: float = 0.25,
    match_focal_gamma: float = 2.0,
    lambda_match_cls: float = 1.0,
    lambda_match_l1: float = 1.0,
    lambda_match_giou: float = 1.0,
    other_class_logit: float = -20.0,
    bg_logit: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Match Stage2 seed proposals with filtered RPN targets and compute match loss.

    Notes:
    - `seed_proposals[i]["class_id"]` is assumed to be in [1..C], consistent with Stage2.
    - `rpn_targets` is a List[Dict] with keys "box", "score", and "class_id".
    - Returned indices are the original indices in `seed_proposals` that were matched successfully.
    """
    from .losses import compute_match_loss

    if imgs is not None:
        device = imgs.device
    elif len(seed_proposals) > 0:
        device = seed_proposals[0]["box"].device
    elif rpn_targets is not None and len(rpn_targets) > 0:
        first_box = rpn_targets[0].get("box", None)
        device = first_box.device if first_box is not None and torch.is_tensor(first_box) else torch.device("cpu")
    else:
        device = torch.device("cpu")

    if rpn_targets is None:
        rpn_targets = []

    batch_size = len(rpn_targets)

    zero = torch.tensor(0.0, device=device)
    empty_loss_dict = {
        "loss_match": zero,
        "loss_match_cls": zero,
        "loss_match_l1": zero,
        "loss_match_giou": zero,
    }

    if batch_size == 0:
        return empty_loss_dict, torch.zeros((0,), device=device, dtype=torch.long)

    if matcher is None:
        matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
        )

    def _build_surrogate_imgs() -> torch.Tensor:
        max_x = torch.tensor(1.0, device=device)
        max_y = torch.tensor(1.0, device=device)

        for proposal in seed_proposals:
            box5 = proposal["box"].to(device=device, dtype=torch.float32)
            max_x = torch.maximum(max_x, box5[3])
            max_y = torch.maximum(max_y, box5[4])

        for target in rpn_targets:
            boxes = target.get("box", None)
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device=device, dtype=torch.float32)
            max_x = torch.maximum(max_x, boxes[:, 2].max())
            max_y = torch.maximum(max_y, boxes[:, 3].max())

        h = max(1, int(torch.ceil(max_y).item()))
        w = max(1, int(torch.ceil(max_x).item()))
        return torch.zeros((batch_size, 1, h, w), device=device, dtype=torch.float32)

    imgs_for_loss = imgs if imgs is not None else _build_surrogate_imgs()

    num_classes = 0
    for proposal in seed_proposals:
        num_classes = max(num_classes, int(proposal["class_id"]))
    for target in rpn_targets:
        class_ids = target.get("class_id", None)
        if class_ids is not None and torch.is_tensor(class_ids) and class_ids.numel() > 0:
            num_classes = max(num_classes, int(class_ids.max().item()))
    num_classes = max(num_classes, 1)

    labeled_targets: List[Dict[str, torch.Tensor]] = []
    for b in range(batch_size):
        target_boxes = rpn_targets[b].get("box", None)
        target_scores = rpn_targets[b].get("score", None)
        target_class_ids = rpn_targets[b].get("class_id", None)

        if target_boxes is None or target_boxes.numel() == 0 or target_class_ids is None or target_class_ids.numel() == 0:
            labeled_targets.append(
                {
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                    "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                    "box5": torch.zeros((0, 5), device=device, dtype=torch.float32),
                }
            )
            continue

        target_boxes = target_boxes.to(device=device, dtype=torch.float32)
        if target_scores is None:
            target_scores = torch.ones((target_boxes.shape[0],), device=device, dtype=torch.float32)
        else:
            target_scores = target_scores.to(device=device, dtype=torch.float32)
        target_class_ids = target_class_ids.to(device=device, dtype=torch.long)

        batch_column = torch.full((target_boxes.shape[0], 1), float(b), device=device, dtype=torch.float32)
        target_box5 = torch.cat([batch_column, target_boxes], dim=1)

        labeled_targets.append(
            {
                "boxes": _box_xyxy_to_cxcywh(target_boxes),
                "labels": target_class_ids - 1,
                "scores": target_scores,
                "box5": target_box5,
            }
        )

    if len(seed_proposals) == 0:
        return empty_loss_dict, torch.zeros((0,), device=device, dtype=torch.long)

    proposals_per_img: List[List[Tuple[int, Dict]]] = [[] for _ in range(batch_size)]
    for proposal_idx, proposal in enumerate(seed_proposals):
        box5 = proposal["box"].to(device=device, dtype=torch.float32)
        b = int(box5[0].item())
        if 0 <= b < batch_size:
            proposals_per_img[b].append((proposal_idx, proposal))

    total_queries = sum(len(v) for v in proposals_per_img)
    total_targets = sum(t["boxes"].shape[0] for t in labeled_targets)
    if total_queries == 0:
        return empty_loss_dict, torch.zeros((0,), device=device, dtype=torch.long)

    if total_targets == 0:
        unmatched_dets = []
        for proposal in seed_proposals:
            det = dict(proposal)
            det["class_id"] = int(proposal["class_id"]) - 1
            unmatched_dets.append(det)
        loss_dict = compute_match_loss(
            matched_pairs=[],
            unmatched_dets=unmatched_dets,
            imgs=imgs_for_loss,
            num_classes=num_classes,
            match_focal_alpha=match_focal_alpha,
            match_focal_gamma=match_focal_gamma,
            lambda_match_cls=lambda_match_cls,
            lambda_match_l1=lambda_match_l1,
            lambda_match_giou=lambda_match_giou,
            other_class_logit=other_class_logit,
            bg_logit=bg_logit,
        )
        return loss_dict, torch.zeros((0,), device=device, dtype=torch.long)

    matched_pairs: List[Tuple[Dict, Dict]] = []
    matched_seed_indices: List[int] = []
    matched_seed_index_set = set()

    for b, proposals_b in enumerate(proposals_per_img):
        target_boxes_b = labeled_targets[b]["boxes"]
        if len(proposals_b) == 0 or target_boxes_b.shape[0] == 0:
            continue

        pred_logits_b = torch.full(
            (1, len(proposals_b), num_classes),
            float(other_class_logit),
            device=device,
            dtype=torch.float32,
        )
        pred_boxes_b = torch.zeros((1, len(proposals_b), 4), device=device, dtype=torch.float32)

        for q, (_, proposal) in enumerate(proposals_b):
            cls_id = int(proposal["class_id"]) - 1
            if not (0 <= cls_id < num_classes):
                continue

            logit = proposal.get("logit", None)
            if logit is None:
                score = proposal.get("score", 1.0)
                if not torch.is_tensor(score):
                    score = torch.tensor(float(score), device=device, dtype=torch.float32)
                else:
                    score = score.to(device=device, dtype=torch.float32)
                logit = torch.log(score.clamp(1e-6, 1.0 - 1e-6))
            elif not torch.is_tensor(logit):
                logit = torch.tensor(float(logit), device=device, dtype=torch.float32)
            else:
                logit = logit.to(device=device, dtype=torch.float32)

            pred_logits_b[0, q, cls_id] = logit
            pred_boxes_b[0, q] = _box_xyxy_to_cxcywh(
                proposal["box"][1:5].to(device=device, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)

        indices_b = matcher(
            {
                "pred_logits": pred_logits_b,
                "pred_boxes": pred_boxes_b,
            },
            [{
                "boxes": labeled_targets[b]["boxes"],
                "labels": labeled_targets[b]["labels"],
            }],
        )[0]

        src_idx, tgt_idx = indices_b
        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
            continue

        target_box5 = labeled_targets[b]["box5"]
        target_labels = labeled_targets[b]["labels"]
        target_scores = labeled_targets[b]["scores"]

        for q_idx, t_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
            proposal_idx, proposal = proposals_b[q_idx]
            det = dict(proposal)
            det["class_id"] = int(proposal["class_id"]) - 1

            target = {
                "box": target_box5[t_idx],
                "class_id": int(target_labels[t_idx].item()),
                "score": target_scores[t_idx],
            }

            matched_pairs.append((det, target))
            matched_seed_indices.append(proposal_idx)
            matched_seed_index_set.add(proposal_idx)

    unmatched_dets: List[Dict] = []
    for proposal_idx, proposal in enumerate(seed_proposals):
        if proposal_idx in matched_seed_index_set:
            continue
        det = dict(proposal)
        det["class_id"] = int(proposal["class_id"]) - 1
        unmatched_dets.append(det)

    loss_dict = compute_match_loss(
        matched_pairs=matched_pairs,
        unmatched_dets=unmatched_dets,
        imgs=imgs_for_loss,
        num_classes=num_classes,
        match_focal_alpha=match_focal_alpha,
        match_focal_gamma=match_focal_gamma,
        lambda_match_cls=lambda_match_cls,
        lambda_match_l1=lambda_match_l1,
        lambda_match_giou=lambda_match_giou,
        other_class_logit=other_class_logit,
        bg_logit=bg_logit,
    )

    matched_seed_indices_tensor = torch.tensor(
        matched_seed_indices,
        device=device,
        dtype=torch.long,
    )

    return loss_dict, matched_seed_indices_tensor


# -----RPN Pseudo Label Generator-----
@dataclass
class RPNPseudoLabelGeneratorConfig:
    device: torch.device
    min_box_size: float = 4.0
    pre_nms_iou_thresh: float = 0.5
    graph_iou_thresh: float = 0.01
    cluster_iou_thresh: float = 0.01
    min_cluster_size: int = 2
    cluster_topk_ratio: float = 0.5
    cluster_merge_mode: str = "envelope"  # "envelope" or "score_weighted"
    use_center_when_small: bool = True
    max_cluster_rounds: int = 5
    keep_empty: bool = False
    overlap_stop_iou_thresh: float = 0.0


class RPNPseudoLabelGenerator(nn.Module):
    def __init__(self, config: RPNPseudoLabelGeneratorConfig) -> None:
        super().__init__()
        self.config = config

        valid_merge_modes = {"envelope", "score_weighted"}
        if self.config.cluster_merge_mode not in valid_merge_modes:
            raise ValueError(
                f"Unsupported cluster_merge_mode: {self.config.cluster_merge_mode}. "
                f"Expected one of {sorted(valid_merge_modes)}."
            )

    def forward(
        self,
        proposals: List[Dict],
        batch_size: int,
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        return self.build(proposals, batch_size, image_sizes)

    def build(
        self,
        proposals: List[Dict],
        batch_size: int,
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        device = self.config.device
        pseudo: List[Dict[str, torch.Tensor]] = []

        boxes_list = [[] for _ in range(batch_size)]
        scores_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]
        original_index_list = [[] for _ in range(batch_size)]

        for proposal_idx, p in enumerate(proposals):
            box5 = p["box"]
            b = int(box5[0].item())
            if b < 0 or b >= batch_size:
                continue

            xyxy = box5[1:5].view(1, 4)
            c = int(p["class_id"])
            s = float(p["score"])

            boxes_list[b].append(xyxy)
            scores_list[b].append(s)
            labels_list[b].append(c)
            original_index_list[b].append(proposal_idx)

        eps = 1e-6
        minsz = float(self.config.min_box_size)

        for b in range(batch_size):
            h_img, w_img = int(image_sizes[b][0]), int(image_sizes[b][1])

            if len(boxes_list[b]) == 0:
                pseudo.append(self._empty_target(device))
                continue

            boxes = torch.cat(boxes_list[b], dim=0).to(device=device, dtype=torch.float32)
            scores = torch.tensor(scores_list[b], device=device, dtype=torch.float32).view(-1)
            labels = torch.tensor(labels_list[b], device=device, dtype=torch.long).view(-1)
            original_indices = torch.tensor(
                original_index_list[b], device=device, dtype=torch.long
            ).view(-1)

            boxes, scores, labels, original_indices = self._sanitize_boxes(
                boxes=boxes,
                scores=scores,
                labels=labels,
                original_indices=original_indices,
                h_img=h_img,
                w_img=w_img,
                eps=eps,
                minsz=minsz,
            )

            if boxes.numel() == 0:
                if not self.config.keep_empty:
                    pseudo.append(self._fallback_target(device, h_img, w_img, minsz))
                else:
                    pseudo.append({"boxes": boxes, "scores": scores, "labels": labels})
                continue

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
                original_indices_c = original_indices[idx_c]

                boxes_c, scores_c, labels_c, original_indices_c = self._pre_nms_one_class(
                    boxes_c, scores_c, labels_c, original_indices_c
                )

                boxes_c, scores_c, labels_c = self._multi_round_cluster_one_class(
                    boxes_c, scores_c, labels_c, original_indices_c
                )

                if boxes_c.numel() == 0:
                    continue

                boxes_c, scores_c, labels_c = self._finalize_class_boxes(
                    boxes_c, scores_c, labels_c, h_img, w_img, eps
                )

                if boxes_c.numel() > 0:
                    out_boxes.append(boxes_c)
                    out_scores.append(scores_c)
                    out_labels.append(labels_c)

            if len(out_boxes) > 0:
                boxes = torch.cat(out_boxes, dim=0)
                scores = torch.cat(out_scores, dim=0)
                labels = torch.cat(out_labels, dim=0)

                order = torch.argsort(scores, descending=True)
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]
            else:
                boxes = boxes[:0]
                scores = scores[:0]
                labels = labels[:0]

            if (not self.config.keep_empty) and boxes.numel() == 0:
                pseudo.append(self._fallback_target(device, h_img, w_img, minsz))
            else:
                pseudo.append(
                    {
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    }
                )

        return pseudo

    def _empty_target(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
            "scores": torch.zeros((0,), device=device, dtype=torch.float32),
            "labels": torch.zeros((0,), device=device, dtype=torch.long),
        }

    def _fallback_target(
        self,
        device: torch.device,
        h_img: int,
        w_img: int,
        minsz: float,
    ) -> Dict[str, torch.Tensor]:
        boxes = torch.tensor([[0.0, 0.0, minsz, minsz]], device=device, dtype=torch.float32)
        boxes[:, 2].clamp_(0.0, max(w_img - 1, 0))
        boxes[:, 3].clamp_(0.0, max(h_img - 1, 0))
        scores = torch.tensor([1.0], device=device, dtype=torch.float32)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
        return w * h

    def _pairwise_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)

        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]

        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-12)

    def _has_overlap(self, boxes: torch.Tensor) -> bool:
        if boxes.size(0) <= 1:
            return False
        ious = self._pairwise_iou(boxes, boxes)
        ious.fill_diagonal_(0.0)
        return bool((ious > self.config.overlap_stop_iou_thresh).any().item())

    def _envelope_fuse_boxes(
        self,
        boxes_in: torch.Tensor,
        scores_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes_in.size(0) == 1:
            return boxes_in[0], scores_in[0]

        fused_box = torch.stack(
            [
                boxes_in[:, 0].min(),
                boxes_in[:, 1].min(),
                boxes_in[:, 2].max(),
                boxes_in[:, 3].max(),
            ],
            dim=0,
        )
        fused_score = scores_in.max()
        return fused_box, fused_score

    def _score_weighted_fuse_boxes(
        self,
        boxes_in: torch.Tensor,
        scores_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes_in.size(0) == 1:
            return boxes_in[0], scores_in[0]

        weights = scores_in.clamp(min=0.0)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = torch.ones_like(scores_in)
            weight_sum = weights.sum()

        normalized_weights = weights / weight_sum
        fused_box = (boxes_in * normalized_weights[:, None]).sum(dim=0)
        fused_score = scores_in.max()
        return fused_box, fused_score

    def _merge_cluster_boxes(
        self,
        boxes_in: torch.Tensor,
        scores_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.cluster_merge_mode == "envelope":
            return self._envelope_fuse_boxes(boxes_in, scores_in)
        if self.config.cluster_merge_mode == "score_weighted":
            return self._score_weighted_fuse_boxes(boxes_in, scores_in)
        raise ValueError(
            f"Unsupported cluster_merge_mode: {self.config.cluster_merge_mode}"
        )

    def _sanitize_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        original_indices: torch.Tensor,
        h_img: int,
        w_img: int,
        eps: float,
        minsz: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        boxes = boxes[finite]
        scores = scores[finite]
        labels = labels[finite]
        original_indices = original_indices[finite]

        if boxes.numel() == 0:
            return boxes, scores, labels, original_indices

        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        boxes[:, 0].clamp_(0.0, max(w_img - 1, 0))
        boxes[:, 2].clamp_(0.0, max(w_img - 1, 0))
        boxes[:, 1].clamp_(0.0, max(h_img - 1, 0))
        boxes[:, 3].clamp_(0.0, max(h_img - 1, 0))

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        pos = (w > eps) & (h > eps)
        boxes = boxes[pos]
        scores = scores[pos]
        labels = labels[pos]
        original_indices = original_indices[pos]

        if boxes.numel() == 0:
            return boxes, scores, labels, original_indices

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

        nx1 = nx1.clamp(0.0, max(w_img - 1, 0))
        nx2 = nx2.clamp(0.0, max(w_img - 1, 0))
        ny1 = ny1.clamp(0.0, max(h_img - 1, 0))
        ny2 = ny2.clamp(0.0, max(h_img - 1, 0))

        ax1 = torch.minimum(nx1, nx2)
        ay1 = torch.minimum(ny1, ny2)
        ax2 = torch.maximum(nx1, nx2)
        ay2 = torch.maximum(ny1, ny2)
        boxes = torch.stack([ax1, ay1, ax2, ay2], dim=-1)

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid = (
            (w > eps)
            & (h > eps)
            & torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(scores)
        )

        return boxes[valid], scores[valid], labels[valid], original_indices[valid]

    def _finalize_class_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        h_img: int,
        w_img: int,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return boxes, scores, labels

        boxes[:, 0].clamp_(0.0, max(w_img - 1, 0))
        boxes[:, 2].clamp_(0.0, max(w_img - 1, 0))
        boxes[:, 1].clamp_(0.0, max(h_img - 1, 0))
        boxes[:, 3].clamp_(0.0, max(h_img - 1, 0))

        fx1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        fy1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        fx2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        fy2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        boxes = torch.stack([fx1, fy1, fx2, fy2], dim=-1)

        fw = boxes[:, 2] - boxes[:, 0]
        fh = boxes[:, 3] - boxes[:, 1]
        valid = (
            (fw > eps)
            & (fh > eps)
            & torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(scores)
        )

        return boxes[valid], scores[valid], labels[valid]

    def _pre_nms_one_class(
        self,
        boxes_c: torch.Tensor,
        scores_c: torch.Tensor,
        labels_c: torch.Tensor,
        original_indices_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if boxes_c.numel() == 0:
            return boxes_c, scores_c, labels_c, original_indices_c

        keep = nms(boxes_c, scores_c, self.config.pre_nms_iou_thresh)
        return (
            boxes_c[keep],
            scores_c[keep],
            labels_c[keep],
            original_indices_c[keep],
        )

    def _find_cluster_centers(
        self,
        boxes_c: torch.Tensor,
        scores_c: torch.Tensor,
        original_indices_c: torch.Tensor,
    ) -> torch.Tensor:
        num_props = boxes_c.size(0)
        if num_props == 0:
            return torch.zeros((0,), device=boxes_c.device, dtype=torch.long)
        if num_props == 1:
            return torch.zeros((1,), device=boxes_c.device, dtype=torch.long)

        ious = self._pairwise_iou(boxes_c, boxes_c)
        adjacency = ious > self.config.graph_iou_thresh
        adjacency.fill_diagonal_(False)

        remaining = torch.ones(num_props, dtype=torch.bool, device=boxes_c.device)
        all_idx = torch.arange(num_props, device=boxes_c.device)
        centers: List[int] = []

        while remaining.any():
            remaining_idx = torch.nonzero(remaining, as_tuple=False).squeeze(1)
            sub_adj = adjacency[remaining_idx][:, remaining_idx]
            degrees = sub_adj.sum(dim=1)

            max_degree = degrees.max()
            candidate_idx = remaining_idx[degrees == max_degree]

            if candidate_idx.numel() > 1:
                candidate_scores = scores_c[candidate_idx]
                max_score = candidate_scores.max()
                candidate_idx = candidate_idx[candidate_scores == max_score]

            if candidate_idx.numel() > 1:
                candidate_orders = original_indices_c[candidate_idx]
                chosen = candidate_idx[torch.argmin(candidate_orders)]
            else:
                chosen = candidate_idx[0]

            chosen_int = int(chosen.item())
            centers.append(chosen_int)

            remove_mask = adjacency[chosen_int] | (all_idx == chosen_int)
            remaining = remaining & (~remove_mask)

        return torch.tensor(centers, device=boxes_c.device, dtype=torch.long)

    def _cluster_one_round(
        self,
        boxes_c: torch.Tensor,
        scores_c: torch.Tensor,
        labels_c: torch.Tensor,
        original_indices_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if boxes_c.numel() == 0:
            return boxes_c[:0], scores_c[:0], labels_c[:0]

        centers = self._find_cluster_centers(boxes_c, scores_c, original_indices_c)
        if centers.numel() == 0:
            return boxes_c[:0], scores_c[:0], labels_c[:0]

        center_boxes = boxes_c[centers]
        center_scores = scores_c[centers]
        center_labels = labels_c[centers]

        center_ious = self._pairwise_iou(boxes_c, center_boxes)

        is_center = torch.zeros(boxes_c.size(0), dtype=torch.bool, device=boxes_c.device)
        is_center[centers] = True

        assignments = torch.full(
            (boxes_c.size(0),), -1, dtype=torch.long, device=boxes_c.device
        )
        assignments[centers] = torch.arange(
            centers.numel(), device=boxes_c.device, dtype=torch.long
        )

        non_center_idx = torch.nonzero(~is_center, as_tuple=False).squeeze(1)
        if non_center_idx.numel() > 0:
            non_center_ious = center_ious[non_center_idx]
            best_iou, best_center = non_center_ious.max(dim=1)
            assign_mask = (
                (best_iou > self.config.graph_iou_thresh)
                & (best_iou >= self.config.cluster_iou_thresh)
            )
            assignments[non_center_idx[assign_mask]] = best_center[assign_mask]

        fused_boxes = []
        fused_scores = []
        fused_labels = []

        for center_pos in range(centers.numel()):
            member_idx = torch.nonzero(assignments == center_pos, as_tuple=False).squeeze(1)
            if member_idx.numel() == 0:
                continue

            member_idx_list = member_idx.tolist()
            member_idx_list = sorted(
                member_idx_list,
                key=lambda i: (-float(scores_c[i].item()), int(original_indices_c[i].item()))
            )

            cluster_size = len(member_idx_list)
            topk = max(1, int(cluster_size * self.config.cluster_topk_ratio + 0.999999))
            topk = min(topk, cluster_size)

            merged_idx = torch.tensor(
                member_idx_list[:topk],
                device=boxes_c.device,
                dtype=torch.long,
            )

            if merged_idx.numel() < self.config.min_cluster_size:
                if not self.config.use_center_when_small:
                    continue
                fused_boxes.append(center_boxes[center_pos].view(1, 4))
                fused_scores.append(center_scores[center_pos].view(1))
                fused_labels.append(center_labels[center_pos].view(1))
                continue

            fused_box, fused_score = self._merge_cluster_boxes(
                boxes_c[merged_idx],
                scores_c[merged_idx],
            )

            fused_boxes.append(fused_box.view(1, 4))
            fused_scores.append(fused_score.view(1))
            fused_labels.append(center_labels[center_pos].view(1))

        if len(fused_boxes) == 0:
            return boxes_c[:0], scores_c[:0], labels_c[:0]

        return (
            torch.cat(fused_boxes, dim=0),
            torch.cat(fused_scores, dim=0),
            torch.cat(fused_labels, dim=0),
        )

    def _multi_round_cluster_one_class(
        self,
        boxes_c: torch.Tensor,
        scores_c: torch.Tensor,
        labels_c: torch.Tensor,
        original_indices_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if boxes_c.numel() == 0:
            return boxes_c[:0], scores_c[:0], labels_c[:0]

        cur_boxes = boxes_c
        cur_scores = scores_c
        cur_labels = labels_c
        cur_indices = original_indices_c

        for _ in range(self.config.max_cluster_rounds):
            if cur_boxes.size(0) <= 1:
                break

            if not self._has_overlap(cur_boxes):
                break

            prev_boxes = cur_boxes.clone()
            prev_scores = cur_scores.clone()
            prev_num = cur_boxes.size(0)

            cur_boxes, cur_scores, cur_labels = self._cluster_one_round(
                cur_boxes, cur_scores, cur_labels, cur_indices
            )

            if cur_boxes.numel() == 0:
                break

            cur_indices = torch.arange(
                cur_boxes.size(0),
                device=cur_boxes.device,
                dtype=torch.long,
            )

            same_num = cur_boxes.size(0) == prev_num
            same_boxes = same_num and torch.allclose(cur_boxes, prev_boxes, atol=1e-6, rtol=1e-6)
            same_scores = same_num and torch.allclose(cur_scores, prev_scores, atol=1e-6, rtol=1e-6)

            if same_boxes and same_scores:
                break

        return cur_boxes, cur_scores, cur_labels


def build_RPN_pseudo_label_generator(device : torch.device) -> RPNPseudoLabelGenerator:
    config = RPNPseudoLabelGeneratorConfig(device=device)

    return RPNPseudoLabelGenerator(config)

