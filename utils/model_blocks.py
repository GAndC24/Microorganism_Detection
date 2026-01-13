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
        wb_labels : torch.Tensor    # class label for weak boxes, [R, num_classes]
    )-> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        :param x:
        :return: prototypes, {class_id: prototype tensor}
        """

        # RoI Align to get weak box features
        roi_features = self.roi_align(x, wboxes)       # [R = num_wboxes, C, H, W]

        # get CAMs
        cams, loss_cam = self.cam_head(roi_features, wb_labels)        # cams, [R, num_classes, H, W]

        # patch embedding
        patch_features = self.patch_embed(roi_features)     # [R, Np = num_patches, D]

        # get fg/bg scores for each patch
        patch_fg_bg_scores = self._cam_to_patch_fg_bg_scores(cams, wb_labels)      # [R, Np, 2], fg_scores = [:, :, 0], bg_scores = [:, :, 1]

        # get anchor feature of each weak box
        R, Np, D = patch_features.shape
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

        # get morphological prototypes
        prototypes = self._get_morphological_prototypes(patch_features, weights, wb_labels)        # {class_id : prototype tensor}

        return loss_cam, prototypes

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