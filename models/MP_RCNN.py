# Morphological Prototype R-CNN
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from dataclasses import dataclass
from utils import random_masking, add_gaussian_noise, MorphologicalPrototypeGenerator, FeatureHook, build_vgg16_backbone_with_hook
from torchvision.ops import RoIAlign

# VGG-16 layer output channel maps, {layer index : output channels}
vgg_layer_out_c_maps = {
    3 : 64,     # Relu1_2
    8 : 128,    # Relu2_2
    15 : 256,   # Relu3_3
    22 : 512,   # Relu4_3
    29 : 512,    # Relu5_3
    30 : 512   # MaxPool5
}
# VGG-16 layer output size ratio maps, {layer index : output size ratio}
vgg_layer_out_size_ratio_maps = {
    3 : 1.0,      # Relu1_2, (H, W)
    8 : 1/2,      # Relu2_2, (H/2, W/2)
    15 : 1/4,     # Relu3_3, (H/4, W/4)
    22 : 1/8,    # Relu4_3, (H/8, W/8)
    29 : 1/16,     # Relu5_3, (H/16, W/16)
    30 : 1/32    # MaxPool5, (H/32, W/32)
}

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
    spatial_scale_high : float # spatial scale for middle feature maps
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag

# Stage 1: Feature Alignment and Construct Prototypes
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
        self.mlp_h = nn.Sequential(
            nn.Flatten(),  # [N, C3, H3, W3] -> [N, C3*H3*W3]
            nn.Linear(hidden_dim_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.ReLU(),
        )
        self.bn_h = nn.BatchNorm1d(config.embed_dim)

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
    )-> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """
        :return:
        - low-level latent features, [B, C1]
        - high-level feature maps, [B, C2, H2, W2]
        - high-level embedding features, [B, D]
        - CAM loss
        - prototypes, {class_id, prototype tensor}
        """
        # -----get multi-level feature maps-----
        self.hook.clear()
        _ = self.encoder(x)
        feature_maps = self.hook.outputs
        low_feature_maps = feature_maps['low']      # [B, C1, H1, W1]
        mid_feature_maps = feature_maps['mid']      # [B, C2, H2, W2]
        high_feature_maps = feature_maps['high']        # [B, C3, H3, W3]

        # -----get low-level latent features-----
        low_latent_features = self.gap_l(low_feature_maps)    # [B, C1, 1, 1]
        B, C1, _, _ = low_latent_features.shape
        low_latent_features = low_latent_features.view(B, C1)
        low_latent_features = self.bn_l(low_latent_features)        # [B, C1]

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
        high_aug_embedding_features = self.mlp_h(roi_views_flat)  # [R*V, D]
        high_aug_embedding_features = self.bn_h(high_aug_embedding_features)  # [R*V, D]
        high_aug_embedding_features = high_aug_embedding_features.view(R, V, self.embed_dim)  # [R, V, D]

        # -----construct prototypes-----
        loss_cam, prototypes = self.mp_generator(mid_feature_maps, wboxes, wb_labels)        # {class_id : prototype tensor}

        return low_latent_features, high_feature_maps, high_aug_embedding_features, loss_cam, prototypes


def build_Stage1_model(
    stage1_config: Stage1Config,        # Stage1 configuration
)-> nn.Module:
    backbone, hook = build_vgg16_backbone_with_hook(stage1_config.layer_indices)

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





