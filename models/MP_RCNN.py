# Morphological Prototype R-CNN
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from model_blocks import MorphologicalPrototypeGenerator, FeatureHook, build_vgg16_backbone_with_hook

# VGG-16 layer output channel maps, {layer index : output channels}
vgg_layer_out_c_maps = {
    3 : 64,     # Relu1_2
    8 : 128,    # Relu2_2
    15 : 256,   # Relu3_3
    22 : 512,   # Relu4_3
    29 : 512    # Relu5_3
}
# VGG-16 layer output size ratio maps, {layer index : output size ratio}
vgg_layer_out_size_ratio_maps = {
    3 : 0.5,      # Relu1_2
    8 : 0.25,      # Relu2_2
    15 : 0.125,     # Relu3_3
    22 : 0.0625,    # Relu4_3
    29 : 0.03125     # Relu5_3
}

# Stage 1: Feature Alignment and Construct Prototypes
class Stage1(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16
        hook: FeatureHook,         # Feature Hook
        mp_generator: MorphologicalPrototypeGenerator,  # Morphological Prototype Generator
        num_classes: int,    # number of classes
        img_size: Tuple[int, int],  # input image size
        batch_size: int,        # batch size
        layer_indices: List[int],     # feature layer indices, [low, mid, high]
        embed_dim : int, # embedding dimension
        # MLP parameters
        hidden_dim : int = 4096,     # MLP hidden dimension

    )-> None:
        super(Stage1, self).__init__()

        self.encoder = backbone
        self.hook = hook
        self.mp_generator = mp_generator
        self.num_classes = num_classes
        self.num_prototypes = num_classes
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.gap_l = nn.AdaptiveAvgPool2d((1, 1))
        low_out_c = vgg_layer_out_c_maps[layer_indices[0]]
        self.bn_l = nn.BatchNorm1d(batch_size * low_out_c)

        hidden_dim_in = int(vgg_layer_out_c_maps[layer_indices[2]] * (img_size[0] * vgg_layer_out_size_ratio_maps[layer_indices[2]])**2)
        self.mlp_h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.bn_h = nn.BatchNorm1d(batch_size * embed_dim)

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

    def forward(self, x: torch.Tensor)-> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """
        :param x: input images, [B, C, H, W]
        :return:
        - low-level latent features, [B, C1]
        - high-level feature maps, [B, C2, H2, W2]
        - high-level embedding features, [B, D]
        - prototypes, {class_id, prototype tensor}
        """
        # get multi-level feature maps
        self.hook.clear()
        _ = self.encoder(x)
        feature_maps = self.hook.outputs
        low_feature_maps = feature_maps['low']      # [B, C1, H1, W1]
        mid_feature_maps = feature_maps['mid']      # [B, C2, H2, W2]
        high_feature_maps = feature_maps['high']        # [B, C3, H3, W3]

        # get low-level latent features
        low_latent_features = self.gap_l(low_feature_maps)    # [B, C1, 1, 1]
        B, C1, _, _ = low_latent_features.shape
        low_latent_features = low_latent_features.view(B, C1)
        low_latent_features = self.bn_l(low_latent_features.view(B * C1, 1)).view(B, C1)        # [B, C1]

        # get high-level embedding features
        B, C3, H3, W3 = high_feature_maps.shape
        high_embedding_features = self.mlp_h(high_feature_maps)    # [B, D]
        high_embedding_features = self.bn_h(high_embedding_features.view(B * self.embed_dim, 1)).view(B, self.embed_dim)    # [B, D]

        # construct prototypes
        prototypes = self.mp_generator(mid_feature_maps)        # {class_id : prototype tensor}

        return low_latent_features, high_feature_maps, high_embedding_features, prototypes


def build_Stage1_model(
    num_classes: int,    # number of classes
    embed_dim: int,  # embedding dimension
    # for hook
    layer_indices: List[int],     # feature layer indices, [low, mid, high]
    # for mp_generator
    in_c : int,      # input channels
    wb_labels: torch.Tensor,  # class label for weak boxes, [R, num_classes]
    patch_size : int,       # patch size
    components_range : List,   # list of number of components for GMM
    random_state : int,     # random state for GMM(seed)
    max_iter : int,     # max iteration for EM
    wboxes : torch.Tensor,      # weak boxes, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
    roi_out_size: Tuple[int, int],  # output size
    # for Stage1
    img_size: Tuple[int, int],  # input image size
    batch_size: int,        # batch size
    hidden_dim : int = 4096,     # MLP hidden dimension
)-> nn.Module:
    backbone, hook = build_vgg16_backbone_with_hook(layer_indices)

    mp_generator = MorphologicalPrototypeGenerator(
        num_classes=num_classes,
        in_c=in_c,
        wb_labels=wb_labels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        components_range=components_range,
        random_state=random_state,
        max_iter=max_iter,
        wboxes=wboxes,
        roi_out_size=roi_out_size
    )

    model = Stage1(
        backbone=backbone,
        hook=hook,
        mp_generator=mp_generator,
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        layer_indices=layer_indices,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )

    return model





