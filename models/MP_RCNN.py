# Morphological Prototype R-CNN
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from model_blocks import MorphologicalPrototypeGenerator, FeatureHook

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
        hooker: FeatureHook,         # Feature Hook
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

        self.backbone = backbone
        self.hooker = hooker
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

    def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """

        :param x: input images, [B, C, H, W]
        :return:
        - low-level feature maps, [B, C1, H1, W1]
        - high-level feature maps, [B, C2, H2, W2]
        - prototypes, {class_id, prototype tensor}
        """



