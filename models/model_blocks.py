import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, List
from torchvision.models import vgg16

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

# Morphological Prototype Generator
class MorphologicalPrototypeGenerator(nn.Module):
    def __init__(
        self,
        backbone : nn.Module,   # Default VGG-16
        num_classes : int,  # number of classes

    )->None:
        super(MorphologicalPrototypeGenerator, self).__init__()

        self.encoder = backbone
        self.num_classes = num_classes
        self.num_prototypes = num_classes

# Build backbone hooker
def build_backbone_hooker(backbone : nn.Module, indices : set = (8, 22, 29)) -> FeatureHook:
    """
    :param backbone: Default VGG-16 backbone
    :param indices: feature layer indices
    :return: FeatureHook object
    """
    hooker = FeatureHook()
    for idx, tag in zip(indices, ['low', 'mid', 'high']):
        module = backbone.features[idx]
        hooker.register(module, tag)
    return hooker

# Build VGG-16 backbone with hooker
def build_vgg16_backbone_with_hooker() -> Tuple[nn.Module, FeatureHook]:
    """
    :return: VGG-16 backbone and FeatureHook object
    """
    # Load default VGG-16 backbone
    backbone = vgg16(pretrained=True)
    # Build backbone hooker
    hooker = build_backbone_hooker(backbone)
    return backbone, hooker