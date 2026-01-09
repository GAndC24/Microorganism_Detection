"""
Part of this code is adapted and modified from the official implementation of:
MuSCLe: A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation
GitHub repository: https://github.com/SCoulY/MuSCLe
"""
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

# -----Image Contrastive Loss-----
def get_img_contrast_loss(emb : torch.Tensor, label : torch.Tensor)->torch.Tensor:
    '''
    get the loss_img
    :param emb: embedding features, [B, D]
    :param label: multi-hot labels, [B, num_classes]
    :return: loss_img
    '''
    loss_img = 0
    batch_size = emb.shape[0]

    # 每个样本作为 anchor
    for i in range(batch_size):
        sim_pos = 1e-6
        sim_neg = 1e-6
        neg_list = range(i + 1, batch_size)
        valid_pos = 0
        valid_neg = 0

        # 构造正负样本对
        for j in neg_list:
            if torch.bitwise_and(label[i].long(), label[j].long()).sum() > 0:
                sim_pos = sim_pos + torch.exp((emb[i] * emb[j]).sum() / 0.1)
                valid_pos += 1
            if torch.bitwise_and(label[i].long(), label[j].long()).sum() == 0:
                sim_neg = sim_neg + torch.exp((emb[i] * emb[j]).sum() / 0.1)
                valid_neg += 1
        if torch.is_tensor(sim_pos) and torch.is_tensor(sim_neg) and valid_neg > valid_pos:
            sim_pos = sim_pos
            sim_neg = sim_pos + sim_neg
            loss_img = loss_img - torch.log(sim_pos / sim_neg)
        else:
            del sim_neg
            del sim_pos

    loss_img /= batch_size
    return loss_img

# -----Weak Bounding Box Contrastive Loss-----
# Default All Views
class LossContrastMode(str, Enum):
    ALL_VIEWS = "all_views"
    ONE_VIEW = "one_view"

# Default Outside
class LossSummationLocation(str, Enum):
    OUTSIDE = "outside"  # sum positives of log-probs
    INSIDE = "inside"    # log of summed probs

# Default All
class LossDenominatorMode(str, Enum):
    ALL = "all"                # denominator includes positives + negatives (except strict self-self)
    ONE_POSITIVE = "one_positive"
    ONLY_NEGATIVES = "only_negatives"

# Loss Config
@dataclass
class WBBLossConfig:
    temperature: float = 0.07
    contrast_mode: LossContrastMode = LossContrastMode.ALL_VIEWS
    summation_location: LossSummationLocation = LossSummationLocation.OUTSIDE
    denominator_mode: LossDenominatorMode = LossDenominatorMode.ALL
    positives_cap: int = -1  # -1 means no cap
    scale_by_temperature: bool = True

def build_diagonal_mask(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """:return diagonal mask, [B, B]"""
    return torch.eye(batch_size, device=device, dtype=dtype)

def _cap_positives_mask(
    untiled_mask: torch.Tensor,   # [B, B] (class positives)
    diagonal_mask: torch.Tensor,  # [B, B] identity
    positives_cap: int,
    num_views: int,
) -> torch.Tensor:
    """
    在样本级别的掩码中（未扩展到视图之前），为每个锚点（anchor）限制额外正样本的数量（不包括对角线上的样本，即排除自我匹配的样本）。
    :return capped mask, [B, B]
    """
    if positives_cap <= -1:
        return untiled_mask

    # Remove diagonal (same sample), keep it separately
    mask_no_diag = torch.minimum(untiled_mask, (1.0 - diagonal_mask))

    k = positives_cap // num_views  # cap in sample-space
    if k <= 0:
        return diagonal_mask.clone()

    # Row-wise topk on {0,1} mask; may include zeros if insufficient positives
    values, indices = torch.topk(mask_no_diag, k=min(k, mask_no_diag.shape[1]), dim=1)

    capped = torch.zeros_like(mask_no_diag)
    row_idx = torch.arange(mask_no_diag.shape[0], device=mask_no_diag.device).unsqueeze(1)
    keep = values > 0
    if keep.any():
        capped[row_idx.expand_as(indices)[keep], indices[keep]] = 1.0

    # Add diagonal back
    capped = torch.maximum(capped, diagonal_mask)
    return capped

def _create_tiled_masks(
    untiled_class_mask_uncapped: torch.Tensor,  # [B, B] class positives (uncapped)
    diagonal_mask: torch.Tensor,                # [B, B] identity
    num_views: int,
    num_anchor_views: int,
    positives_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positives/negatives masks in view-expanded space.

    - positives_mask is built from (optionally capped) class positives mask
    - negatives_mask is built from UNCAPPED class positives mask complement
      (matches the TF intent: cap affects positives aggregation, not class definition of negatives)
    - strict self-self (same sample, same view) is removed from both

    Returns:
      positives_mask: [B*num_anchor_views, B*num_views]
      negatives_mask: same
      num_positives_per_row: [B*num_anchor_views]
    """
    device = untiled_class_mask_uncapped.device
    dtype = untiled_class_mask_uncapped.dtype
    B = untiled_class_mask_uncapped.shape[0]

    # Apply cap to class positives only for positives aggregation
    if positives_cap > -1:
        untiled_class_mask = _cap_positives_mask(
            untiled_mask=untiled_class_mask_uncapped,
            diagonal_mask=diagonal_mask,
            positives_cap=positives_cap,
            num_views=num_views,
        )
    else:
        untiled_class_mask = untiled_class_mask_uncapped

    # Tile into view-expanded masks
    positives_mask = untiled_class_mask.repeat(num_anchor_views, num_views)  # [B*AV, B*V]
    uncapped_positives_mask = untiled_class_mask_uncapped.repeat(num_anchor_views, num_views)

    negatives_mask = (1.0 - uncapped_positives_mask)

    # Remove strict self-self: anchor (sample i, view av) vs global (same sample i, same view av)
    all_but_strict_self = torch.ones((B * num_anchor_views, B * num_views), device=device, dtype=dtype)

    i = torch.arange(B, device=device)
    for av in range(num_anchor_views):
        anchor_rows = av * B + i
        # With the packing used below (view-major packing), global columns for view av are in [av*B : (av+1)*B)
        global_cols = av * B + i
        all_but_strict_self[anchor_rows, global_cols] = 0.0

    positives_mask = positives_mask * all_but_strict_self
    negatives_mask = negatives_mask * all_but_strict_self

    num_pos = positives_mask.sum(dim=1)  # [B*AV]
    return positives_mask, negatives_mask, num_pos