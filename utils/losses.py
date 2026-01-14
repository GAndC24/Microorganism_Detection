"""
Copyright：Part of this code is adapted and modified from the official implementation of:
MuSCLe: A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation
GitHub repository: https://github.com/SCoulY/MuSCLe
"""
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import torch.nn.functional as F

# -----Image Contrastive Loss-----
def get_img_contrast_loss(emb : torch.Tensor, label : torch.Tensor, tau : float = 0.1, eps : float = 1e-6)->torch.Tensor:
    '''
    get the loss_img
    :param emb: embedding features, [B, D]
    :param label: multi-hot labels, [B, num_classes]
    :return: loss_img
    '''
    # loss_img = 0
    # batch_size = emb.shape[0]
    #
    # # 每个样本作为 anchor
    # for i in range(batch_size):
    #     sim_pos = 1e-6
    #     sim_neg = 1e-6
    #     neg_list = range(i + 1, batch_size)
    #     valid_pos = 0
    #     valid_neg = 0
    #
    #     # 构造正负样本对
    #     for j in neg_list:
    #         if torch.bitwise_and(label[i].long(), label[j].long()).sum() > 0:       # 有共同类别，正样本对
    #             sim_pos = sim_pos + torch.exp((emb[i] * emb[j]).sum() / 0.1)
    #             valid_pos += 1
    #         if torch.bitwise_and(label[i].long(), label[j].long()).sum() == 0:      # 无共同类别，负样本对
    #             sim_neg = sim_neg + torch.exp((emb[i] * emb[j]).sum() / 0.1)
    #             valid_neg += 1
    #     if torch.is_tensor(sim_pos) and torch.is_tensor(sim_neg) and valid_neg > valid_pos:
    #         sim_pos = sim_pos
    #         sim_neg = sim_pos + sim_neg
    #         loss_img = loss_img - torch.log(sim_pos / sim_neg)
    #     else:
    #         del sim_neg
    #         del sim_pos
    #
    # loss_img /= batch_size
    # return loss_img

    emb = F.normalize(emb, dim=-1, eps=eps)
    B = emb.size(0)

    # pairwise logits: [B, B]
    logits = emb @ emb.t() / tau

    loss_sum = emb.new_tensor(0.0)
    valid_i = 0

    for i in range(B):
        # j > i 和你原始一致（避免重复计数）
        js = torch.arange(i + 1, B, device=emb.device)

        if js.numel() == 0:
            continue

        li = label[i]
        lj = label[js]

        pos_mask = (lj == li).all(dim=-1)
        neg_mask = (li.long() & lj.long()).sum(dim=-1) == 0

        pos_js = js[pos_mask]
        neg_js = js[neg_mask]

        if pos_js.numel() == 0 or neg_js.numel() == 0:
            continue

        # log numerator = logsumexp(logits[i, pos_js])
        log_num = torch.logsumexp(logits[i, pos_js], dim=0)

        # log denominator = logsumexp(concat(pos, neg))
        denom_js = torch.cat([pos_js, neg_js], dim=0)
        log_den = torch.logsumexp(logits[i, denom_js], dim=0)

        loss_i = -(log_num - log_den)
        loss_sum = loss_sum + loss_i
        valid_i += 1

    if valid_i == 0:
        # 避免 0/0；也可以直接 return 0
        return emb.new_tensor(0.0)

    return loss_sum / valid_i

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
    reduction : str = "mean"  # 'mean' or 'sum' or 'none'

def build_diagonal_mask(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """:return diagonal mask, [B, B]"""
    return torch.eye(batch_size, device=device, dtype=dtype)

def _create_tiled_masks(
    untiled_class_mask_uncapped: torch.Tensor,  # [B, B] positives (uncapped)
    diagonal_mask: torch.Tensor,                # [B, B] identity
    num_views: int,
    num_anchor_views: int,
    positives_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positives/negatives masks in view-expanded space.
    - positives_mask is built from (optionally capped) class positives mask
    - negatives_mask is built from UNCAPPED class positives mask complement
    - strict self-self (same sample, same view) is removed from both

    :return:
    - positives_mask: [B*num_anchor_views, B*num_views]
    - negatives_mask: same
    - num_positives_per_row: [B*num_anchor_views]
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

def _cap_positives_mask(
    untiled_mask: torch.Tensor,   # [B, B] (positives)
    diagonal_mask: torch.Tensor,  # [B, B]
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

def supervised_contrastive_loss(
    features: torch.Tensor,                 # [B = num_wbb, V = num_views, D]
    labels: Optional[torch.Tensor] = None,  # [B = num_wbb, num_classes] one-hot
    cfg: WBBLossConfig = WBBLossConfig(),
) -> torch.Tensor:
    """:return: loss: Weal Box Contrastive Loss pre sample, [B]"""
    # Minimal inline checks
    if features.dim() < 3:
        raise ValueError(f"`features` must be [B, V, D] with dim>=3, got {tuple(features.shape)}")

    features = F.normalize(features, dim=-1, eps=1e-6)
    B, V = features.shape[0], features.shape[1]
    if B <= 0 or V <= 0:
        raise ValueError(f"Invalid B or V from features shape={tuple(features.shape)}")

    if labels is not None:
        if labels.dim() != 2:
            raise ValueError(f"`labels` must be [B, C] (rank-2), got {tuple(labels.shape)}")
        if labels.shape[0] != B:
            raise ValueError(f"labels.shape[0] must equal B. labels={labels.shape[0]}, B={B}")

    if cfg.positives_cap is not None and cfg.positives_cap > -1:
        if cfg.positives_cap % V != 0:
            raise ValueError(
                f"positives_cap must be multiple of num_views. positives_cap={cfg.positives_cap}, num_views={V}"
            )

    device = features.device

    # Flatten to [B, V, D]
    if features.dim() > 3:
        features = features.view(B, V, -1)
    features = features.float()

    # Single GPU: global == local
    global_features = features  # [B, V, D]
    diagonal_mask = build_diagonal_mask(B, device=device, dtype=features.dtype)  # [B, B]

    # Build sample-level class mask (uncapped)
    if labels is None:
        # self-supervised (SimCLR-like): sample-level positives are "same sample"
        untiled_class_mask_uncapped = diagonal_mask
    else:
        labels = labels.to(device=device, dtype=features.dtype)
        global_labels = labels
        class_sim = torch.matmul(labels, global_labels.t())  # [B, B]
        untiled_class_mask_uncapped = (class_sim > 0).to(features.dtype)

    # Pack global features view-major: [B,V,D] -> [V,B,D] -> [V*B, D]
    all_global = global_features.permute(1, 0, 2).contiguous().view(V * B, -1)

    # Select anchor features
    if cfg.contrast_mode == LossContrastMode.ONE_VIEW:
        anchor = features[:, 0, :].contiguous()  # [B, D]
        num_anchor_views = 1
    elif cfg.contrast_mode == LossContrastMode.ALL_VIEWS:
        anchor = features.permute(1, 0, 2).contiguous().view(V * B, -1)  # [V*B, D]
        num_anchor_views = V
    else:
        raise ValueError(f"Unknown contrast_mode: {cfg.contrast_mode}")

    # Logits: [B*AV, V*B]
    logits = torch.matmul(anchor, all_global.t()) / float(cfg.temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # stability
    exp_logits = torch.exp(logits)

    # Masks
    positives_mask, negatives_mask, num_pos = _create_tiled_masks(
        untiled_class_mask_uncapped=untiled_class_mask_uncapped,
        diagonal_mask=diagonal_mask,
        num_views=V,
        num_anchor_views=num_anchor_views,
        positives_cap=cfg.positives_cap,
    )

    eps = 1e-12

    # ---- Denominator ----
    if cfg.denominator_mode == LossDenominatorMode.ALL:
        denom = exp_logits.sum(dim=1, keepdim=True)  # [B*AV, 1]
        denom_matrix = None
    elif cfg.denominator_mode == LossDenominatorMode.ONLY_NEGATIVES:
        denom = (exp_logits * negatives_mask).sum(dim=1, keepdim=True)  # [B*AV, 1]
        denom_matrix = None
    elif cfg.denominator_mode == LossDenominatorMode.ONE_POSITIVE:
        neg_sum = (exp_logits * negatives_mask).sum(dim=1, keepdim=True)  # [B*AV, 1]
        denom_matrix = neg_sum + exp_logits  # [B*AV, V*B]
        denom = None
    else:
        raise ValueError(f"Unknown denominator_mode: {cfg.denominator_mode}")

    # Summation location
    if cfg.summation_location == LossSummationLocation.OUTSIDE:
        if cfg.denominator_mode == LossDenominatorMode.ONE_POSITIVE:
            log_prob = logits - torch.log(denom_matrix + eps)
            pos_log_prob = (log_prob * positives_mask).sum(dim=1)
        else:
            log_prob = logits - torch.log(denom + eps)
            pos_log_prob = (log_prob * positives_mask).sum(dim=1)

        pos_log_prob = torch.where(num_pos > 0, pos_log_prob / (num_pos + eps), torch.zeros_like(pos_log_prob))
        loss = -pos_log_prob

    elif cfg.summation_location == LossSummationLocation.INSIDE:
        if cfg.denominator_mode == LossDenominatorMode.ONE_POSITIVE:
            probs = (exp_logits / (denom_matrix + eps)) * positives_mask
        else:
            probs = (exp_logits / (denom + eps)) * positives_mask

        pos_prob_sum = probs.sum(dim=1)
        mean_pos_prob = torch.where(num_pos > 0, pos_prob_sum / (num_pos + eps), torch.zeros_like(pos_prob_sum))
        loss = -torch.log(mean_pos_prob + eps)
    else:
        raise ValueError(f"Unknown summation_location: {cfg.summation_location}")

    if cfg.scale_by_temperature:
        loss = loss * float(cfg.temperature)

    # Reduce anchors -> per-sample [B]
    if num_anchor_views > 1:
        loss = loss.view(num_anchor_views, B).mean(dim=0)
    else:
        loss = loss.view(B)

    # Final reduction
    if cfg.reduction == "mean":
        loss = loss.mean()
    elif cfg.reduction == "sum":
        loss = loss.sum()
    elif cfg.reduction == "none":
        pass
    else:
        raise ValueError(f"Unknown reduction: {cfg.reduction}")

    return loss

