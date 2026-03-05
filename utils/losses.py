"""
Copyright：Part of this code is adapted and modified from the official implementation of:
MuSCLe: A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation
GitHub repository: https://github.com/SCoulY/MuSCLe
"""
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops.boxes import generalized_box_iou

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
class SupConLossConfig:
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
    cfg: SupConLossConfig = SupConLossConfig(),
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

# ----- Patch Loss -----
def get_patch_loss(
    patch_logits : torch.Tensor,  # [R * k, num_classes]
    wb_labels : torch.Tensor,      # [R, num_classes]
)->torch.Tensor:
    '''return loss_patch'''
    k = patch_logits.shape[0] // wb_labels.shape[0]  # number of patches per weak box
    targets = wb_labels.argmax(dim=1).repeat_interleave(k)
    loss_patch = F.cross_entropy(patch_logits, targets)

    return loss_patch

# ----- CCAM Loss -----
def cos_sim(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)

def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim

def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# ----- Object Loss -----
def get_object_loss(
    proposal_obj_logits : torch.Tensor,  # [N_s = total_num_sparse_proposals, C = num_classes]
    bg_proposal_obj_logits : torch.Tensor,  # [N_b = total_num_bg_proposals, C = num_classes]
    proposal_labels : torch.Tensor,  # [N_s], dtype long, range [1, C]
    fg_scores : torch.Tensor, # [N_s] float, range [0, 1]
)-> torch.Tensor:
    """
    compute object loss
    :return:
    loss_obj
    """
    Ns = proposal_obj_logits.shape[0]
    Nb = bg_proposal_obj_logits.shape[0]
    total = Ns + Nb

    # compute loss_sparse
    ce_per = F.cross_entropy(proposal_obj_logits, proposal_labels, reduction='none')  # [N_s]
    denom = fg_scores.sum().clamp_min(1e-12)
    loss_sparse = (fg_scores * ce_per).sum() / denom

    # compute loss_bg
    bg_labels = torch.zeros(Nb, dtype=torch.long, device=bg_proposal_obj_logits.device)  # background class index = 0
    loss_bg = F.cross_entropy(bg_proposal_obj_logits, bg_labels)

    # compute loss_obj
    loss_obj = (Ns * loss_sparse + Nb * loss_bg) / total

    return loss_obj

# ----- Constrain Loss -----
def get_constrain_loss(
    proposal_embeddings: torch.Tensor,  # Normalized, [N = total_num_sparse_proposals, D = embed_dim]
    prototypes_embeddings: torch.Tensor,  # Normalized, [C = num_classes, D]
    bg_prototype_embedding : torch.Tensor,  # Normalized, [D]
    labels : torch.Tensor, # [N], dtype long, range [1, C]
    fg_scores : torch.Tensor, # [N] float, range [0, 1]
    w_proto_loss : float,
    w_pull_loss : float,
    w_push_loss : float,
    tau : float = 0.07,
    push_margin : float  = 0.1,
    push_tau : float = 0.2
)-> Dict[str, torch.Tensor]:
    """
    compute constrain loss
    :return:
    constrain_loss_dict = {
        "loss_constrain" : total_loss,
        "loss_proto" : loss_proto,
        "loss_pull" : loss_pull,
        "loss_push" : loss_push,
    }
    """
    device = proposal_embeddings.device
    N = proposal_embeddings.shape[0]
    C = prototypes_embeddings.shape[0]
    z = proposal_embeddings  # [N, D]
    p = prototypes_embeddings  # [C, D]
    bg_p = bg_prototype_embedding.view(1, -1)  # [1, D]
    labels = (labels - 1).clamp(min=0, max=C - 1)    # shift to range [0, C - 1], shape [N]
    a = fg_scores  # [N]

    # compute loss_proto
    logits_fg = (z @ p.t()) / tau  # [N, C]
    log_prob_fg = F.log_softmax(logits_fg, dim=1)  # [N, C]
    log_p_true = log_prob_fg.gather(1, labels.view(-1, 1)).squeeze(1)  # [N]
    loss_proto = -(a * log_p_true).mean()

    # compute loss_pull
    logits_bg = (z @ bg_p.t()).squeeze(1) / tau  # [N]
    logits_all = torch.cat([logits_bg.view(-1, 1), logits_fg], dim=1)  # [N, 1+C]
    log_prob_all = F.log_softmax(logits_all, dim=1)  # [N, 1+C]
    log_p_bg = log_prob_all[:, 0]  # [N]
    loss_pull = -((1.0 - a) * log_p_bg).mean()

    # compute loss_push
    # exp_arg = torch.clamp(logits_bg, max=50.0)
    # loss_push = (a * torch.exp(exp_arg)).mean()
    # softplus version
    s_bg = (z @ bg_p.t()).squeeze(1)  # [N]
    loss_push = (a * F.softplus((s_bg - push_margin) / push_tau)).mean()

    # total
    total_loss = (w_proto_loss * loss_proto) + (w_pull_loss * loss_pull) + (w_push_loss * loss_push)

    return {
        "loss_constrain": total_loss,
        "loss_proto": loss_proto,
        "loss_pull": loss_pull,
        "loss_push": loss_push,
    }

# ----- Match Loss -----
def _softmax_focal_loss(
    logits: torch.Tensor,  # [N, C+1]
    targets: torch.Tensor,  # [N] in [0..C] (C=background)
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)  # [N, C+1]
    probs = torch.softmax(logits, dim=-1)  # [N, C+1]

    idx = torch.arange(logits.size(0), device=logits.device)
    pt = probs[idx, targets]  # [N]
    logpt = log_probs[idx, targets]  # [N]

    loss = - (alpha * (1.0 - pt).pow(gamma) * logpt)  # [N]
    return loss.mean()

def compute_match_loss(
    matched_pairs: List[Tuple[Dict, Dict]],   # (det_dict, aug_seed_dict)
    unmatched_dets: List[Dict],               # det_dict list, treated as background
    imgs: torch.Tensor,                       # [B,C,H,W]
    num_classes: int,
    match_focal_alpha: float,
    match_focal_gamma: float,
    lambda_match_cls: float,
    lambda_match_l1: float,
    lambda_match_giou: float,
    other_class_logit: float = -20.0,
    bg_logit: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Match loss for one-to-one matching between:
      - detections (pred): det_dict from _match_with_hungarian
      - aug seeds (tgt): aug_seed_dict

    det_dict format:
      {
        "sp_idx": int,
        "box": Tensor[5] (batch_idx,x1,y1,x2,y2),
        "class_id": int in [0..C-1],
        "score": Tensor scalar,
        "logit": Tensor scalar (logit(score) proxy)
      }

    aug_seed_dict format:
      {
        "box": Tensor[5],
        "class_id": int in [0..C-1],
        "score": float or Tensor (optional),
        "logit": Tensor scalar (optional)
      }
    """
    device = imgs.device
    C = num_classes
    bg_id = C  # background index in [0..C]

    # --------------------------
    # (0) collect all detections participating in match loss
    #     matched dets are positives (target = seed class)
    #     unmatched dets are negatives (target = background)
    # --------------------------
    det_all: List[Dict] = [p[0] for p in matched_pairs] + list(unmatched_dets)
    N = len(det_all)
    if N == 0:
        z = torch.tensor(0.0, device=device)
        return {"loss_match": z, "loss_match_cls": z, "loss_match_l1": z, "loss_match_giou": z}

    # --------------------------
    # (1) classification loss (Softmax focal over C+1 classes)
    # --------------------------
    # targets: matched -> seed class, unmatched -> background
    cls_targets = torch.full((N,), bg_id, device=device, dtype=torch.long)
    for k, (det, seed) in enumerate(matched_pairs):
        cls_targets[k] = int(seed["class_id"])  # [0..C-1]

    # build logits: [N, C+1]
    # - all foreground channels default other_class_logit
    # - put det["logit"] at its predicted class channel
    # - background channel = bg_logit
    cls_logits = torch.full((N, C + 1), other_class_logit, device=device, dtype=torch.float32)
    cls_logits[:, bg_id] = float(bg_logit)

    for i, det in enumerate(det_all):
        pred_c = int(det["class_id"])
        lg = det.get("logit", None)
        if lg is None:
            # if logit missing, derive from score
            s = det.get("score", 1.0)
            if not torch.is_tensor(s):
                s = torch.tensor(float(s), device=device, dtype=torch.float32)
            else:
                s = s.to(device=device, dtype=torch.float32)
            lg = torch.log(s.clamp(1e-6, 1 - 1e-6))
        else:
            if torch.is_tensor(lg):
                lg = lg.to(device=device, dtype=torch.float32)
            else:
                lg = torch.tensor(float(lg), device=device, dtype=torch.float32)

        if 0 <= pred_c < C:
            cls_logits[i, pred_c] = lg

    loss_cls = _softmax_focal_loss(
        logits=cls_logits,
        targets=cls_targets,
        alpha=match_focal_alpha,
        gamma=match_focal_gamma,
    )

    # --------------------------
    # (2) regression loss (only for matched pairs)
    # --------------------------
    K = len(matched_pairs)
    if K == 0:
        loss_l1 = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
    else:
        det_boxes5 = torch.stack([p[0]["box"] for p in matched_pairs], dim=0).to(device=device)
        seed_boxes5 = torch.stack([p[1]["box"] for p in matched_pairs], dim=0).to(device=device)

        det_xyxy = det_boxes5[:, 1:5].float()
        seed_xyxy = seed_boxes5[:, 1:5].float()

        H, W = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([W, H, W, H], device=device, dtype=torch.float32).unsqueeze(0)

        loss_l1 = F.l1_loss(det_xyxy / scale, seed_xyxy / scale, reduction="none").sum(dim=1).mean()

        giou = generalized_box_iou(det_xyxy, seed_xyxy)  # [K,K]
        loss_giou = (1.0 - giou.diag()).mean()

    # --------------------------
    # (3) weighted sum
    # --------------------------
    lam_cls = lambda_match_cls
    lam_l1 = lambda_match_l1
    lam_giou = lambda_match_giou

    loss_match = lam_cls * loss_cls + lam_l1 * loss_l1 + lam_giou * loss_giou

    return {
        "loss_match": loss_match,
        "loss_match_cls": loss_cls,
        "loss_match_l1": loss_l1,
        "loss_match_giou": loss_giou,
    }
