import torch

# Random Masking
def random_masking(
    roi_feat: torch.Tensor,     # [N, C, H, W]
    tau_drop: float = 0.2,        # mask threshold, [0.1, 0.3]
)-> torch.Tensor:
    ''':return: masked_roi_feat : [N, C, H, W]'''
    N, C, H, W = roi_feat.shape

    # Generate random mask
    D = torch.rand(
        (N, H, W),
        device=roi_feat.device,
        dtype=roi_feat.dtype,
    )

    D_drop = (D >= tau_drop).to(dtype=roi_feat.dtype)  # [N, H, W], 1: keep, 0: drop
    # 扩展通道维度，使其可以与 [N, C, H, W] 广播相乘
    D_drop = D_drop.unsqueeze(1)  # shape: [N, 1, H, W]

    masked_roi_feat = roi_feat * D_drop  # [N, C, H, W]

    return masked_roi_feat

# Add Gaussian Noise
def add_gaussian_noise(
    roi_feat: torch.Tensor,     # [N, C, H, W]
    sigma: float = 0.5        # standard deviation of the Gaussian noise, [0.01, 0.1]
) -> torch.Tensor:
    ''':return: noisy_roi_feat : [N, C, H, W]'''
    # Sample Gaussian noise
    noise = torch.normal(
        mean=0.0,
        std=sigma,
        size=roi_feat.shape,
        device=roi_feat.device,
        dtype=roi_feat.dtype,
    )

    noisy_roi_feat = roi_feat + noise  # [N, C, H, W]

    return noisy_roi_feat