import torch


def masked_mse_loss(pred, target, mask):
    """
    pred:   [B, N]
    target: [B, N]
    mask:   [B, N] (bool)

    returns: scalar loss
    """

    # 只在 valid 节点上计算误差
    diff = pred - target
    diff = diff[mask]

    if diff.numel() == 0:
        # 极端情况：一个 batch 没有 valid 点
        return torch.tensor(0.0, device=pred.device)

    return torch.mean(diff ** 2)


def nse(pred, target, mask, eps=1e-6):
    """
    pred, target: [B, N]
    mask: [B, N] (bool)
    """
    pred = pred[mask]
    target = target[mask]

    if pred.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)

    num = torch.sum((pred - target) ** 2)
    den = torch.sum((target - target.mean()) ** 2) + eps
    return 1.0 - num / den
