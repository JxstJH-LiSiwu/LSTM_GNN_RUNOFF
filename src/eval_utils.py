# src/eval_utils.py
import numpy as np
import torch
from typing import Tuple


@torch.no_grad()
def predict_on_loader(
    model: torch.nn.Module,
    dataloader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 dataloader 上跑完整个数据集（通常是 test_loader），汇总：
      observed:  [T_total, N]
      predicted: [T_total, N]
      mask:      [T_total, N]  (bool)

    注意：这里 observed/predicted 默认是你 Dataset 里 target 的尺度（当前为 log1p(q)）:contentReference[oaicite:1]{index=1}
    """
    model.eval()

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    obs_list, pred_list, mask_list = [], [], []

    for batch in dataloader:
        dynamic = batch["dynamic"].to(device)  # [B, seq_len, N, F]
        static  = batch["static"].to(device)   # [B, N, D]  or [N, D]（你现在是 [B,N,D]）:contentReference[oaicite:2]{index=2}
        target  = batch["target"].to(device)   # [B, N]
        mask    = batch["mask"].to(device)     # [B, N] bool

        pred = model(
            dynamic_features=dynamic,
            static_features=static,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )  # [B, N] :contentReference[oaicite:3]{index=3}

        obs_list.append(target.detach().cpu())
        pred_list.append(pred.detach().cpu())
        mask_list.append(mask.detach().cpu())

    observed = torch.cat(obs_list, dim=0).numpy()     # [T_total, N]
    predicted = torch.cat(pred_list, dim=0).numpy()   # [T_total, N]
    mask = torch.cat(mask_list, dim=0).numpy().astype(bool)

    return observed, predicted, mask
