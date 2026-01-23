# src/metrics.py
import numpy as np
from typing import Dict, Optional, Sequence, Union


def compute_nse_per_basin(
    observed: np.ndarray,          # [T, N]
    predicted: np.ndarray,         # [T, N]
    mask: Optional[np.ndarray] = None,     # [T, N]  True/1=valid
    basin_ids: Optional[Sequence[Union[str, int]]] = None,  # len=N
    min_valid: int = 10,
) -> Dict[Union[str, int], float]:
    """
    逐流域计算 NSE (Nash–Sutcliffe Efficiency)

    NSE = 1 - sum((y - yhat)^2) / sum((y - ybar)^2)

    - 正确处理 NaN
    - 支持 mask（mask=None 则认为全有效）
    - 有效点过少 或 分母为0 -> np.nan
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    if obs.shape != pred.shape:
        raise ValueError(f"observed/predicted shape mismatch: {obs.shape} vs {pred.shape}")
    if obs.ndim != 2:
        raise ValueError(f"expected 2D arrays [T,N], got {obs.ndim}D")

    T, N = obs.shape

    if basin_ids is None:
        basin_ids = list(range(N))
    if len(basin_ids) != N:
        raise ValueError(f"basin_ids length mismatch: {len(basin_ids)} vs N={N}")

    if mask is None:
        valid = np.isfinite(obs) & np.isfinite(pred)
    else:
        m = np.asarray(mask).astype(bool)
        if m.shape != (T, N):
            raise ValueError(f"mask shape mismatch: {m.shape} vs {(T, N)}")
        valid = m & np.isfinite(obs) & np.isfinite(pred)

    out: Dict[Union[str, int], float] = {}
    for j in range(N):
        v = valid[:, j]
        if v.sum() < min_valid:
            out[basin_ids[j]] = np.nan
            continue

        y = obs[v, j]
        yhat = pred[v, j]

        ybar = np.mean(y)
        denom = np.sum((y - ybar) ** 2)
        if not np.isfinite(denom) or denom <= 0:
            out[basin_ids[j]] = np.nan
            continue

        num = np.sum((y - yhat) ** 2)
        nse = 1.0 - (num / denom)
        out[basin_ids[j]] = float(nse) if np.isfinite(nse) else np.nan

    return out


def compute_kge_per_basin(
    observed: np.ndarray,          # [T, N]
    predicted: np.ndarray,         # [T, N]
    mask: Optional[np.ndarray] = None,     # [T, N] True = valid
    basin_ids: Optional[Sequence[Union[str, int]]] = None,  # len = N
    min_valid: int = 10,
) -> Dict[Union[str, int], float]:
    """
    逐流域计算 KGE (Kling–Gupta Efficiency)

    KGE = 1 - sqrt( (r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2 )

    where:
        r     = Pearson correlation
        alpha = std(pred) / std(obs)
        beta  = mean(pred) / mean(obs)

    特点：
    - 与 compute_nse_per_basin 接口完全一致
    - 正确处理 NaN / Inf
    - 支持 mask
    - 有效样本过少 / 方差为 0 / 均值为 0 → np.nan
    """

    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    if obs.shape != pred.shape:
        raise ValueError(f"observed/predicted shape mismatch: {obs.shape} vs {pred.shape}")
    if obs.ndim != 2:
        raise ValueError(f"expected 2D arrays [T,N], got {obs.ndim}D")

    T, N = obs.shape

    if basin_ids is None:
        basin_ids = list(range(N))
    if len(basin_ids) != N:
        raise ValueError(f"basin_ids length mismatch: {len(basin_ids)} vs N={N}")

    if mask is None:
        valid = np.isfinite(obs) & np.isfinite(pred)
    else:
        m = np.asarray(mask).astype(bool)
        if m.shape != (T, N):
            raise ValueError(f"mask shape mismatch: {m.shape} vs {(T, N)}")
        valid = m & np.isfinite(obs) & np.isfinite(pred)

    out: Dict[Union[str, int], float] = {}

    for j in range(N):
        v = valid[:, j]
        if v.sum() < min_valid:
            out[basin_ids[j]] = np.nan
            continue

        y = obs[v, j]
        yhat = pred[v, j]

        mu_y = np.mean(y)
        mu_yhat = np.mean(yhat)

        # mean = 0 → beta 无定义
        if not np.isfinite(mu_y) or mu_y == 0.0:
            out[basin_ids[j]] = np.nan
            continue

        std_y = np.std(y, ddof=0)
        std_yhat = np.std(yhat, ddof=0)

        # std = 0 → r / alpha 无定义
        if std_y <= 0.0 or not np.isfinite(std_y):
            out[basin_ids[j]] = np.nan
            continue

        # r: Pearson correlation
        r = np.corrcoef(y, yhat)[0, 1]
        if not np.isfinite(r):
            out[basin_ids[j]] = np.nan
            continue

        alpha = std_yhat / std_y
        beta = mu_yhat / mu_y

        kge = 1.0 - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

        out[basin_ids[j]] = float(kge) if np.isfinite(kge) else np.nan

    return out
