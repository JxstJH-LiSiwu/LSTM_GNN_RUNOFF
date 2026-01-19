# src/plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union


def plot_nse_cdf_multi_model(
    nse_dict_per_model: Dict[str, Dict[Union[str, int], float]],
    save_path: str,
    *,
    xlim: tuple = (-1.0, 1.0),
    title: str = "NSE–CDF",
):
    """
    输入:
      nse_dict_per_model = {
        "LSTM": {basin_id: nse, ...},
        "LSTM-GAT": {...},
        ...
      }
    输出:
      保存 NSE-CDF 图到 save_path
    """
    plt.figure(figsize=(7.2, 5.2))

    for model_name, nse_dict in nse_dict_per_model.items():
        nse_vals = np.array(list(nse_dict.values()), dtype=np.float64)
        nse_vals = nse_vals[np.isfinite(nse_vals)]  # 去 NaN/inf
        if nse_vals.size == 0:
            continue

        nse_sorted = np.sort(nse_vals)
        n = nse_sorted.size
        cdf = np.arange(1, n + 1) / n

        plt.plot(nse_sorted, cdf, label=f"{model_name} (n={n})")

    plt.xlabel("NSE")
    plt.ylabel("Cumulative Probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(*xlim)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
