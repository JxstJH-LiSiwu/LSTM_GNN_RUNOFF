import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset.lamah_dataset import LamaHDataset
from dataset.data_prepare import (
    load_lamah_from_cache,
    positive_robust_log_per_basin_inverse,
)
from dataset.dataloader import create_dataloader
from src.metrics import compute_nse_per_basin

# ===== model imports (alias to avoid name conflict) =====
from src.lstm_only import CombinedLSTMWithStatic2Hop as LSTMOnly
from src.lstm_gat  import CombinedLSTMWithStatic2Hop as LSTMGAT
from src.lstm_gcn  import CombinedLSTMWithStatic2Hop as LSTMGCN
from src.lstm_cheb import CombinedLSTMWithStatic2Hop as LSTMCheb
from src.lstm_sage import CombinedLSTMWithStatic2Hop as LSTMSage




# ============================================================
# Model registry (🔥 核心：增删模型只改这里)
# ============================================================
MODEL_REGISTRY = {
    "LSTM": {
        "class": LSTMOnly,
        "ckpt": "lstm_only_model_epoch.pth",
        "color": "#7f7f7f",
        "linestyle": "--",
    },
    "LSTM-GAT": {
        "class": LSTMGAT,
        "ckpt": "lstm_gat_model_epoch.pth",
        "color": "#d62728",
        "linestyle": "-",
    },
    "LSTM-GCN": {
        "class": LSTMGCN,
        "ckpt": "lstm_gcn_model_epoch.pth",
        "color": "#1f77b4",
        "linestyle": "-.",
    },
    "LSTM-Cheb": {
        "class": LSTMCheb,
        "ckpt": "lstm_cheb_model_epoch.pth",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    "LSTM-SAGE": {
        "class": LSTMSage,
        "ckpt": "lstm_sage_model_epoch.pth",
        "color": "#ff7f0e",
        "linestyle": ":",
    },
}

# 👉 控制哪些模型参与画图（Figure 4 / 5）
PLOT_MODELS = [
    "LSTM",
    "LSTM-GAT",
    "LSTM-GCN",
    "LSTM-Cheb",
    "LSTM-SAGE",
]


# ============================================================
# Prediction helper
# ============================================================
def predict_on_loader(model, dataloader, edge_index, edge_weight, device):
    model.eval()
    preds, targets, masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            static  = batch["static"].to(device, non_blocking=True)
            target  = batch["target"]
            mask    = batch["mask"]

            pred = model(
                dynamic_features=dynamic,
                forecast_features=forecast,
                static_features=static,
                edge_index=edge_index.to(device),
                edge_weight=edge_weight.to(device),
            )

            preds.append(pred.cpu())
            targets.append(target)
            masks.append(mask)

    obs_log  = torch.cat(targets, dim=0).numpy()
    pred_log = torch.cat(preds, dim=0).numpy()
    mask_all = torch.cat(masks, dim=0).numpy().astype(bool)

    return obs_log, pred_log, mask_all


# ============================================================
# Figure 4 — NSE CDF (multi-model)
# ============================================================
def plot_figure4_nse_cdf(nse_dict_all, save_path):
    plt.figure(figsize=(6.8, 5.2))

    for name in PLOT_MODELS:
        cfg = MODEL_REGISTRY[name]
        nse = np.array(
            [v for v in nse_dict_all[name].values() if np.isfinite(v)],
            dtype=np.float64,
        )
        nse.sort()
        cdf = np.arange(1, len(nse) + 1) / len(nse)

        plt.plot(
            nse,
            cdf,
            label=name,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=0.6,
        )

    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1.2)

    plt.xlabel("Nash–Sutcliffe Efficiency (NSE)", fontsize=12)
    plt.ylabel("Cumulative fraction of basins", fontsize=12)
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ============================================================
# Figure 5 — Scatter plots (select one model)
# ============================================================
def plot_figure5_scatter(obs_q, pred_q, mask, model_name, out_dir, plot_loglog=True):
    """
    obs_q, pred_q : (T, N)
    mask          : (T, N) boolean
    """

    # ======================================================
    # Step 1: flatten valid points (ONLY ONCE)
    # ======================================================
    valid = (
        mask
        & np.isfinite(obs_q)
        & np.isfinite(pred_q)
    )

    x = obs_q[valid].astype(np.float64)   # (K,)
    y = pred_q[valid].astype(np.float64)  # (K,)

    # ======================================================
    # Figure 5a: linear scale
    # ======================================================
    maxv = np.nanpercentile(x, 99.9)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=6, alpha=0.25)
    plt.plot([0, maxv], [0, maxv], "k--", linewidth=1.4)
    plt.xlabel("Observed Streamflow (m³/s)")
    plt.ylabel("Predicted Streamflow (m³/s)")
    plt.title(model_name)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"figure5_scatter_linear_{model_name}.png", dpi=300)
    plt.close()

    if plot_loglog:
        # ======================================================
        # Figure 5b: log–log scale
        #   (filter positives on 1D arrays)
        # ======================================================
        pos = (x > 0) & (y > 0)
        lx = np.log1p(x[pos])
        ly = np.log1p(y[pos])

        maxv = np.nanpercentile(lx, 99.9)

        plt.figure(figsize=(6, 6))
        plt.scatter(lx, ly, s=6, alpha=0.25)
        plt.plot([0, maxv], [0, maxv], "k--", linewidth=1.4)
        plt.xlabel("log1p(Observed Streamflow)")
        plt.ylabel("log1p(Predicted Streamflow)")
        plt.title(f"{model_name} (log–log)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"figure5_scatter_loglog_{model_name}.png", dpi=300)
        plt.close()


def plot_figure5a_multi_scatter(predictions, out_dir):
    plt.figure(figsize=(6, 6))
    maxv = 0.0

    for name in PLOT_MODELS:
        obs_q, pred_q, mask = predictions[name]
        valid = (
            mask
            & np.isfinite(obs_q)
            & np.isfinite(pred_q)
        )

        x = obs_q[valid].astype(np.float64)
        y = pred_q[valid].astype(np.float64)

        if x.size == 0:
            continue

        maxv = max(maxv, np.nanpercentile(x, 99.9))
        plt.scatter(
            x,
            y,
            s=6,
            alpha=0.2,
            label=name,
            color=MODEL_REGISTRY[name]["color"],
        )

    plt.plot([0, maxv], [0, maxv], "k--", linewidth=1.4)
    plt.xlabel("Observed Streamflow (m³/s)")
    plt.ylabel("Predicted Streamflow (m³/s)")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "figure5_scatter_linear_multi_model.png", dpi=300)
    plt.close()



# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    SAVE_DIR   = base_dir / "checkpoints"
    CACHE_FILE = SAVE_DIR / "data_cache" / "lamah_daily.pt"
    OUT_DIR    = SAVE_DIR / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEQ_LEN = 180
    BATCH_SIZE = 4
    MIN_VALID = 30

    # ----- load cache -----
    cache = load_lamah_from_cache(CACHE_FILE)

    precip_df  = cache["precip_df"]
    temp_df    = cache["temp_df"]
    soil_df    = cache["soil_df"]
    runoff_df  = cache["runoff_df"]
    static_df  = cache["static_df"]
    edge_index = cache["edge_index"]
    edge_weight = cache["edge_weight"]
    edge_weight_raw = cache.get("edge_weight_raw", edge_weight)
    split = cache["split"]

    basin_ids = list(static_df.index)
    runoff_median = (
        cache["meta"]["scalers"]["runoff"]["median_per_basin"]
        .loc[basin_ids]
        .to_numpy(dtype=np.float64)
    )

    test_dataset = LamaHDataset(
        precip_df, temp_df, soil_df, runoff_df, static_df,
        seq_len=SEQ_LEN,
        lead_days=1,
        indices=split["test"],
        sample_weights=None,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # ========================================================
    # Loop over models
    # ========================================================
    nse_dict_all = {}
    predictions = {}

    raw_edge_models = {"LSTM-GAT", "LSTM-GCN", "LSTM-Cheb"}

    for name in PLOT_MODELS:
        cfg = MODEL_REGISTRY[name]
        edge_weight_use = edge_weight_raw if name in raw_edge_models else edge_weight

        print(f"\n[Eval] {name}")
        model = cfg["class"](
            dynamic_input_dim=3,
            static_input_dim=static_df.shape[1],
            forecast_input_dim=2,
            lstm_hidden_dim=128,
            gnn_hidden_dim=64,
            output_dim=1,
            lstm_layers=2,
            gat_heads=4,
            lstm_dropout=0.35,
            gnn_dropout=0.2,
            cheb_k=3
        ).to(device)

        ckpt_path = SAVE_DIR / cfg["ckpt"]
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        obs_log, pred_log, mask = predict_on_loader(
            model, test_loader, edge_index, edge_weight_use, device
        )

        obs_q  = positive_robust_log_per_basin_inverse(obs_log, runoff_median)
        pred_q = positive_robust_log_per_basin_inverse(pred_log, runoff_median)

        nse_dict = compute_nse_per_basin(
            obs_q, pred_q, mask, basin_ids, MIN_VALID
        )

        nse_dict_all[name] = nse_dict
        predictions[name] = (obs_q, pred_q, mask)

    # ========================================================
    # Figure 4
    # ========================================================
    plot_figure4_nse_cdf(
        nse_dict_all,
        save_path=OUT_DIR / "figure4_nse_cdf_multi_model.png",
    )

    # ========================================================
    # Figure 5 (pick best model, e.g. LSTM-GAT)
    # ========================================================
    plot_figure5a_multi_scatter(predictions, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
