# ============================================================
# Figure 6: Spatial distribution of ΔNSE / ΔKGE
# (LSTM-GAT minus LSTM)
# ============================================================

import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

from dataset.lamah_dataset import LamaHDataset
from dataset.data_prepare import (
    load_lamah_from_cache,
    positive_robust_log_per_basin_inverse,
)
from dataset.dataloader import create_dataloader
from src.metrics import (
    compute_nse_per_basin,
    compute_kge_per_basin,
)

# ============================================================
# Predict helper (same interface as你 Figure 9 用的)
# ============================================================
def predict_on_loader(model, dataloader, edge_index, edge_weight, device):
    model.eval()
    preds, targets, masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            static  = batch["static"].to(device, non_blocking=True)
            target  = batch["target"]   # CPU
            mask    = batch["mask"]     # CPU

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
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    BASE = Path(__file__).resolve().parent

    CACHE_FILE = BASE / "checkpoints/data_cache/lamah_daily.pt"
    CKPT_LSTM  = BASE / "checkpoints/lstm_only_model_epoch.pth"
    CKPT_GAT   = BASE / "checkpoints/lstm_gat_model_epoch.pth"

    GAUGE_SHP = BASE / "dataset/D_gauges/3_shapefiles/Gauges.shp"

    OUT_DIR = BASE / "checkpoints/analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Params
    # --------------------------------------------------------
    SEQ_LEN = 180
    BATCH_SIZE = 4
    MIN_VALID = 30
    DELTA_TH = 0.05
    DPI = 600

    # --------------------------------------------------------
    # Load LamaH cache
    # --------------------------------------------------------
    cache = load_lamah_from_cache(CACHE_FILE)

    precip_df  = cache["precip_df"]
    temp_df    = cache["temp_df"]
    soil_df    = cache["soil_df"]
    runoff_df  = cache["runoff_df"]
    static_df  = cache["static_df"]
    edge_index = cache["edge_index"]
    edge_weight = cache["edge_weight"]
    split = cache["split"]

    basin_ids = list(static_df.index)

    runoff_median = (
        cache["meta"]["scalers"]["runoff"]["median_per_basin"]
        .loc[basin_ids]
        .to_numpy(dtype=np.float64)
    )

    # --------------------------------------------------------
    # Test loader
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------
    from src.lstm_only import CombinedLSTMWithStatic2Hop
    from src.lstm_gat import CombinedLSTMWithStatic2Hop as LSTM_GAT

    lstm = CombinedLSTMWithStatic2Hop(
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
        cheb_k=3,
    ).to(device)

    gat = LSTM_GAT(
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
        cheb_k=3,
    ).to(device)

    lstm.load_state_dict(torch.load(CKPT_LSTM, map_location=device))
    gat.load_state_dict(torch.load(CKPT_GAT, map_location=device))

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    obs_log, pred_lstm_log, mask = predict_on_loader(
        lstm, test_loader, edge_index, edge_weight, device
    )
    _, pred_gat_log, _ = predict_on_loader(
        gat, test_loader, edge_index, edge_weight, device
    )

    obs_q = positive_robust_log_per_basin_inverse(obs_log, runoff_median)
    pred_lstm_q = positive_robust_log_per_basin_inverse(pred_lstm_log, runoff_median)
    pred_gat_q  = positive_robust_log_per_basin_inverse(pred_gat_log, runoff_median)

    # --------------------------------------------------------
    # NSE / KGE per basin
    # --------------------------------------------------------
    nse_lstm = compute_nse_per_basin(
        obs_q, pred_lstm_q, mask, basin_ids, MIN_VALID
    )
    nse_gat = compute_nse_per_basin(
        obs_q, pred_gat_q, mask, basin_ids, MIN_VALID
    )

    kge_lstm = compute_kge_per_basin(
        obs_q, pred_lstm_q, mask, basin_ids, MIN_VALID
    )
    kge_gat = compute_kge_per_basin(
        obs_q, pred_gat_q, mask, basin_ids, MIN_VALID
    )

    # --------------------------------------------------------
    # Build delta dataframe
    # --------------------------------------------------------
    rows = []
    for bid in basin_ids:
        if bid not in nse_lstm or bid not in nse_gat:
            continue

        dnse = nse_gat[bid] - nse_lstm[bid]
        dkge = kge_gat[bid] - kge_lstm[bid]

        def classify(d):
            if d > DELTA_TH:
                return "gat_better"
            elif d < -DELTA_TH:
                return "lstm_better"
            else:
                return "no_diff"

        rows.append({
            "basin_id": bid,
            "delta_nse": dnse,
            "delta_kge": dkge,
            "cls_nse": classify(dnse),
            "cls_kge": classify(dkge),
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Load gauge shapefile
    # --------------------------------------------------------
    gdf = gpd.read_file(GAUGE_SHP)

    # 统一 ID 类型，兼容数值型或 "ID_###" 形式
    gdf["ID"] = gdf["ID"].astype(str)
    if not gdf["ID"].str.startswith("ID_").any():
        gdf["ID"] = "ID_" + gdf["ID"]

    gdf = gdf.merge(
        df, left_on="ID", right_on="basin_id", how="left"
    )

    # --------------------------------------------------------
    # Plot function
    # --------------------------------------------------------
    def plot_panel(ax, cls_col, title):
        colors = {
            "gat_better": "#d62728",   # red
            "lstm_better": "#1f77b4",  # blue
            "no_diff": "white",
        }

        for cls, c in colors.items():
            sub = gdf[gdf[cls_col] == cls]
            sub.plot(
                ax=ax,
                color=c,
                markersize=18,
                edgecolor="black",
                linewidth=0.3,
                label=cls,
            )

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # --------------------------------------------------------
    # Figure 6
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(9, 10), constrained_layout=True
    )

    plot_panel(axes[0], "cls_nse", "ΔNSE (LSTM-GAT − LSTM)")
    plot_panel(axes[1], "cls_kge", "ΔKGE (LSTM-GAT − LSTM)")

    axes[0].legend(
        frameon=False,
        loc="lower left",
        fontsize=9,
    )

    out = OUT_DIR / "figure6_spatial_delta_nse_kge.png"
    plt.savefig(out, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Figure 6 saved to: {out}")


if __name__ == "__main__":
    main()
