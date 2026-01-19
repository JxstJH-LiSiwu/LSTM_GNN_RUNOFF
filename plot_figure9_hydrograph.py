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




# ============================================================
# Predict helper (identical interface for both models)
# ============================================================
def predict_on_loader(model, dataloader, edge_index, edge_weight, device):
    model.eval()
    preds = []
    targets = []
    masks = []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            static  = batch["static"].to(device, non_blocking=True)
            target  = batch["target"]   # CPU
            mask    = batch["mask"]     # CPU

            pred = model(
                dynamic_features=dynamic,
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
# Figure 9 plotting
# ============================================================
def plot_hydrograph_comparison(
    time_idx,
    obs,
    pred_lstm,
    pred_gat,
    nse_lstm,
    nse_gat,
    station_id,
    save_path=None,
):
    plt.plot(time_idx, obs, color="black", linewidth=1.8, label="Observed")
    plt.plot(
        time_idx,
        pred_lstm,
        linestyle="--",
        color="#1f77b4",
        linewidth=1.6,
        label="LSTM",
    )
    plt.plot(
        time_idx,
        pred_gat,
        color="#d62728",
        linewidth=1.6,
        label="LSTM-GAT",
    )

    plt.title(f"Station {station_id}", fontsize=11)
    plt.ylabel("Streamflow (m³/s)")

    txt = (
        f"LSTM NSE = {nse_lstm:.3f}\n"
        f"LSTM-GAT NSE = {nse_gat:.3f}"
    )
    plt.text(
        0.99,
        0.97,
        txt,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=9)


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    SAVE_DIR   = Path("/home/lisiwu/jxwork/1-gnn-lstm/checkpoints")
    CACHE_FILE = SAVE_DIR / "data_cache" / "lamah_daily.pt"

    CKPT_LSTM     = SAVE_DIR / "lstm_only_model_epoch.pth"
    CKPT_LSTM_GAT = SAVE_DIR / "lstm_gat_model_epoch.pth"

    OUT_DIR = SAVE_DIR / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Params
    # --------------------------------------------------------
    SEQ_LEN = 180
    BATCH_SIZE = 4
    MIN_VALID = 30

    TARGET_STATIONS = ["ID_532", "ID_530", "ID_533"]

    # --------------------------------------------------------
    # Load data
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
    basin_id_to_idx = {bid: i for i, bid in enumerate(basin_ids)}

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
    # Models
    # --------------------------------------------------------
    from src.lstm_only import CombinedLSTMWithStatic2Hop

    lstm = CombinedLSTMWithStatic2Hop(
        dynamic_input_dim=3,
        static_input_dim=static_df.shape[1],
        lstm_hidden_dim=128,
        gnn_hidden_dim=64,
        output_dim=1,
        lstm_layers=2,
        gat_heads=4,
        lstm_dropout=0.35,
        gnn_dropout=0.2,
        cheb_k=3
    ).to(device)

    from src.lstm_gat import CombinedLSTMWithStatic2Hop

    gat = CombinedLSTMWithStatic2Hop(
        dynamic_input_dim=3,
        static_input_dim=static_df.shape[1],
        lstm_hidden_dim=128,
        gnn_hidden_dim=64,
        output_dim=1,
        lstm_layers=2,
        gat_heads=4,
        lstm_dropout=0.35,
        gnn_dropout=0.2,
        cheb_k=3
    ).to(device)

    lstm.load_state_dict(torch.load(CKPT_LSTM, map_location=device))
    gat.load_state_dict(torch.load(CKPT_LSTM_GAT, map_location=device))

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
    # NSE per basin
    # --------------------------------------------------------
    nse_lstm = compute_nse_per_basin(
        obs_q, pred_lstm_q, mask, basin_ids, MIN_VALID
    )
    nse_gat = compute_nse_per_basin(
        obs_q, pred_gat_q, mask, basin_ids, MIN_VALID
    )

    # --------------------------------------------------------
    # Plot Figure 9
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(10.5, 6.8), sharex=True
    )

    time_idx = np.arange(obs_q.shape[0])

    for ax, station_id in zip(axes, TARGET_STATIONS):
        idx = basin_id_to_idx[station_id]

        plt.sca(ax)
        plot_hydrograph_comparison(
            time_idx,
            obs_q[:, idx],
            pred_lstm_q[:, idx],
            pred_gat_q[:, idx],
            nse_lstm[station_id],
            nse_gat[station_id],
            station_id,
        )

    axes[-1].set_xlabel("Time step")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure9_hydrograph_lstm_vs_gat.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
