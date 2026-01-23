import numpy as np
import pandas as pd
import torch
from pathlib import Path

from dataset.lamah_dataset import LamaHDataset
from dataset.data_prepare import load_lamah_from_cache, split_time_indices, positive_robust_log_fit, positive_robust_log_inverse
from dataset.dataloader import create_dataloader
from src.lstm_gat import CombinedLSTMGATWithStatic2Hop
from src.metrics import compute_nse_per_basin  # :contentReference[oaicite:1]{index=1}


def predict_on_loader(model, dataloader, edge_index, edge_weight, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device)
            static  = batch["static"].to(device)
            target  = batch["target"].to(device)
            mask    = batch["mask"].to(device)

            pred = model(
                dynamic_features=dynamic,
                static_features=static,
                edge_index=edge_index.to(device),
                edge_weight=edge_weight.to(device),
            )
            preds.append(pred.cpu())
            targets.append(target.cpu())
            masks.append(mask.cpu())

    pred_all   = torch.cat(preds, dim=0).numpy()
    target_all = torch.cat(targets, dim=0).numpy()
    mask_all   = torch.cat(masks, dim=0).numpy().astype(bool)
    return target_all, pred_all, mask_all


def per_basin_diagnostics(obs_q, pred_q, mask, basin_ids):
    """
    obs_q, pred_q, mask: [T, N]
    return: DataFrame indexed by basin_id with NSE + hydrologic stats.
    """
    T, N = obs_q.shape
    rows = []

    for j in range(N):
        v = mask[:, j] & np.isfinite(obs_q[:, j]) & np.isfinite(pred_q[:, j])
        n_valid = int(v.sum())
        if n_valid < 30:
            continue

        y = obs_q[v, j]
        yhat = pred_q[v, j]

        ybar = float(np.mean(y))
        denom = float(np.sum((y - ybar) ** 2))  # NSE denominator
        num = float(np.sum((y - yhat) ** 2))

        # NSE（与 compute_nse_per_basin 一致的定义）
        nse = np.nan
        if np.isfinite(denom) and denom > 0:
            nse = 1.0 - (num / denom)

        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae = float(np.mean(np.abs(y - yhat)))

        # bias / variability proxies
        mean_pred = float(np.mean(yhat))
        std_obs = float(np.std(y))
        std_pred = float(np.std(yhat))
        bias_ratio = float(mean_pred / (ybar + 1e-8))
        var_ratio = float(std_pred / (std_obs + 1e-8))

        # flow regime stats
        p05 = float(np.quantile(y, 0.05))
        p50 = float(np.quantile(y, 0.50))
        p95 = float(np.quantile(y, 0.95))

        rows.append({
            "basin_id": basin_ids[j],
            "n_valid": n_valid,
            "NSE": float(nse) if np.isfinite(nse) else np.nan,
            "denom_var_sum": denom,
            "obs_mean": ybar,
            "pred_mean": mean_pred,
            "bias_ratio": bias_ratio,
            "obs_std": std_obs,
            "pred_std": std_pred,
            "var_ratio": var_ratio,
            "RMSE": rmse,
            "MAE": mae,
            "obs_p05": p05,
            "obs_p50": p50,
            "obs_p95": p95,
        })

    df = pd.DataFrame(rows).set_index("basin_id").sort_values("NSE")
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== paths (match your project) =====
    base_dir = Path(__file__).resolve().parent
    DATA_ROOT = base_dir / "dataset"
    SAVE_DIR  = base_dir / "checkpoints"
    CACHE_FILE = SAVE_DIR / "data_cache" / "lamah_daily.pt"
    CKPT_PATH  = SAVE_DIR / "lstm_gat_model_epoch.pth"

    # ===== loader config =====
    BASE_BATCH_SIZE = 2
    SEQ_LEN = 180
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    # ===== load cached data =====
    precip_df, temp_df, soil_df, runoff_df, static_df, edge_index, edge_weight = load_lamah_from_cache(CACHE_FILE)

    # ===== build test split (same as your plot script) =====
    split = split_time_indices(
        precip_df.index,
        seq_len=SEQ_LEN,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    train_indices = split["train"]
    test_indices  = split["test"]

    train_y_raw = runoff_df.iloc[train_indices + 1].values  # (T_train, N)
    runoff_scaler = positive_robust_log_fit(train_y_raw.reshape(-1))

    test_dataset = LamaHDataset(
        precip_df, temp_df, soil_df, runoff_df, static_df,
        seq_len=SEQ_LEN,
        indices=test_indices,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=BASE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    basin_ids = test_dataset.basin_ids

    # ===== load model =====
    model = CombinedLSTMGATWithStatic2Hop(
        dynamic_input_dim=3,
        static_input_dim=static_df.shape[1],
        lstm_hidden_dim=128,
        gnn_hidden_dim=64,
        output_dim=1,
        lstm_layers=2,
        gat_heads=4,
        lstm_dropout=0.35,
        gnn_dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)

    # ===== predict (log-space) =====
    obs_log, pred_log, mask = predict_on_loader(model, test_loader, edge_index, edge_weight, device)

    # ===== back to original Q =====
    obs_q  = positive_robust_log_inverse(obs_log, runoff_scaler)
    pred_q = positive_robust_log_inverse(pred_log, runoff_scaler)

    # ===== per-basin NSE dict (your existing function) =====
    nse_dict = compute_nse_per_basin(
        observed=obs_q,
        predicted=pred_q,
        mask=mask,
        basin_ids=basin_ids,
        min_valid=30,
    )

    nse_vals = np.array([v for v in nse_dict.values() if np.isfinite(v)], dtype=np.float64)
    print(f"[NSE] valid basins = {len(nse_vals)} / {len(basin_ids)}")
    if len(nse_vals) > 0:
        print(f"[NSE] mean={nse_vals.mean():.3f}, median={np.median(nse_vals):.3f}, p25={np.quantile(nse_vals,0.25):.3f}, p75={np.quantile(nse_vals,0.75):.3f}")

    # ===== diagnostics table =====
    diag = per_basin_diagnostics(obs_q, pred_q, mask, basin_ids)

    # define "bad basins" threshold
    BAD_NSE_TH = -5.0
    bad = diag[diag["NSE"] < BAD_NSE_TH].copy()
    print(f"\n[Bad basins] NSE < {BAD_NSE_TH}: {len(bad)} basins")

    # show top 20 worst
    print("\n=== Top 20 worst basins by NSE ===")
    cols_show = ["NSE", "obs_mean", "pred_mean", "bias_ratio", "obs_std", "pred_std", "var_ratio", "RMSE", "denom_var_sum", "obs_p05", "obs_p95", "n_valid"]
    print(bad[cols_show].head(20).to_string())

    # ===== static feature comparison (uses cached static_df: may be minmax-scaled) =====
    # Align static_df rows with diag index
    static_aligned = static_df.loc[static_df.index.intersection(diag.index)].copy()
    diag_aligned = diag.loc[static_aligned.index].copy()

    bad_mask = diag_aligned["NSE"] < BAD_NSE_TH
    if bad_mask.sum() >= 5:
        # compute mean difference of each static feature: bad - good
        mean_bad = static_aligned[bad_mask].mean(axis=0)
        mean_good = static_aligned[~bad_mask].mean(axis=0)
        diff = (mean_bad - mean_good).sort_values(ascending=False)

        print("\n=== Static features most different (bad - good), TOP 15 ===")
        print(diff.head(15).to_string())

        print("\n=== Static features most different (bad - good), BOTTOM 15 ===")
        print(diff.tail(15).to_string())
    else:
        print("\n[WARN] Too few bad basins to compare static features robustly.")

    # ===== save outputs =====
    out_dir = SAVE_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    diag.to_csv(out_dir / "per_basin_diagnostics.csv")
    bad.to_csv(out_dir / "bad_basins_nse_lt_-5.csv")

    print(f"\n[Saved] {out_dir/'per_basin_diagnostics.csv'}")
    print(f"[Saved] {out_dir/'bad_basins_nse_lt_-5.csv'}")


if __name__ == "__main__":
    main()
