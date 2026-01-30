import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataset.lamah_dataset import LamaHDataset
from dataset.data_prepare import load_lamah_from_cache, positive_robust_log_per_basin_inverse
from dataset.dataloader import create_dataloader
from src.metrics import compute_nse_per_basin, compute_kge_per_basin
from src import MODEL_REGISTRY


def predict_on_loader(model, dataloader, edge_index, edge_weight, device):
    model.eval()
    preds, targets, masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            static = batch["static"].to(device, non_blocking=True)
            target = batch["target"]
            mask = batch["mask"]

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

    obs_log = torch.cat(targets, dim=0).numpy()
    pred_log = torch.cat(preds, dim=0).numpy()
    mask_all = torch.cat(masks, dim=0).numpy().astype(bool)

    return obs_log, pred_log, mask_all


def summarize_metric(metric_dict):
    vals = np.array([v for v in metric_dict.values() if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "count": 0,
            "frac_gt0": np.nan,
            "frac_gt05": np.nan,
        }
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "count": int(vals.size),
        "frac_gt0": float(np.mean(vals > 0.0)),
        "frac_gt05": float(np.mean(vals > 0.5)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = Path(__file__).resolve().parent
    save_dir = base / "checkpoints"
    cache_file = save_dir / "data_cache" / "lamah_daily.pt"

    cache = load_lamah_from_cache(cache_file)

    precip_df = cache["precip_df"]
    temp_df = cache["temp_df"]
    soil_df = cache["soil_df"]
    runoff_df = cache["runoff_df"]
    static_df = cache["static_df"]
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
        seq_len=180,
        lead_days=1,
        indices=split["test"],
        sample_weights=None,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    raw_edge_models = {"LSTM-GAT", "LSTM-GCN", "LSTM-Cheb"}

    rows = []
    for model_name, cfg in MODEL_REGISTRY.items():
        base_ckpt = Path(cfg["ckpt"])
        base_stem = base_ckpt.stem

        ckpts = []
        default_ckpt = save_dir / base_ckpt.name
        if default_ckpt.exists():
            ckpts.append(("default", default_ckpt))

        prefix = f"{base_stem}_"
        for path in sorted(save_dir.glob(f"{base_stem}_*.pth")):
            name = path.name
            if not name.startswith(prefix) or not name.endswith(".pth"):
                continue
            tag = name[len(prefix):-4]
            ckpts.append((tag, path))

        if not ckpts:
            continue

        edge_weight_use = edge_weight_raw if model_name in raw_edge_models else edge_weight

        for tag, ckpt_path in ckpts:
            model = cfg["class"](
                dynamic_input_dim=3,
                static_input_dim=static_df.shape[1],
                forecast_input_dim=2,
                lstm_hidden_dim=128,
                gnn_hidden_dim=64,
                output_dim=1,
                lstm_layers=2,
                gat_heads=4,
                lstm_dropout=0.2,
                gnn_dropout=0.2,
                cheb_k=3,
            ).to(device)

            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            obs_log, pred_log, mask = predict_on_loader(
                model, test_loader, edge_index, edge_weight_use, device
            )

            obs_q = positive_robust_log_per_basin_inverse(obs_log, runoff_median)
            pred_q = positive_robust_log_per_basin_inverse(pred_log, runoff_median)

            nse = compute_nse_per_basin(obs_q, pred_q, mask, basin_ids, min_valid=30)
            kge = compute_kge_per_basin(obs_q, pred_q, mask, basin_ids, min_valid=30)

            nse_stats = summarize_metric(nse)
            kge_stats = summarize_metric(kge)

            rows.append({
                "model": model_name,
                "tag": tag,
                "ckpt": str(ckpt_path),
                "nse_mean": nse_stats["mean"],
                "nse_median": nse_stats["median"],
                "nse_p25": nse_stats["p25"],
                "nse_p75": nse_stats["p75"],
                "nse_count": nse_stats["count"],
                "nse_frac_gt0": nse_stats["frac_gt0"],
                "nse_frac_gt05": nse_stats["frac_gt05"],
                "kge_mean": kge_stats["mean"],
                "kge_median": kge_stats["median"],
                "kge_p25": kge_stats["p25"],
                "kge_p75": kge_stats["p75"],
                "kge_count": kge_stats["count"],
                "kge_frac_gt0": kge_stats["frac_gt0"],
                "kge_frac_gt05": kge_stats["frac_gt05"],
            })

    out_dir = save_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlw_nse_kge_summary.csv"
    df = pd.DataFrame(rows)
    if df.empty:
        print("No checkpoints matched. Check filenames and MODEL_REGISTRY.")
        for model_name, cfg in MODEL_REGISTRY.items():
            base_stem = Path(cfg["ckpt"]).stem
            matches = list(save_dir.glob(f"{base_stem}_*.pth"))
            print(f"{model_name}: found {len(matches)} tagged checkpoints")
        return

    df = df.sort_values(["model", "tag"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    df["label"] = df["model"] + ":" + df["tag"]
    df = df.sort_values(["model", "tag"]).reset_index(drop=True)

    def plot_bar(metric, out_name, title):
        plt.figure(figsize=(10, 4.5))
        x = np.arange(len(df))
        plt.bar(x, df[metric].to_numpy(), color="#4c78a8")
        plt.xticks(x, df["label"].tolist(), rotation=45, ha="right", fontsize=8)
        plt.ylabel(metric)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=300)
        plt.close()

    plot_bar("nse_mean", "mlw_nse_mean_bar.png", "Mean NSE by Model/Tag")
    plot_bar("kge_mean", "mlw_kge_mean_bar.png", "Mean KGE by Model/Tag")


if __name__ == "__main__":
    main()
