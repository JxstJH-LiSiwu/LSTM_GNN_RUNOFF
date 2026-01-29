import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.metrics import compute_nse_per_basin, compute_kge_per_basin


def parse_args():
    p = argparse.ArgumentParser(description="Analyze forecast vs no-forecast impact from cached .pt files.")
    p.add_argument("--infer_dir", type=str, default="checkpoints/infer_cache")
    p.add_argument("--out_dir", type=str, default="checkpoints/infer_cache/analysis_forecast")
    p.add_argument("--lead_days", type=str, default="1",
                   help="Comma-separated lead days to analyze (default: 1).")
    p.add_argument("--min_valid", type=int, default=30)
    p.add_argument("--peak_q", type=float, default=0.95)
    p.add_argument("--low_q", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def load_pt(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    obs_q = d.get("obs_q", None)
    pred_q = d.get("pred_q", None)
    mask = d.get("mask", None)
    basin_ids = d.get("basin_ids", None)
    lead_days = d.get("lead_days", None)
    if obs_q is None or pred_q is None or mask is None:
        raise ValueError(f"Missing obs_q/pred_q/mask in {path}")
    lead_val = int(lead_days) if lead_days is not None else None
    return np.asarray(obs_q), np.asarray(pred_q), np.asarray(mask).astype(bool), basin_ids, lead_val


def parse_filename(name: str):
    if name.endswith(".pt"):
        name = name[:-3]
    if "_noforecast_hop" in name:
        model = name.split("_noforecast_hop")[0]
        hop = int(name.split("_noforecast_hop")[1])
        return {"model": model, "hop": hop, "lead": None, "forecast": False}
    if "_lead" in name and "_hop" in name:
        pre, hop_part = name.rsplit("_hop", 1)
        hop = int(hop_part)
        model, lead_part = pre.rsplit("_lead", 1)
        lead = int(lead_part)
        return {"model": model, "hop": hop, "lead": lead, "forecast": True}
    return None


def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size > 0 else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size > 0 else np.nan


def per_basin_basic_metrics(obs, pred, mask, min_valid=30):
    T, N = obs.shape
    rmse = np.full(N, np.nan, dtype=np.float64)
    mae = np.full(N, np.nan, dtype=np.float64)
    bias = np.full(N, np.nan, dtype=np.float64)
    rel_bias = np.full(N, np.nan, dtype=np.float64)
    corr = np.full(N, np.nan, dtype=np.float64)

    for j in range(N):
        v = mask[:, j] & np.isfinite(obs[:, j]) & np.isfinite(pred[:, j])
        if v.sum() < min_valid:
            continue
        o = obs[v, j]
        p = pred[v, j]
        diff = p - o
        rmse[j] = math.sqrt(np.mean(diff ** 2))
        mae[j] = np.mean(np.abs(diff))
        bias[j] = np.mean(diff)
        mu = np.mean(o)
        if np.isfinite(mu) and mu != 0.0:
            rel_bias[j] = bias[j] / mu
        if np.std(o) > 0 and np.std(p) > 0:
            corr[j] = np.corrcoef(o, p)[0, 1]
    return rmse, mae, bias, rel_bias, corr


def per_basin_peak_low_metrics(obs, pred, mask, peak_q=0.95, low_q=0.5, min_valid=30):
    T, N = obs.shape
    peak_rmse = np.full(N, np.nan, dtype=np.float64)
    peak_rel_bias = np.full(N, np.nan, dtype=np.float64)
    low_rmse = np.full(N, np.nan, dtype=np.float64)

    for j in range(N):
        v = mask[:, j] & np.isfinite(obs[:, j]) & np.isfinite(pred[:, j])
        if v.sum() < min_valid:
            continue
        o = obs[v, j]
        p = pred[v, j]
        if o.size < min_valid:
            continue
        q_peak = np.quantile(o, peak_q)
        q_low = np.quantile(o, low_q)

        peak_mask = o >= q_peak
        low_mask = o <= q_low

        if peak_mask.sum() >= max(5, min_valid // 3):
            op = o[peak_mask]
            pp = p[peak_mask]
            diffp = pp - op
            peak_rmse[j] = math.sqrt(np.mean(diffp ** 2))
            mu = np.mean(op)
            if np.isfinite(mu) and mu != 0.0:
                peak_rel_bias[j] = np.mean(diffp) / mu

        if low_mask.sum() >= max(5, min_valid // 3):
            ol = o[low_mask]
            pl = p[low_mask]
            diffl = pl - ol
            low_rmse[j] = math.sqrt(np.mean(diffl ** 2))

    return peak_rmse, peak_rel_bias, low_rmse


def plot_cdf(values_a, values_b, label_a, label_b, title, out_path, dpi=160):
    plt.figure(figsize=(6.8, 5.2))
    style = [
        (values_a, label_a, "#d62728", "-"),
        (values_b, label_b, "#7f7f7f", "--"),
    ]
    for vals, label, color, linestyle in style:
        x = np.asarray(vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        x.sort()
        cdf = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, cdf, label=label, color=color, linestyle=linestyle, linewidth=0.6)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1.2)
    plt.xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=12)
    plt.ylabel("Cumulative fraction of basins", fontsize=12)
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_cdf_multi(series, title, out_path, dpi=160):
    plt.figure(figsize=(6.8, 5.2))
    for vals, label, color, linestyle in series:
        x = np.asarray(vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        x.sort()
        cdf = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, cdf, label=label, color=color, linestyle=linestyle, linewidth=0.6)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1.2)
    plt.xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=12)
    plt.ylabel("Cumulative fraction of basins", fontsize=12)
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_delta_hist(delta, title, out_path, dpi=160):
    x = np.asarray(delta, dtype=np.float64)
    x = x[np.isfinite(x)]
    plt.figure(figsize=(6.4, 4.6))
    if x.size > 0:
        plt.hist(x, bins=50, alpha=0.85)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    plt.xlabel("Delta (forecast - noforecast)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_scatter(a, b, title, out_path, dpi=160):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    plt.figure(figsize=(5.5, 5.5))
    if x.size > 0:
        plt.scatter(x, y, s=6, alpha=0.25)
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    plt.xlabel("Forecast NSE")
    plt.ylabel("No-forecast NSE")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_line(x, y, title, xlabel, ylabel, out_path, dpi=160):
    plt.figure(figsize=(6.4, 4.6))
    plt.plot(x, y, marker="o", linewidth=1.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    args = parse_args()
    infer_dir = Path(args.infer_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lead_days = [int(x.strip()) for x in args.lead_days.split(",") if x.strip()]

    forecast_files = {}
    noforecast_files = {}

    for p in infer_dir.glob("*.pt"):
        info = parse_filename(p.name)
        if info is None:
            continue
        if info["forecast"]:
            if info["lead"] not in lead_days:
                continue
            key = (info["model"], info["hop"], info["lead"])
            forecast_files[key] = p
        else:
            key = (info["model"], info["hop"])
            noforecast_files[key] = p

    summary_rows = []
    nse_forecast_map = {}

    for (model, hop, lead), f_path in sorted(forecast_files.items()):
        nf_key = (model, hop)
        if nf_key not in noforecast_files:
            continue
        nf_path = noforecast_files[nf_key]

        obs_f, pred_f, mask_f, basin_ids_f, lead_f = load_pt(f_path)
        obs_n, pred_n, mask_n, basin_ids_n, lead_n = load_pt(nf_path)

        if lead_f is not None and lead_f != lead:
            print(f"[Warn] lead mismatch in {f_path.name}: parsed={lead}, meta={lead_f}")
        if basin_ids_f is not None and basin_ids_n is not None and list(basin_ids_f) != list(basin_ids_n):
            print(f"[Warn] basin_ids mismatch for {model} hop{hop}")

        nse_f = compute_nse_per_basin(obs_f, pred_f, mask_f, basin_ids=None, min_valid=args.min_valid)
        nse_n = compute_nse_per_basin(obs_n, pred_n, mask_n, basin_ids=None, min_valid=args.min_valid)
        kge_f = compute_kge_per_basin(obs_f, pred_f, mask_f, basin_ids=None, min_valid=args.min_valid)
        kge_n = compute_kge_per_basin(obs_n, pred_n, mask_n, basin_ids=None, min_valid=args.min_valid)

        nse_f_arr = np.array(list(nse_f.values()), dtype=np.float64)
        nse_n_arr = np.array(list(nse_n.values()), dtype=np.float64)
        kge_f_arr = np.array(list(kge_f.values()), dtype=np.float64)
        kge_n_arr = np.array(list(kge_n.values()), dtype=np.float64)

        rmse_f, mae_f, bias_f, rbias_f, corr_f = per_basin_basic_metrics(
            obs_f, pred_f, mask_f, min_valid=args.min_valid
        )
        rmse_n, mae_n, bias_n, rbias_n, corr_n = per_basin_basic_metrics(
            obs_n, pred_n, mask_n, min_valid=args.min_valid
        )

        peak_rmse_f, peak_rbias_f, low_rmse_f = per_basin_peak_low_metrics(
            obs_f, pred_f, mask_f, peak_q=args.peak_q, low_q=args.low_q, min_valid=args.min_valid
        )
        peak_rmse_n, peak_rbias_n, low_rmse_n = per_basin_peak_low_metrics(
            obs_n, pred_n, mask_n, peak_q=args.peak_q, low_q=args.low_q, min_valid=args.min_valid
        )

        delta_nse = nse_f_arr - nse_n_arr
        delta_kge = kge_f_arr - kge_n_arr
        delta_rmse = rmse_f - rmse_n
        delta_mae = mae_f - mae_n
        delta_corr = corr_f - corr_n
        delta_peak_rmse = peak_rmse_f - peak_rmse_n
        delta_peak_rbias = peak_rbias_f - peak_rbias_n
        delta_low_rmse = low_rmse_f - low_rmse_n

        base = f"{model}_lead{lead}_hop{hop}"
        plot_cdf(
            nse_f_arr, nse_n_arr,
            label_a="forecast", label_b="no-forecast",
            title=f"NSE CDF | {base}",
            out_path=out_dir / f"cdf_nse_{base}.png",
            dpi=args.dpi,
        )
        plot_delta_hist(
            delta_nse,
            title=f"Delta NSE (forecast - noforecast) | {base}",
            out_path=out_dir / f"hist_delta_nse_{base}.png",
            dpi=args.dpi,
        )
        plot_scatter(
            nse_f_arr, nse_n_arr,
            title=f"NSE Scatter | {base}",
            out_path=out_dir / f"scatter_nse_{base}.png",
            dpi=args.dpi,
        )

        nse_forecast_map[(model, lead, hop)] = nse_f_arr

        summary_rows.append({
            "model": model,
            "lead": lead,
            "hop": hop,
            "mean_nse_forecast": safe_mean(nse_f_arr),
            "mean_nse_noforecast": safe_mean(nse_n_arr),
            "median_nse_forecast": safe_median(nse_f_arr),
            "median_nse_noforecast": safe_median(nse_n_arr),
            "mean_delta_nse": safe_mean(delta_nse),
            "median_delta_nse": safe_median(delta_nse),
            "frac_improved_nse": safe_mean(delta_nse > 0),
            "mean_delta_kge": safe_mean(delta_kge),
            "mean_delta_rmse": safe_mean(delta_rmse),
            "mean_delta_mae": safe_mean(delta_mae),
            "mean_delta_corr": safe_mean(delta_corr),
            "mean_delta_peak_rmse": safe_mean(delta_peak_rmse),
            "mean_delta_peak_rbias": safe_mean(delta_peak_rbias),
            "mean_delta_low_rmse": safe_mean(delta_low_rmse),
        })

    # summary plots across lead for each model+hop
    for (model, hop), group in group_by(summary_rows, keys=("model", "hop")).items():
        leads = sorted(set(r["lead"] for r in group))
        mean_delta_nse = [first_val(group, lead, "mean_delta_nse") for lead in leads]
        mean_delta_kge = [first_val(group, lead, "mean_delta_kge") for lead in leads]
        mean_delta_rmse = [first_val(group, lead, "mean_delta_rmse") for lead in leads]
        mean_delta_peak_rmse = [first_val(group, lead, "mean_delta_peak_rmse") for lead in leads]
        mean_delta_low_rmse = [first_val(group, lead, "mean_delta_low_rmse") for lead in leads]

        plot_line(
            leads, mean_delta_nse,
            title=f"Mean Delta NSE vs Lead | {model} hop{hop}",
            xlabel="Lead days",
            ylabel="Mean Delta NSE",
            out_path=out_dir / f"delta_nse_vs_lead_{model}_hop{hop}.png",
            dpi=args.dpi,
        )
        plot_line(
            leads, mean_delta_kge,
            title=f"Mean Delta KGE vs Lead | {model} hop{hop}",
            xlabel="Lead days",
            ylabel="Mean Delta KGE",
            out_path=out_dir / f"delta_kge_vs_lead_{model}_hop{hop}.png",
            dpi=args.dpi,
        )
        plot_line(
            leads, mean_delta_rmse,
            title=f"Mean Delta RMSE vs Lead | {model} hop{hop}",
            xlabel="Lead days",
            ylabel="Mean Delta RMSE",
            out_path=out_dir / f"delta_rmse_vs_lead_{model}_hop{hop}.png",
            dpi=args.dpi,
        )
        plot_line(
            leads, mean_delta_peak_rmse,
            title=f"Mean Delta Peak RMSE vs Lead | {model} hop{hop}",
            xlabel="Lead days",
            ylabel="Mean Delta Peak RMSE",
            out_path=out_dir / f"delta_peak_rmse_vs_lead_{model}_hop{hop}.png",
            dpi=args.dpi,
        )
        plot_line(
            leads, mean_delta_low_rmse,
            title=f"Mean Delta Low-flow RMSE vs Lead | {model} hop{hop}",
            xlabel="Lead days",
            ylabel="Mean Delta Low-flow RMSE",
            out_path=out_dir / f"delta_low_rmse_vs_lead_{model}_hop{hop}.png",
            dpi=args.dpi,
        )

    # combined NSE CDF: LSTM vs LSTM-GAT with/without forecast on one figure
    for hop in sorted({k[2] for k in nse_forecast_map.keys()}):
        for lead in sorted({k[1] for k in nse_forecast_map.keys()}):
            lstm_f = nse_forecast_map.get(("LSTM", lead, hop), None)
            gat_f = nse_forecast_map.get(("LSTM-GAT", lead, hop), None)
            if lstm_f is None or gat_f is None:
                continue

            # load no-forecast for same hop (lead not applicable)
            lstm_nf_path = noforecast_files.get(("LSTM", hop), None)
            gat_nf_path = noforecast_files.get(("LSTM-GAT", hop), None)
            if lstm_nf_path is None or gat_nf_path is None:
                continue
            obs_nf_l, pred_nf_l, mask_nf_l, _, _ = load_pt(lstm_nf_path)
            obs_nf_g, pred_nf_g, mask_nf_g, _, _ = load_pt(gat_nf_path)
            nse_l_nf = np.array(list(compute_nse_per_basin(
                obs_nf_l, pred_nf_l, mask_nf_l, basin_ids=None, min_valid=args.min_valid
            ).values()), dtype=np.float64)
            nse_g_nf = np.array(list(compute_nse_per_basin(
                obs_nf_g, pred_nf_g, mask_nf_g, basin_ids=None, min_valid=args.min_valid
            ).values()), dtype=np.float64)

            series = [
                (lstm_f, "LSTM (forecast)", "#7f7f7f", "--"),
                (nse_l_nf, "LSTM (no-forecast)", "#7f7f7f", ":"),
                (gat_f, "LSTM-GAT (forecast)", "#d62728", "-"),
                (nse_g_nf, "LSTM-GAT (no-forecast)", "#d62728", "-."),
            ]
            plot_cdf_multi(
                series,
                title=f"NSE CDF | lead{lead} hop{hop}",
                out_path=out_dir / f"cdf_nse_LSTM_GAT_forecast_vs_noforecast_lead{lead}_hop{hop}.png",
                dpi=args.dpi,
            )

    # write summary CSV
    csv_path = out_dir / "summary_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[Done] Saved plots and summary to {out_dir}")


def group_by(rows, keys):
    out = {}
    for r in rows:
        k = tuple(r[key] for key in keys)
        out.setdefault(k, []).append(r)
    return out


def first_val(rows, lead, key):
    for r in rows:
        if r["lead"] == lead:
            return r.get(key, np.nan)
    return np.nan


if __name__ == "__main__":
    main()
