import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset.data_prepare import load_lamah_from_cache
from src.metrics import compute_nse_per_basin




# ============================================================
# Predict helper (identical interface for both models)
# ============================================================
def load_infer_pt(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["obs_q"], d["pred_q"], d["mask"], d.get("basin_ids", None)


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
    pred_lstm_nf=None,
    pred_gat_nf=None,
    nse_lstm_nf=None,
    nse_gat_nf=None,
    save_path=None,
):
    plt.plot(time_idx, obs, color="black", linewidth=1.8, label="Observed")
    plt.plot(
        time_idx,
        pred_lstm,
        linestyle="--",
        color="#1f77b4",
        linewidth=1.6,
        label="LSTM (no-forecast)",
    )
    plt.plot(
        time_idx,
        pred_gat,
        color="#d62728",
        linewidth=1.6,
        label="LSTM-GAT",
    )
    if pred_lstm_nf is not None:
        plt.plot(
            time_idx,
            pred_lstm_nf,
            linestyle=":",
            color="#1f77b4",
            linewidth=1.2,
            label="LSTM (no-forecast)",
        )
    if pred_gat_nf is not None:
        plt.plot(
            time_idx,
            pred_gat_nf,
            linestyle=":",
            color="#d62728",
            linewidth=1.2,
            label="LSTM-GAT (no-forecast)",
        )

    plt.title(f"Station {station_id}", fontsize=11)
    plt.ylabel("Streamflow (m³/s)")

    txt = f"LSTM (no-forecast) NSE = {nse_lstm:.3f}\nLSTM-GAT NSE = {nse_gat:.3f}"
    if nse_lstm_nf is not None and nse_gat_nf is not None:
        txt = (
            f"LSTM (no-forecast) NSE = {nse_lstm:.3f}\n"
            f"LSTM-GAT NSE = {nse_gat:.3f}\n"
            f"LSTM (no-forecast) NSE = {nse_lstm_nf:.3f}\n"
            f"LSTM-GAT (no-forecast) NSE = {nse_gat_nf:.3f}"
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


def find_peak_windows(series, mask=None, *, peak_count=5, window=30, min_sep=30):
    series = np.asarray(series, dtype=np.float64)
    if mask is None:
        valid = np.isfinite(series)
    else:
        valid = np.asarray(mask).astype(bool) & np.isfinite(series)
    if valid.sum() < 3:
        return []

    s = series.copy()
    s[~valid] = -np.inf

    peaks = []
    for i in range(1, len(s) - 1):
        if not np.isfinite(s[i]):
            continue
        if s[i - 1] < s[i] >= s[i + 1]:
            peaks.append(i)

    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda i: s[i], reverse=True)
    selected = []
    for idx in peaks:
        if all(abs(idx - s) >= min_sep for s in selected):
            selected.append(idx)
        if len(selected) >= peak_count:
            break

    windows = []
    for idx in sorted(selected):
        start = max(0, idx - window)
        end = min(len(series), idx + window + 1)
        windows.append((start, end, idx))

    return windows


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplet_file", type=str, default="",
                   help="Path to selected_triplets_*.txt (A,B -> C). If set, uses first triplet.")
    p.add_argument("--infer_dir", type=str, default="checkpoints/infer_cache")
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--hop", type=int, default=3)
    p.add_argument("--compare_noforecast", action="store_true",
                   help="Also plot no-forecast LSTM/GAT (from *_noforecast_hop*.pt)")
    return p.parse_args()


def load_first_triplet(path: Path):
    line = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                line = raw
                break
    if line is None:
        raise ValueError(f"Empty triplet file: {path}")
    parts = {kv.split("=")[0]: kv.split("=")[1] for kv in line.split(",") if "=" in kv}
    a = parts.get("A", "")
    b = parts.get("B", "")
    c = parts.get("C", "")
    if not (a and b and c):
        raise ValueError(f"Invalid triplet line: {line}")
    return [a, b, c], c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    SAVE_DIR   = base_dir / "checkpoints"
    CACHE_FILE = SAVE_DIR / "data_cache" / "lamah_daily.pt"
    OUT_DIR = SAVE_DIR / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Params
    # --------------------------------------------------------
    MIN_VALID = 30

    TARGET_STATIONS = ["ID_532", "ID_530", "ID_533"]
    PEAK_REF_STATION = "ID_532"
    if args.triplet_file:
        triplet_path = Path(args.triplet_file)
        TARGET_STATIONS, PEAK_REF_STATION = load_first_triplet(triplet_path)
    PEAK_COUNT = 5
    PEAK_WINDOW = 210
    PEAK_MIN_SEP = 30
    SAVE_DPI = 600

    # --------------------------------------------------------
    # Load cached inference (.pt)
    # --------------------------------------------------------
    infer_dir = Path(args.infer_dir)
    lstm_pt = infer_dir / f"LSTM_noforecast_hop{args.hop}.pt"
    gat_pt = infer_dir / f"LSTM-GAT_lead{args.lead}_hop{args.hop}.pt"
    if not lstm_pt.exists():
        raise FileNotFoundError(f"Missing {lstm_pt}")
    if not gat_pt.exists():
        raise FileNotFoundError(f"Missing {gat_pt}")

    obs_q, pred_lstm_q, mask, basin_ids = load_infer_pt(lstm_pt)
    obs_q2, pred_gat_q, mask2, basin_ids2 = load_infer_pt(gat_pt)

    if basin_ids is not None and basin_ids2 is not None and list(basin_ids) != list(basin_ids2):
        raise ValueError("Basin IDs mismatch between LSTM and LSTM-GAT .pt")
    if mask2 is not None:
        mask = mask & mask2

    pred_lstm_nf = pred_gat_nf = None
    nse_lstm_nf = nse_gat_nf = None
    if args.compare_noforecast:
        lstm_nf_pt = infer_dir / f"LSTM_noforecast_hop{args.hop}.pt"
        gat_nf_pt = infer_dir / f"LSTM-GAT_noforecast_hop{args.hop}.pt"
        if lstm_nf_pt.exists() and gat_nf_pt.exists():
            obs_nf, pred_lstm_nf, mask_nf, basin_ids_nf = load_infer_pt(lstm_nf_pt)
            obs_nf2, pred_gat_nf, mask_nf2, basin_ids_nf2 = load_infer_pt(gat_nf_pt)
            if basin_ids_nf is not None and basin_ids is not None and list(basin_ids_nf) != list(basin_ids):
                raise ValueError("Basin IDs mismatch in no-forecast LSTM .pt")
            if basin_ids_nf2 is not None and basin_ids is not None and list(basin_ids_nf2) != list(basin_ids):
                raise ValueError("Basin IDs mismatch in no-forecast LSTM-GAT .pt")
            mask = mask & mask_nf & mask_nf2
        else:
            print("[Warn] no-forecast .pt not found, skip no-forecast plotting.")

    if basin_ids is None:
        cache = load_lamah_from_cache(CACHE_FILE)
        static_df = cache["static_df"]
        basin_ids = list(static_df.index)
    basin_id_to_idx = {bid: i for i, bid in enumerate(basin_ids)}

    # --------------------------------------------------------
    # NSE per basin
    # --------------------------------------------------------
    nse_lstm = compute_nse_per_basin(
        obs_q, pred_lstm_q, mask, basin_ids, MIN_VALID
    )
    nse_gat = compute_nse_per_basin(
        obs_q, pred_gat_q, mask, basin_ids, MIN_VALID
    )
    if pred_lstm_nf is not None and pred_gat_nf is not None:
        nse_lstm_nf = compute_nse_per_basin(
            obs_q, pred_lstm_nf, mask, basin_ids, MIN_VALID
        )
        nse_gat_nf = compute_nse_per_basin(
            obs_q, pred_gat_nf, mask, basin_ids, MIN_VALID
        )

    # --------------------------------------------------------
    # Plot Figure 9
    # --------------------------------------------------------
    time_idx = np.arange(obs_q.shape[0])

    ref_idx = basin_id_to_idx[PEAK_REF_STATION]
    peak_windows = find_peak_windows(
        obs_q[:, ref_idx],
        peak_count=PEAK_COUNT,
        window=PEAK_WINDOW,
        min_sep=PEAK_MIN_SEP,
    )
    print(peak_windows)

    if not peak_windows:
        peak_windows = [(0, len(time_idx), None)]

    for win_id, (start, end, peak_idx) in enumerate(peak_windows, start=1):
        print(win_id)
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=(10.5, 6.8), sharex=True
        )

        for ax, station_id in zip(axes, TARGET_STATIONS):
            idx = basin_id_to_idx[station_id]

            plt.sca(ax)
            plot_hydrograph_comparison(
                time_idx[start:end],
                obs_q[start:end, idx],
                pred_lstm_q[start:end, idx],
                pred_gat_q[start:end, idx],
                nse_lstm[station_id],
                nse_gat[station_id],
                station_id,
                pred_lstm_nf=pred_lstm_nf[start:end, idx] if pred_lstm_nf is not None else None,
                pred_gat_nf=pred_gat_nf[start:end, idx] if pred_gat_nf is not None else None,
                nse_lstm_nf=nse_lstm_nf[station_id] if nse_lstm_nf is not None else None,
                nse_gat_nf=nse_gat_nf[station_id] if nse_gat_nf is not None else None,
            )

        axes[-1].set_xlabel("Time step")
        plt.tight_layout()

        fname = f"figure9_hydrograph_lstm_vs_gat_peak_{win_id}.png"
        plt.savefig(OUT_DIR / fname, dpi=SAVE_DPI)
        plt.close(fig)


if __name__ == "__main__":
    main()
