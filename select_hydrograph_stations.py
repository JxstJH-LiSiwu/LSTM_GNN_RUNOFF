import argparse
import math
from pathlib import Path

import numpy as np
import torch

from dataset.data_prepare import load_lamah_from_cache


def parse_args():
    p = argparse.ArgumentParser(description="Select hydrologically meaningful A,B -> C station triplets.")
    p.add_argument("--cache_file", type=str, default="checkpoints/data_cache/lamah_daily.pt")
    p.add_argument("--infer_dir", type=str, default="checkpoints/infer_cache")
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--hop", type=int, default=3)
    p.add_argument("--min_valid", type=int, default=30)
    p.add_argument("--min_nse_gat", type=float, default=0.8,
                   help="Minimum NSE for station C (GAT) before matching LSTM.")
    # only keep required filters
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()


def load_pt(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    obs_q = d.get("obs_q", None)
    pred_q = d.get("pred_q", None)
    mask = d.get("mask", None)
    basin_ids = d.get("basin_ids", None)
    if obs_q is None or pred_q is None or mask is None:
        raise ValueError(f"Missing obs_q/pred_q/mask in {path}")
    return np.asarray(obs_q), np.asarray(pred_q), np.asarray(mask).astype(bool), basin_ids


def nse_per_basin(obs, pred, mask, min_valid=30):
    T, N = obs.shape
    out = np.full(N, np.nan, dtype=np.float64)
    for j in range(N):
        v = mask[:, j] & np.isfinite(obs[:, j]) & np.isfinite(pred[:, j])
        if v.sum() < min_valid:
            continue
        o = obs[v, j]
        p = pred[v, j]
        denom = np.sum((o - o.mean()) ** 2)
        if not np.isfinite(denom) or denom <= 0:
            continue
        out[j] = 1.0 - np.sum((o - p) ** 2) / denom
    return out



def build_upstream_map(edge_index, n_nodes):
    # edge_index: (2, E), edges are up -> down
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    upstream = {i: [] for i in range(n_nodes)}
    for s, d in zip(src, dst):
        upstream[d].append(s)
    return upstream


def main():
    args = parse_args()

    cache = load_lamah_from_cache(Path(args.cache_file))
    edge_index = cache["edge_index"]
    static_df = cache["static_df"]
    basin_ids = list(static_df.index)
    n_nodes = len(basin_ids)
    upstream_map = build_upstream_map(edge_index, n_nodes)

    infer_dir = Path(args.infer_dir)
    lstm_path = infer_dir / f"LSTM_lead{args.lead}_hop{args.hop}.pt"
    gat_path = infer_dir / f"LSTM-GAT_lead{args.lead}_hop{args.hop}.pt"
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing {lstm_path}")
    if not gat_path.exists():
        raise FileNotFoundError(f"Missing {gat_path}")

    obs_l, pred_l, mask_l, basin_ids_l = load_pt(lstm_path)
    obs_g, pred_g, mask_g, basin_ids_g = load_pt(gat_path)

    if basin_ids_l is not None and basin_ids_l != basin_ids:
        raise ValueError("Basin IDs mismatch between cache and LSTM .pt")
    if basin_ids_g is not None and basin_ids_g != basin_ids:
        raise ValueError("Basin IDs mismatch between cache and GAT .pt")

    nse_l = nse_per_basin(obs_l, pred_l, mask_l, min_valid=args.min_valid)
    nse_g = nse_per_basin(obs_g, pred_g, mask_g, min_valid=args.min_valid)

    delta_nse = nse_g - nse_l

    candidates = []
    for c in range(n_nodes):
        ups = upstream_map.get(c, [])
        if len(ups) < 2:
            continue
        if not np.isfinite(nse_g[c]) or nse_g[c] < args.min_nse_gat:
            continue
        if not np.isfinite(nse_l[c]) or not np.isfinite(delta_nse[c]) or delta_nse[c] <= 0:
            continue
        # choose two upstream with valid NSE (GAT)
        ups_valid = [u for u in ups if np.isfinite(nse_g[u])]
        if len(ups_valid) < 2:
            continue

        # score: prioritize high GAT NSE at C, and improvement over LSTM
        score = 0.7 * nse_g[c] + 0.3 * delta_nse[c]

        ups_sorted = sorted(ups_valid, key=lambda u: nse_g[u], reverse=True)
        a, b = ups_sorted[0], ups_sorted[1]
        candidates.append((score, a, b, c))

    # sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[: args.top_k]

    print(f"\nTop {len(top)} triplets (A,B -> C) for hop={args.hop}, lead={args.lead}:")
    for rank, (score, a, b, c) in enumerate(top, 1):
        print(
            f"{rank:02d}. A={basin_ids[a]}  B={basin_ids[b]}  C={basin_ids[c]} | "
            f"score={score:.3f}  dNSE(C)={delta_nse[c]:.3f}  NSE(C)={nse_g[c]:.3f}"
        )

    # save to txt
    out_path = infer_dir / f"selected_triplets_lead{args.lead}_hop{args.hop}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for score, a, b, c in top:
            f.write(
            f"A={basin_ids[a]},B={basin_ids[b]},C={basin_ids[c]},"
            f"score={score:.4f},dNSE_C={delta_nse[c]:.4f},NSE_C={nse_g[c]:.4f}\n"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
