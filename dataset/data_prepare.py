# dataset/data_prepare.py
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# ============================================================
# Utilities: scaling / transforms
# ============================================================

def minmax_fit_from_train(df: pd.DataFrame, eps: float = 1e-6):
    """
    Fit min/max on TRAIN ONLY.
    Returns dict with per-column min/max (pd.Series).
    """
    minv = df.min(axis=0)
    maxv = df.max(axis=0)
    return {"min": minv, "max": maxv, "eps": float(eps)}

def minmax_apply(df: pd.DataFrame, scaler: dict):
    eps = scaler.get("eps", 1e-6)
    minv = scaler["min"]
    maxv = scaler["max"]
    scaled = (df - minv) / (maxv - minv + eps)
    return scaled

def per_basin_median_fit_train_only(df_raw: pd.DataFrame, train_time_mask: np.ndarray, eps: float = 1e-6):
    """
    Fit per-basin median on TRAIN ONLY (time dimension only).
    df_raw: (T, N) columns are basin_ids.
    train_time_mask: boolean mask over rows (T,)
    Returns:
        median_per_basin: pd.Series indexed by basin_id
    """
    x = df_raw.copy()
    x = x.clip(lower=0.0)
    x_train = x.loc[train_time_mask]
    med = x_train.median(axis=0, skipna=True)

    # avoid zeros: if median==0, set to small positive value to keep log1p(x/med) stable
    med = med.replace(0.0, np.nan)
    med = med.fillna(eps)
    return med

def positive_robust_log_per_basin_apply(df_raw: pd.DataFrame, median_per_basin: pd.Series, eps: float = 1e-6):
    """
    Per-basin positive robust log transform (median scale):
        y = log1p( x / (median_i + eps) )
    Works for precipitation and streamflow (non-negative).
    Keeps NaNs as NaNs.
    """
    x = df_raw.clip(lower=0.0)
    denom = (median_per_basin + eps)
    y = np.log1p(x / denom)
    return y

def positive_robust_log_per_basin_inverse(y: np.ndarray, median_per_basin: np.ndarray, eps: float = 1e-6):
    """
    Inverse of y = log1p(x/(median_i+eps)):
        x = (median_i+eps) * (expm1(y))
    y: (B, N) or (T, N)
    median_per_basin: (N,)
    """
    y = np.asarray(y, dtype=np.float64)
    med = np.asarray(median_per_basin, dtype=np.float64).reshape(1, -1)
    x = (med + eps) * np.expm1(y)
    x = np.clip(x, 0.0, None)
    return x.astype(np.float64)

# ============================================================
# 0. Load selected basin ids (from 222.csv)
# ============================================================

def load_selected_basin_ids(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    id_col = "gauge_id" if "gauge_id" in df.columns else "ID"
    ids = df[id_col].astype(str).tolist()
    return [f"ID_{i}" for i in ids]

# ============================================================
# 1. Read per-basin csv (timeseries)
# ============================================================

def read_per_basin_csv(csv_path: Path, value_col: str):
    df = pd.read_csv(csv_path, sep=";")
    df[value_col] = df[value_col].replace(-999, np.nan)

    date = pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
    )
    return pd.Series(df[value_col].values, index=date)

def load_lamah_meteo_daily(daily_dir: Path, basin_ids: list[str], value_col: str):
    series_list = []
    for bid in basin_ids:
        s = read_per_basin_csv(daily_dir / f"{bid}.csv", value_col)
        s.name = bid
        series_list.append(s)
    return pd.concat(series_list, axis=1).sort_index()

# ============================================================
# 2. edge_index + edge_weight (Kirpich travel-time inverse)
# ============================================================

def build_edge_index_and_weight_from_stream_dist(
    stream_dist_csv: str,
    basin_ids: list[str],
    eps: float = 1e-6,
    normalize: bool =True,
):
    df = pd.read_csv(stream_dist_csv, sep=";")

    basin_to_idx = {bid: i for i, bid in enumerate(basin_ids)}
    src, dst, w_raw = [], [], []

    K_kirpich = 0.0195  # SI, minutes

    for _, row in df.iterrows():
        try:
            up = f"ID_{int(row['ID'])}"
            down = f"ID_{int(row['NEXTDOWNID'])}"

            if up not in basin_to_idx or down not in basin_to_idx:
                continue

            dist_hdn = float(row["dist_hdn"])
            strm_slope = float(row["strm_slope"])

            if (not np.isfinite(dist_hdn)) or (not np.isfinite(strm_slope)):
                continue
            if dist_hdn <= 0 or strm_slope <= 0:
                continue

            # unit conversion (assumed)
            L_m = dist_hdn * 1000.0          # km -> m
            S_dimless = strm_slope / 1000.0  # m/km -> m/m
            S_dimless = max(S_dimless, eps)

            t_c_min = K_kirpich * (L_m ** 0.77) * (S_dimless ** (-0.385))
            if (not np.isfinite(t_c_min)) or (t_c_min <= 0):
                continue

            weight = 1.0 / (t_c_min + eps)
            if not np.isfinite(weight):
                continue

            src.append(basin_to_idx[up])
            dst.append(basin_to_idx[down])
            w_raw.append(weight)

        except Exception:
            continue

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight_raw = torch.tensor(w_raw, dtype=torch.float32)

    if edge_weight_raw.numel() == 0:
        return edge_index, edge_weight_raw, edge_weight_raw

    if normalize:
        # per-downstream normalize incoming sum to 1
        num_nodes = len(basin_ids)
        sum_in = torch.zeros(num_nodes, dtype=torch.float32)
        sum_in.scatter_add_(0, edge_index[1], edge_weight_raw)

        denom = sum_in[edge_index[1]].clamp_min(eps)
        edge_weight = edge_weight_raw / denom
        edge_weight = torch.where(torch.isfinite(edge_weight), edge_weight, torch.zeros_like(edge_weight))
    else:
        edge_weight = edge_weight_raw

    return edge_index, edge_weight, edge_weight_raw

# ============================================================
# 3. Time split (70/15/15) - done in data_prepare.py
#     - test: last 15% contiguous
#     - train/val: random split within first 85%
# ============================================================

def build_time_split_indices(time_index, seq_len: int, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Return dict of indices t where sample uses [t-seq_len+1..t] -> predict t+1.
    So valid t range: [seq_len, T-2] if T is len(time_index) and target at t+1 exists.
    We implement:
      - test: last test_ratio of valid t (contiguous)
      - train/val: random split from the remaining
    """
    T = len(time_index)
    all_t = np.arange(seq_len, T - 1)  # t+1 exists

    n_test = int(len(all_t) * test_ratio)
    test_idx = all_t[-n_test:] if n_test > 0 else np.array([], dtype=np.int64)
    trainval_idx = all_t[:-n_test] if n_test > 0 else all_t

    # train/val split within trainval
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trainval_idx))

    # train_ratio and val_ratio are absolute (0.7, 0.15) over full
    # within trainval block (0.85), train fraction = 0.7/0.85
    denom = (train_ratio + val_ratio)
    train_frac_in_trainval = train_ratio / denom

    n_train = int(len(trainval_idx) * train_frac_in_trainval)
    train_idx = np.sort(trainval_idx[perm[:n_train]])
    val_idx = np.sort(trainval_idx[perm[n_train:]])

    return {"train": train_idx.astype(np.int64), "val": val_idx.astype(np.int64), "test": test_idx.astype(np.int64)}

# ============================================================
# 4. Main loader (raw, no scaling here)
#    - Keep interface load_lamah_daily(root) but return RAW aligned data
# ============================================================

def load_lamah_daily(root: str):
    root = Path(root)

    # ---------- selected basins ----------
    selected_ids_csv = root / "222.csv"
    selected_basin_ids = load_selected_basin_ids(selected_ids_csv)

    # ---------- static attributes ----------
    attr_path = root / "B_basins_diff_upstrm_all/1_attributes/Catchment_attributes.csv"
    static_df = pd.read_csv(attr_path, sep=";")

    id_col = "gauge_id" if "gauge_id" in static_df.columns else "ID"
    static_df[id_col] = static_df[id_col].astype(str)
    static_df.index = static_df[id_col].apply(lambda x: f"ID_{x}")

    static_df = static_df.apply(pd.to_numeric, errors="coerce")
    static_df = static_df.dropna(axis=1, how="all")
    static_df = static_df.fillna(static_df.mean())

    if id_col in static_df.columns:
        static_df = static_df.drop(columns=[id_col])

    static_df = static_df.loc[static_df.index.intersection(selected_basin_ids)]
    basin_ids = list(static_df.index)

    # ---------- dynamic time series ----------
    meteo_dir = root / "B_basins_diff_upstrm_all/2_timeseries/daily"
    runoff_dir = root / "D_gauges/2_timeseries/daily"

    precip_df = load_lamah_meteo_daily(meteo_dir, basin_ids, "prec")
    temp_df   = load_lamah_meteo_daily(meteo_dir, basin_ids, "2m_temp_mean")
    soil_df   = load_lamah_meteo_daily(meteo_dir, basin_ids, "volsw_123")
    runoff_df = load_lamah_meteo_daily(runoff_dir, basin_ids, "qobs")  # raw Q

    # ---------- align time index ----------
    common_index = precip_df.index.intersection(runoff_df.index)
    precip_df = precip_df.loc[common_index]
    temp_df   = temp_df.loc[common_index]
    soil_df   = soil_df.loc[common_index]
    runoff_df = runoff_df.loc[common_index]

    # ---------- time range (1987–2017) ----------
    precip_df = precip_df.loc["1987-01-01":"2017-12-31"]
    temp_df   = temp_df.loc["1987-01-01":"2017-12-31"]
    soil_df   = soil_df.loc["1987-01-01":"2017-12-31"]
    runoff_df = runoff_df.loc["1987-01-01":"2017-12-31"]

    # ---------- graph ----------
    stream_csv = root / "B_basins_diff_upstrm_all/1_attributes/Stream_dist.csv"
    edge_index, edge_weight, edge_weight_raw = build_edge_index_and_weight_from_stream_dist(
        str(stream_csv), basin_ids
    )

    return (
        precip_df,
        temp_df,
        soil_df,
        runoff_df,
        static_df,
        edge_index,
        edge_weight,
        edge_weight_raw
    )

# ============================================================
# 5. Preprocess + cache (NOW does: split + transforms + scaling)
# ============================================================

def prepare_and_save_lamah_daily(
    root: str,
    cache_path: Path,
    *,
    seq_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
    eps: float = 1e-6,
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {cache_path} (set overwrite=True)")

    print("[Cache] Building LamaH daily data from raw CSV...")
    (
        precip_raw,
        temp_raw,
        soil_raw,
        runoff_raw,
        static_raw,
        edge_index,
        edge_weight,
        edge_weight_raw
    ) = load_lamah_daily(root)

    basin_ids = list(static_raw.index)
    time_index = precip_raw.index
    T = len(time_index)

    # ---------- build split indices (t indices used by dataset) ----------
    split = build_time_split_indices(
        time_index=time_index,
        seq_len=seq_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_idx = split["train"]

    # ---------- train time mask over rows for fitting scalers ----------
    # Use rows that appear in the input window and target range:
    # For transforms/scale, we fit on TRAIN PERIOD rows only (time rows), simplest is:
    # - use raw rows corresponding to (train_idx - (seq_len-1) .. train_idx+1)
    # But to keep minimal & stable, fit using the union of rows involved in train samples:
    train_rows = np.zeros(T, dtype=bool)
    for t in train_idx:
        start = max(0, t - (seq_len - 1))
        end = min(T - 1, t + 1)  # include target row t+1
        train_rows[start:end + 1] = True

    # ============================================================
    # (A) positive_robust_log (per-basin median) for precip & runoff
    # ============================================================
    precip_median = per_basin_median_fit_train_only(precip_raw, train_rows, eps=eps)
    runoff_median = per_basin_median_fit_train_only(runoff_raw, train_rows, eps=eps)

    precip_df = positive_robust_log_per_basin_apply(precip_raw, precip_median, eps=eps)
    runoff_df = positive_robust_log_per_basin_apply(runoff_raw, runoff_median, eps=eps)

    # NOTE: keep invalid discharge as NaN (mask later in Dataset)
    # (runoff_raw might include <=0 or NaN; after clip+log1p it becomes 0 for zeros,
    # but NaNs remain NaN. If你坚持“<=0 也算无效”，可以在这里强制置 NaN：
    runoff_df = runoff_df.where(np.isfinite(runoff_raw) & (runoff_raw > 0), np.nan)

    # ============================================================
    # (B) min-max scaling [0,1] for other dynamics (train-only)
    #     Each variable separately
    # ============================================================
    temp_scaler = minmax_fit_from_train(temp_raw.loc[train_rows], eps=eps)
    soil_scaler = minmax_fit_from_train(soil_raw.loc[train_rows], eps=eps)

    temp_df = minmax_apply(temp_raw, temp_scaler)
    soil_df = minmax_apply(soil_raw, soil_scaler)

    # ============================================================
    # (C) min-max scaling [0,1] for static (each feature separately)
    #     Static has no time; fit over basins (selected set)
    # ============================================================
    static_scaler = minmax_fit_from_train(static_raw, eps=eps)
    static_df = minmax_apply(static_raw, static_scaler)
    precip_df = precip_df.fillna(0.0)
    temp_df   = temp_df.fillna(0.0)
    soil_df   = soil_df.fillna(0.0)

    # ---------- DEBUG prints ----------
    print("\n========== [DEBUG] LamaH Processed Data Shapes ==========")
    print(f"precip_df shape : {precip_df.shape}  (T, N)")
    print(f"temp_df   shape : {temp_df.shape}  (T, N)")
    print(f"soil_df   shape : {soil_df.shape}  (T, N)")
    print(f"runoff_df shape : {runoff_df.shape}  (T, N)  [transformed target]")
    print(f"static_df shape : {static_df.shape}  (N, F_static)")
    print("\n---------- Split ----------")
    print(f"train/val/test sizes: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
    print("\n---------- Graph ----------")
    print(f"edge_index shape : {edge_index.shape}  (2, E)")
    print(f"edge_weight shape: {edge_weight.shape}  (E,)")
    print(f"edge_weight_raw shape: {edge_weight_raw.shape}  (E,)")  
    print("========================================================\n")

    cache = {
        "precip_df": precip_df,
        "temp_df": temp_df,
        "soil_df": soil_df,
        "runoff_df": runoff_df,  # transformed target
        "static_df": static_df,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_weight_raw": edge_weight_raw,
        "split": split,
        "meta": {
            "root": str(root),
            "basin_ids": basin_ids,
            "time_index": time_index,
            "seq_len": int(seq_len),
            "ratios": {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)},
            "seed": int(seed),
            "eps": float(eps),
            "scalers": {
                "precip": {"type": "positive_robust_log_per_basin_median", "median_per_basin": precip_median.astype(float)},
                "runoff": {"type": "positive_robust_log_per_basin_median", "median_per_basin": runoff_median.astype(float)},
                "temp": {"type": "minmax_train_only", "min": temp_scaler["min"].astype(float), "max": temp_scaler["max"].astype(float)},
                "soil": {"type": "minmax_train_only", "min": soil_scaler["min"].astype(float), "max": soil_scaler["max"].astype(float)},
                "static": {"type": "minmax_per_feature", "min": static_scaler["min"].astype(float), "max": static_scaler["max"].astype(float)},
            },
        },
    }

    torch.save(cache, cache_path)
    print(f"[Cache] Saved to {cache_path}")
    return cache

def load_lamah_from_cache(cache_path: Path):
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"[Cache] Loaded LamaH data from {cache_path}")
    return cache
