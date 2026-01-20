# train.py
from pathlib import Path
import torch
import time
import logging
from datetime import datetime, timedelta
import numpy as np

from dataset.data_prepare import (
    prepare_and_save_lamah_daily,
    load_lamah_from_cache,
    positive_robust_log_per_basin_inverse,
)
from dataset.lamah_dataset import LamaHDataset
from dataset.dataloader import create_dataloader
from src.train_one_epoch import train_one_epoch
from src import MODEL_REGISTRY


# ============================================================
# 全局配置（Paper-aligned）
# ============================================================

DATA_ROOT = Path("/home/lisiwu/jxwork/1-gnn-lstm/dataset")
SAVE_DIR = Path("/home/lisiwu/jxwork/1-gnn-lstm/checkpoints")

USE_CACHE = True
CACHE_DIR = SAVE_DIR / "data_cache"
CACHE_FILE = CACHE_DIR / "lamah_daily.pt"
INIT_FROM_RAW = True

TRAIN_MODE = 0
RESUME_EPOCH = 0
MAX_EPOCH = 100
MODEL_LIST = ["LSTM", "LSTM-GAT", "LSTM-GCN", "LSTM-Cheb", "LSTM-GraphSAGE"]

# FOR PAPER
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
GNN_HIDDEN_DIM = 64
GAT_HEADS = 4
LSTM_DROPOUT = 0.35
GNN_DROPOUT = 0.2
OUTPUT_DIM = 1
CHEBK = 3  # defult = 3 if use LSTM_CheNet

USE_AMP = True

BASE_BATCH_SIZE = 4
ACCUM_STEPS = 2

LEARNING_RATE = 5e-4
MIN_LR = 1e-5
LR_PATIENCE = 3

SEQ_LEN = 180
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

AUTO_PLOT = True

PEAK_WEIGHTING = True
PEAK_TOP_PCT = 0.025
PEAK_WEIGHT = 4.0

RAW_EDGE_MODELS = {"LSTM-GAT", "LSTM-GCN", "LSTM-Cheb"}


# ============================================================
# 日志
# ============================================================

SAVE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

LOG_DIR = SAVE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"train_{run_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)
program_start_time = time.time()


# ============================================================
# 数据加载（data_prepare.py 已完成：读取+切分+变换+缩放+缓存）
# ============================================================

if INIT_FROM_RAW or (USE_CACHE and not CACHE_FILE.exists()):
    logger.info("🔄 从原始 CSV 构建数据并缓存（含切分+变换+缩放）")
    cache = prepare_and_save_lamah_daily(
        str(DATA_ROOT),
        CACHE_FILE,
        seq_len=SEQ_LEN,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SPLIT_SEED,
        overwrite=INIT_FROM_RAW,
    )
elif USE_CACHE:
    logger.info("⚡ 从缓存加载数据")
    cache = load_lamah_from_cache(CACHE_FILE)
else:
    raise RuntimeError("现在要求所有读取/处理都在 data_prepare.py 完成，请使用 cache 流程。")

precip_df = cache["precip_df"]
temp_df = cache["temp_df"]
soil_df = cache["soil_df"]
runoff_df = cache["runoff_df"]  # transformed target (per-basin robust log)
static_df = cache["static_df"]
edge_index = cache["edge_index"]
edge_weight = cache["edge_weight"]
edge_weight_raw = cache.get("edge_weight_raw", None)
split = cache["split"]

basin_ids = list(static_df.index)

if edge_weight_raw is None:
    edge_weight_raw = edge_weight

# per-basin medians for inverse (runoff)
runoff_median_series = cache["meta"]["scalers"]["runoff"]["median_per_basin"]
# ensure order aligned to basin_ids
runoff_median_per_basin = runoff_median_series.loc[basin_ids].to_numpy(dtype=np.float64)

train_indices = split["train"]
val_indices = split["val"]
test_indices = split["test"]

logger.info(f"[Split] train/val/test = {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")


# ============================================================
# Dataset / DataLoader
# ============================================================

train_dataset = LamaHDataset(
    precip_df, temp_df, soil_df, runoff_df, static_df,
    seq_len=SEQ_LEN,
    indices=train_indices,
    sample_weights=None,
)

val_dataset = LamaHDataset(
    precip_df, temp_df, soil_df, runoff_df, static_df,
    seq_len=SEQ_LEN,
    indices=val_indices,
    sample_weights=None,
)

train_loader = create_dataloader(
    train_dataset,
    batch_size=BASE_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=2,  # keep small for 6GB
    shuffle=False,
    num_workers=4,
)


test_dataset = LamaHDataset(
    precip_df, temp_df, soil_df, runoff_df, static_df,
    seq_len=SEQ_LEN,
    indices=test_indices,
    sample_weights=None,
)
test_loader = create_dataloader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
)


# ============================================================
# 模型
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_edge_weight(model_name, edge_weight_norm, edge_weight_raw):
    if model_name in RAW_EDGE_MODELS:
        return edge_weight_raw
    return edge_weight_norm


# ============================================================
# 评估函数：target/pred 是 per-basin robust-log，需要 inverse 回 Q
# ============================================================

def evaluate(model, dataloader, edge_index, edge_weight, device, basin_ids, runoff_median_per_basin, *, min_valid=30):
    """
    Returns:
        val_mse_q  : scalar, computed in ORIGINAL Q (m^3/s) for LR scheduler
        nse_dict   : {basin_id: nse}
        stats      : dict of summary stats
    """
    model.eval()

    edge_index = edge_index.to(device, non_blocking=True)
    edge_weight = edge_weight.to(device, non_blocking=True)

    mse_num = 0.0
    mse_den = 0.0

    obs_sum = {bid: 0.0 for bid in basin_ids}
    obs_cnt = {bid: 0 for bid in basin_ids}
    nse_num = {bid: 0.0 for bid in basin_ids}
    nse_den = {bid: 0.0 for bid in basin_ids}

    # Pass 1: mean(Q) per basin
    for batch in dataloader:
        target = batch["target"].numpy()  # transformed
        mask = batch["mask"].numpy()

        q_obs = positive_robust_log_per_basin_inverse(
            target, runoff_median_per_basin, eps=1e-6
        )  # (B, N)

        for j, bid in enumerate(basin_ids):
            m = mask[:, j]
            if not np.any(m):
                continue
            obs_sum[bid] += q_obs[m, j].sum()
            obs_cnt[bid] += int(m.sum())

    obs_mean = {
        bid: (obs_sum[bid] / obs_cnt[bid]) if obs_cnt[bid] >= min_valid else None
        for bid in basin_ids
    }

    # Pass 2: forward + accumulate
    for batch in dataloader:
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        static = batch["static"].to(device, non_blocking=True)
        target = batch["target"].numpy()  # transformed
        mask = batch["mask"].numpy()

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=bool(device.type == "cuda")):
                pred = model(
                    dynamic_features=dynamic,
                    static_features=static,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

        pred_np = pred.cpu().numpy()  # transformed
        mask = mask & np.isfinite(pred_np) & np.isfinite(target)

        q_pred = positive_robust_log_per_basin_inverse(pred_np, runoff_median_per_basin, eps=1e-6)
        q_obs = positive_robust_log_per_basin_inverse(target, runoff_median_per_basin, eps=1e-6)

        mask = mask & np.isfinite(q_pred) & np.isfinite(q_obs)

        diff2 = (q_pred - q_obs) ** 2
        bad = mask & (~np.isfinite(diff2))
        if bad.any():
            logger.info(f"[EvalDebug] bad diff2 count = {bad.sum()}")

        # 🔒 CRITICAL: drop non-finite diff2 explicitly
        valid = mask & np.isfinite(diff2)

        mse_num += diff2[valid].sum()
        mse_den += valid.sum()

        for j, bid in enumerate(basin_ids):
            mu = obs_mean[bid]
            if mu is None:
                continue

            m = mask[:, j]
            if not np.any(m):
                continue

            o = q_obs[m, j]
            p = q_pred[m, j]

            nse_num[bid] += ((o - p) ** 2).sum()
            nse_den[bid] += ((o - mu) ** 2).sum()

        del dynamic, static, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_mse_q = mse_num / max(mse_den, 1.0)

    nse_dict = {
        bid: 1.0 - nse_num[bid] / nse_den[bid]
        for bid in basin_ids
        if obs_cnt[bid] >= min_valid and nse_den[bid] > 0.0
    }

    nse_vals = np.array(list(nse_dict.values()), dtype=np.float32)

    if nse_vals.size == 0:
        stats = {
            "num_valid_basins": 0,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "frac_gt0": np.nan,
            "frac_gt05": np.nan,
        }
        return val_mse_q, nse_dict, stats

    stats = {
        "num_valid_basins": int(nse_vals.size),
        "mean": float(np.mean(nse_vals)),
        "median": float(np.median(nse_vals)),
        "p25": float(np.percentile(nse_vals, 25)),
        "p75": float(np.percentile(nse_vals, 75)),
        "frac_gt0": float(np.mean(nse_vals > 0.0)),
        "frac_gt05": float(np.mean(nse_vals > 0.5)),
    }

    return val_mse_q, nse_dict, stats


# ============================================================
# 训练循环
# ============================================================

for model_name in MODEL_LIST:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg = MODEL_REGISTRY[model_name]
    ckpt_path = SAVE_DIR / cfg["ckpt"]
    edge_weight_use = select_edge_weight(model_name, edge_weight, edge_weight_raw)

    logger.info(f"\n===== [Model] {model_name} =====")

    model = cfg["class"](
        dynamic_input_dim=3,
        static_input_dim=static_df.shape[1],
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        lstm_layers=LSTM_LAYERS,
        gat_heads=GAT_HEADS,
        lstm_dropout=LSTM_DROPOUT,
        gnn_dropout=GNN_DROPOUT,
        cheb_k=CHEBK,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    start_epoch = 1
    if TRAIN_MODE == 1 and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start_epoch = RESUME_EPOCH + 1
        logger.info(f"🔁 从 epoch {RESUME_EPOCH} 恢复训练")
    else:
        logger.info("🆕 从头开始训练")

    best_val = float("inf")

    for epoch in range(start_epoch, MAX_EPOCH + 1):
        logger.info(f"Epoch {epoch:03d} started")
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            edge_index=edge_index,
            edge_weight=edge_weight_use,
            device=device,
            use_amp=USE_AMP,
            accum_steps=ACCUM_STEPS,
            peak_weighting=PEAK_WEIGHTING,
            peak_top_pct=PEAK_TOP_PCT,
            peak_weight=PEAK_WEIGHT,
        )

        val_mse, val_nse_dict, val_stats = evaluate(
            model,
            val_loader,
            edge_index,
            edge_weight_use,
            device,
            basin_ids,
            runoff_median_per_basin,
            min_valid=30,
        )

        scheduler.step(val_mse)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_MSE(Q)={val_mse:.4f} | "
            f"val_NSE_mean={val_stats['mean']:.3f} | "
            f"median={val_stats['median']:.3f} | "
            f"p25={val_stats['p25']:.3f} | "
            f"p75={val_stats['p75']:.3f} | "
            f"valid_basins={val_stats['num_valid_basins']} | "
            f"frac(NSE>0)={val_stats['frac_gt0']:.2%} | "
            f"frac(NSE>0.5)={val_stats['frac_gt05']:.2%} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={time.time() - t0:.1f}s"
        )

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"[BEST] val_MSE(Q)={best_val:.4f} -> saved {ckpt_path.name}")

    test_mse, test_nse_dict, test_stats = evaluate(
        model,
        test_loader,
        edge_index,
        edge_weight_use,
        device,
        basin_ids,
        runoff_median_per_basin,
        min_valid=30,
    )

    logger.info(
        f"[TEST] MSE(Q)={test_mse:.4f} | "
        f"NSE mean={test_stats['mean']:.3f} | "
        f"median={test_stats['median']:.3f} | "
        f"valid_basins={test_stats['num_valid_basins']}"
    )

    metrics_path = SAVE_DIR / f"test_metrics_{model_name.replace('-', '_')}.pt"
    torch.save(
        {
            "test_nse_per_basin": test_nse_dict,
            "test_nse_values": np.array([v for v in test_nse_dict.values() if np.isfinite(v)], dtype=np.float32),
            "test_stats": test_stats,
        },
        metrics_path,
    )

total_time = time.time() - program_start_time
logger.info(f"Training finished, total time = {timedelta(seconds=int(total_time))}")

if AUTO_PLOT:
    logger.info("📈 Auto-plotting NSE CDF and Figure 5 outputs...")
    from plot_nse_cdf import main as plot_main

    plot_main()
