# train.py
from pathlib import Path
import logging
import os
import time
from datetime import datetime, timedelta
import argparse

import numpy as np
import torch

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

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "dataset"
SAVE_DIR = BASE_DIR / "checkpoints"

USE_CACHE = True
CACHE_DIR = SAVE_DIR / "data_cache"
CACHE_FILE = CACHE_DIR / "lamah_daily.pt"
INIT_FROM_RAW = False

NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))  # 保留兼容，但不再用于 Pool

# -------------------- Training control --------------------
TRAIN_MODE = 0        # default: 0 -> train from scratch; 1 -> resume from checkpoint
RESUME_EPOCH = 0      # default: 0 -> resume starting epoch index (used when TRAIN_MODE=1)
MAX_EPOCH = 65        # default: 65 -> total training epochs
MODEL_LIST = ["LSTM", "LSTM-GAT", "LSTM-GCN", "LSTM-Cheb", "LSTM-GraphSAGE"]  # default list

# FOR PAPER
# -------------------- Model hyperparameters --------------------
LSTM_HIDDEN_DIM = 128  # default: 128 -> LSTM hidden size (d_lstm)
LSTM_LAYERS = 2        # default: 2 -> number of LSTM layers
GNN_HIDDEN_DIM = 64    # default: 64 -> GNN hidden size
GAT_HEADS = 4          # default: 4 -> GAT attention heads
LSTM_DROPOUT = 0.2    # default: 0.2 -> LSTM dropout (applied between layers)
GNN_DROPOUT = 0.2      # default: 0.2 -> fusion + GNN dropout
OUTPUT_DIM = 1         # default: 1 -> per-node scalar prediction
CHEBK = 3              # default: 3 -> ChebNet K (only for LSTM-Cheb)
NUM_HOPS = int(os.getenv("HOP", "2"))  # default: 2 -> multi-hop routing depth (env: HOP)

# -------------------- Optimization hyperparameters --------------------
USE_AMP = True         # default: True -> use mixed precision on CUDA
BASE_BATCH_SIZE = 12   # default: 12 -> per-step batch size
ACCUM_STEPS = 2        # default: 2 -> gradient accumulation steps
LEARNING_RATE = 5e-4   # default: 5e-4 -> Adam lr
MIN_LR = 1e-5          # default: 1e-5 -> scheduler min lr
LR_PATIENCE = 3        # default: 3 -> ReduceLROnPlateau patience

# -------------------- Data/forecast hyperparameters --------------------
SEQ_LEN = 180          # default: 180 -> lookback days
FORECAST_INPUT_DIM = 2 # default: 2 -> forecast features (precip + temp)
LEAD_DAYS = int(os.getenv("LEAD_DAYS", "1"))  # default: 1 -> forecast horizon in days (env: LEAD_DAYS)
TRAIN_RATIO = 0.7      # default: 0.7 -> train split ratio
VAL_RATIO = 0.15       # default: 0.15 -> validation split ratio
TEST_RATIO = 0.15      # default: 0.15 -> test split ratio
SPLIT_SEED = 42        # default: 42 -> split random seed

AUTO_PLOT = False  # 仍保留，但默认不在每个模型进程里画（见 PLOT_AFTER_TRAIN）

# -------------------- Loss shaping hyperparameters --------------------
PEAK_WEIGHTING = True  # default: True -> apply peak weighting
PEAK_TOP_PCT = 0.025   # default: 0.025 -> top percentile for peak emphasis
PEAK_WEIGHT = 4.0      # default: 4.0 -> weight multiplier for peaks

RAW_EDGE_MODELS = {"LSTM-GAT", "LSTM-GCN", "LSTM-Cheb"}
MEAN_LOSS_WEIGHT = float(os.getenv("MEAN_LOSS_WEIGHT", "0.0"))  # default: 0.0 -> mean penalty weight (env)
RUN_TAG = os.getenv("RUN_TAG", "").strip()

# DataLoader workers：单进程下可安全开多 worker（40 核机器建议每进程 6~10）
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "5"))


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

def load_data():
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

    runoff_median_series = cache["meta"]["scalers"]["runoff"]["median_per_basin"]
    runoff_median_per_basin = runoff_median_series.loc[basin_ids].to_numpy(dtype=np.float64)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    logger.info(f"[Split] train/val/test = {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")

    return (
        precip_df,
        temp_df,
        soil_df,
        runoff_df,
        static_df,
        edge_index,
        edge_weight,
        edge_weight_raw,
        basin_ids,
        runoff_median_per_basin,
        train_indices,
        val_indices,
        test_indices,
    )


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
        forecast = batch["forecast"].to(device, non_blocking=True)
        static = batch["static"].to(device, non_blocking=True)
        target = batch["target"].numpy()  # transformed
        mask = batch["mask"].numpy()

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=bool(device.type == "cuda")):
                pred = model(
                    dynamic_features=dynamic,
                    forecast_features=forecast,
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

        del dynamic, forecast, static, pred
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
# 训练循环：单模型训练函数
# ============================================================

def train_single_model(model_name, device_id=0, gpu_index=0):
    """在当前进程（单GPU）上训练单个模型。

    外部用 CUDA_VISIBLE_DEVICES 控制 GPU 绑定；进程内默认使用 cuda:0。
    """
    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # 为每个进程/模型创建独立日志
    model_log_dir = LOG_DIR / f"gpu{gpu_index}"
    model_log_dir.mkdir(exist_ok=True)
    model_run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日志文件名加入 model + RUN_TAG + timestamp，避免并发冲突
    tag_part = f"_{RUN_TAG}" if RUN_TAG else ""
    model_log_path = model_log_dir / f"{model_name}{tag_part}_{model_run_time}.log"

    model_logger = logging.getLogger(f"gpu{gpu_index}_{model_name}_{tag_part}_{model_run_time}")
    model_logger.setLevel(logging.INFO)
    model_logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(model_log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    model_logger.addHandler(file_handler)
    model_logger.addHandler(console_handler)

    model_logger.info(f"Starting training {model_name} on {device} | MEAN_LOSS_WEIGHT={MEAN_LOSS_WEIGHT} | RUN_TAG={RUN_TAG}")
    model_logger.info(f"DataLoader NUM_WORKERS={NUM_WORKERS}")

    # 载入数据（每进程一次；最小侵入不做共享内存优化）
    (
        precip_df,
        temp_df,
        soil_df,
        runoff_df,
        static_df,
        edge_index,
        edge_weight,
        edge_weight_raw,
        basin_ids,
        runoff_median_per_basin,
        train_indices,
        val_indices,
        test_indices,
    ) = load_data()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg = MODEL_REGISTRY[model_name]
    ckpt_path = SAVE_DIR / cfg["ckpt"]
    if RUN_TAG:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_{RUN_TAG}{ckpt_path.suffix}")

    edge_weight_use = select_edge_weight(model_name, edge_weight, edge_weight_raw)

    # 创建模型
    model = cfg["class"](
        dynamic_input_dim=3,
        static_input_dim=static_df.shape[1],
        forecast_input_dim=FORECAST_INPUT_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        lstm_layers=LSTM_LAYERS,
        gat_heads=GAT_HEADS,
        lstm_dropout=LSTM_DROPOUT,
        gnn_dropout=GNN_DROPOUT,
        cheb_k=CHEBK,
        num_hops=NUM_HOPS,  # MODIFIED: multi-hop routing with residual + layernorm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    start_epoch = 1
    if TRAIN_MODE == 1 and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start_epoch = RESUME_EPOCH + 1
        model_logger.info(f"🔁 从 epoch {RESUME_EPOCH} 恢复训练")
    else:
        model_logger.info("🆕 从头开始训练")

    best_val = float("inf")

    train_dataset = LamaHDataset(
        precip_df, temp_df, soil_df, runoff_df, static_df,
        seq_len=SEQ_LEN,
        lead_days=LEAD_DAYS,
        indices=train_indices,
        sample_weights=None,
    )
    val_dataset = LamaHDataset(
        precip_df, temp_df, soil_df, runoff_df, static_df,
        seq_len=SEQ_LEN,
        lead_days=LEAD_DAYS,
        indices=val_indices,
        sample_weights=None,
    )
    test_dataset = LamaHDataset(
        precip_df, temp_df, soil_df, runoff_df, static_df,
        seq_len=SEQ_LEN,
        lead_days=LEAD_DAYS,
        indices=test_indices,
        sample_weights=None,
    )

    # val/test loader：同样开 worker（注意：评估里主要在 CPU 做 inverse + NSE，开 worker 能提速）
    val_loader = create_dataloader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # 训练循环
    for epoch in range(start_epoch, MAX_EPOCH + 1):
        model_logger.info(f"Epoch {epoch:03d} started")
        t0 = time.time()

        train_loader = create_dataloader(
            train_dataset,
            batch_size=BASE_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

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
            mean_loss_weight=MEAN_LOSS_WEIGHT,
        )

        val_mse, val_nse_dict, val_stats = evaluate(
            model, val_loader, edge_index, edge_weight_use, device, basin_ids,
            runoff_median_per_basin, min_valid=30,
        )

        scheduler.step(val_mse)

        model_logger.info(
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
            model_logger.info(f"[BEST] val_MSE(Q)={best_val:.4f} -> saved {ckpt_path.name}")

    # 测试阶段
    test_mse, test_nse_dict, test_stats = evaluate(
        model, test_loader, edge_index, edge_weight_use, device, basin_ids,
        runoff_median_per_basin, min_valid=30,
    )
    model_logger.info(
        f"[TEST] MSE(Q)={test_mse:.4f} | "
        f"NSE mean={test_stats['mean']:.3f} | "
        f"median={test_stats['median']:.3f} | "
        f"valid_basins={test_stats['num_valid_basins']}"
    )

    # 保存测试指标
    metrics_name = f"test_metrics_{model_name.replace('-', '_')}"
    if RUN_TAG:
        metrics_name = f"{metrics_name}_{RUN_TAG}"
    metrics_path = SAVE_DIR / f"{metrics_name}.pt"
    torch.save(
        {
            "test_nse_per_basin": test_nse_dict,
            "test_nse_values": np.array([v for v in test_nse_dict.values() if np.isfinite(v)], dtype=np.float32),
            "test_stats": test_stats,
        },
        metrics_path,
    )
    model_logger.info(f"Saved test metrics to {metrics_path}")

    return model_name, best_val, test_stats["mean"]


# ============================================================
# 主入口：每个进程只训练一个模型
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "").strip(),
                   help="Model name (e.g., LSTM, LSTM-GAT, LSTM-GCN, LSTM-Cheb, LSTM-GraphSAGE). "
                        "If empty, will use MODEL_NAME env var; if still empty, will use MODEL_LIST[0].")
    p.add_argument("--lead-days", type=int, default=LEAD_DAYS,
                   help="Forecast horizon in days (1-7). Overrides LEAD_DAYS env var.")
    # 兼容你之前脚本的 NUM_GPUS / RUN_TAG / MEAN_LOSS_WEIGHT 均走 env，不在这里重复
    return p.parse_args()


def main():
    args = parse_args()

    model_name = args.model if args.model else MODEL_LIST[0]
    global LEAD_DAYS
    LEAD_DAYS = int(args.lead_days)
    if not (1 <= LEAD_DAYS <= 7):
        raise ValueError(f"--lead-days must be in [1,7], got {LEAD_DAYS}")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    # 仅用于日志显示：外部通常用 CUDA_VISIBLE_DEVICES 绑定
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_index = 0
    if visible:
        # 仅作标签展示，不影响实际 device 选择（我们始终用 cuda:0）
        try:
            gpu_index = int(visible.split(",")[0])
        except Exception:
            gpu_index = 0

    logger.info(f"🚀 Single-process / single-model run: model={model_name} | CUDA_VISIBLE_DEVICES={visible} | NUM_WORKERS={NUM_WORKERS}")
    result = train_single_model(model_name, device_id=0, gpu_index=gpu_index)

    logger.info("\n===== Training Result =====")
    logger.info(f"{result[0]}: best_val_MSE={result[1]:.4f}, test_NSE_mean={result[2]:.3f}")

    total_time = time.time() - program_start_time
    logger.info(f"Training finished, total time = {timedelta(seconds=int(total_time))}")

    # 避免 5 个模型进程同时画图抢文件：只有你明确开启才画
    ##if AUTO_PLOT and os.getenv("PLOT_AFTER_TRAIN", "0") == "1":
    ##    logger.info("📈 Auto-plotting NSE CDF and Figure 5 outputs...")
    ##    from plot_nse_cdf import main as plot_main
    ##    plot_main()


if __name__ == "__main__":
    main()
