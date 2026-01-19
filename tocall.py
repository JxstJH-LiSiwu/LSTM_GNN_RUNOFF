# ======= 在 train.py 末尾追加（训练结束后）=======
from pathlib import Path
import numpy as np
import torch
from dataset.data_prepare import (
    load_lamah_daily,
    prepare_and_save_lamah_daily,
    load_lamah_from_cache,
    split_time_indices
)

from src.eval_utils import predict_on_loader
from src.metrics import compute_nse_per_basin
from src.plot_utils import plot_nse_cdf_multi_model
from dataset.lamah_dataset import LamaHDataset
from dataset.dataloader import create_dataloader
from src.lstm_gat import CombinedLSTMGATWithStatic2Hop
from plot_nse_cdf import run_nse_cdf_evaluation

DATA_ROOT = Path("/home/lisiwu/jxwork/1-gnn-lstm/dataset")     # LamaH 原始数据
SAVE_DIR  = Path("/home/lisiwu/jxwork/1-gnn-lstm/checkpoints")# 模型与日志
USE_CACHE = True                                             # 是否启用缓存
CACHE_DIR = SAVE_DIR / "data_cache"
CACHE_FILE = CACHE_DIR / "lamah_daily.pt"
INIT_FROM_RAW = False          


if INIT_FROM_RAW or (USE_CACHE and not CACHE_FILE.exists()):
    data = prepare_and_save_lamah_daily(
        DATA_ROOT,
        CACHE_FILE,
        overwrite=INIT_FROM_RAW,
    )
elif USE_CACHE:

    data = load_lamah_from_cache(CACHE_FILE)
else:

    data = load_lamah_daily(DATA_ROOT)

(
    precip_df,
    temp_df,
    soil_df,
    runoff_df,
    static_df,
    edge_index,
    edge_weight,
) = data                              # 强制重建缓存

TRAIN_MODE = 0           # 0: 从头训练; 1: 继续训练
RESUME_EPOCH = 0
MAX_EPOCH = 150

# ---------- 模型结构 ----------
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
GNN_HIDDEN_DIM = 64
GAT_HEADS = 4
LSTM_DROPOUT = 0.35
GNN_DROPOUT = 0.2
OUTPUT_DIM = 1

# ---------- 训练超参数 ----------
USE_AMP = True
ACCUM_STEPS = 2
BASE_BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
LR_PATIENCE = 5

# ---------- 序列 & 数据划分 ----------
SEQ_LEN = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


split = split_time_indices(
    precip_df.index,
    seq_len=SEQ_LEN,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
)

# -------------------------
# 1) 构建 test_dataset / test_loader
# -------------------------
test_indices = split["test"]  # 你上面 split_time_indices 已经算出来了 :contentReference[oaicite:6]{index=6}

test_dataset = LamaHDataset(
    precip_df, temp_df, soil_df, runoff_df, static_df,
    seq_len=SEQ_LEN,
    indices=test_indices,
    sample_weights=None,  # 测试不需要 loss 权重
)

test_loader = create_dataloader(
    test_dataset,
    batch_size=24,
    shuffle=False,     # 关键：按时间顺序评估
    num_workers=4,
    pin_memory=True,
)

model = CombinedLSTMGATWithStatic2Hop(
    dynamic_input_dim=3,
    static_input_dim=static_df.shape[1],
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    gnn_hidden_dim=GNN_HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    lstm_layers=LSTM_LAYERS,
    gat_heads=GAT_HEADS,
    lstm_dropout=LSTM_DROPOUT,
    gnn_dropout=GNN_DROPOUT,
).to(device)
# -------------------------
# 2) 加载已训练好的模型权重（以当前 CKPT_PATH 为例）
# -------------------------
CKPT_PATH = SAVE_DIR / "lstm_gat_model_epoch.pth"
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.to(device)

# -------------------------
# 3) 跑全集预测，拿到 [T_total, N]
# -------------------------
observed, predicted, mask = predict_on_loader(
    model=model,
    dataloader=test_loader,
    edge_index=edge_index,
    edge_weight=edge_weight,
    device=device,
)

# -------------------------
# 4) （可选）如果论文 NSE 在“原始流量尺度”算：
#    你 Dataset 的 target 是 log1p(q) :contentReference[oaicite:7]{index=7}
#    则需要 expm1 还原再算 NSE
# -------------------------
USE_ORIGINAL_Q = True
if USE_ORIGINAL_Q:
    observed_nse = np.expm1(observed)
    predicted_nse = np.expm1(predicted)
else:
    observed_nse = observed
    predicted_nse = predicted

# -------------------------
# 5) per-basin NSE
# -------------------------
basin_ids = list(static_df.index)  # 你 static_df.index 就是 ID_xxx :contentReference[oaicite:8]{index=8}
nse_gat = compute_nse_per_basin(
    observed=observed_nse,
    predicted=predicted_nse,
    mask=mask,
    basin_ids=basin_ids,
    min_valid=10,
)

# -------------------------
# 6) 多模型 NSE-CDF：组织字典并绘图
#    （这里只演示 LSTM-GAT，你其他模型重复同样流程即可）
# -------------------------
# 假设你也算出了其它模型的 nse_lstm / nse_gcn / nse_graphsage / nse_cheb
# 这里用同一个 nse_gat 临时占位（你实际使用时替换掉）
####nse_lstm = nse_gat
####nse_gcn = nse_gat
####nse_graphsage = nse_gat
####nse_cheb = nse_gat
####
nse_dict_per_model = {
    ####"LSTM": nse_lstm,
    "LSTM-GAT": nse_gat,
   #### "LSTM-GCN": nse_gcn,
   #### "LSTM-GraphSAGE": nse_graphsage,
    ####"LSTM-ChebNet": nse_cheb,
}


run_nse_cdf_evaluation(
    model=model,
    device=device,
    edge_index=edge_index,
    edge_weight=edge_weight,
    basin_ids=basin_ids,
    save_dir=SAVE_DIR,
    seq_len=60,
    batch_size=32,
)