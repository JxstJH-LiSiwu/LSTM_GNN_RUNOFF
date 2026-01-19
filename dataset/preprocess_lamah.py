# dataset/preprocess_lamah.py
import torch
from data_prepare import load_lamah_daily

DATA_ROOT = "/home/lisiwu/jxwork/1-gnn-lstm/dataset"
OUT_PATH = "/home/lisiwu/jxwork/1-gnn-lstm/dataset/lamah_cache.pt"

print("📦 Loading raw LamaH data (this is slow, only once)...")
precip_df, temp_df, soil_df, runoff_df, static_df = load_lamah_daily(DATA_ROOT)

cache = {
    "precip": precip_df.values.astype("float32"),
    "temp":   temp_df.values.astype("float32"),
    "soil":   soil_df.values.astype("float32"),
    "runoff": runoff_df.values.astype("float32"),
    "static": static_df.values.astype("float32"),
    "basin_ids": list(static_df.index),
}

torch.save(cache, OUT_PATH)
print(f"✅ Cache saved to {OUT_PATH}")
