# dataset/preprocess_lamah.py
import torch
from pathlib import Path
from data_prepare import load_lamah_daily

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "dataset"
OUT_PATH = DATA_ROOT / "lamah_cache.pt"

print("📦 Loading raw LamaH data (this is slow, only once)...")
precip_df, temp_df, soil_df, runoff_df, static_df = load_lamah_daily(str(DATA_ROOT))

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
