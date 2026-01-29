import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from dataset.lamah_dataset import LamaHDataset
from dataset.data_prepare import (
    load_lamah_from_cache,
    positive_robust_log_per_basin_inverse,
)
from dataset.dataloader import create_dataloader

from src import MODEL_REGISTRY as BASE_MODEL_REGISTRY


# ============================================================
# Model registry (must match plot_nse_cdf.py)
#   - reuse src/__init__.py registry and add LSTM-SAGE alias
# ============================================================
MODEL_REGISTRY = {
    "LSTM": BASE_MODEL_REGISTRY["LSTM"],
    "LSTM-GAT": BASE_MODEL_REGISTRY["LSTM-GAT"],
    "LSTM-GCN": BASE_MODEL_REGISTRY["LSTM-GCN"],
    "LSTM-Cheb": BASE_MODEL_REGISTRY["LSTM-Cheb"],
    "LSTM-SAGE": BASE_MODEL_REGISTRY["LSTM-GraphSAGE"],
}

RAW_EDGE_MODELS = {"LSTM-GAT", "LSTM-GCN", "LSTM-Cheb"}


def parse_args():
    p = argparse.ArgumentParser(description="Offline inference and cache to .pt")
    p.add_argument("--cache_file", type=str, default="checkpoints/data_cache/lamah_daily.pt")
    p.add_argument("--save_dir", type=str, default="checkpoints/infer_cache")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--seq_len", type=int, default=180)
    p.add_argument("--lead_days", type=str, default="1")
    p.add_argument("--lead_day", type=str, default="", help="Alias for --lead_days")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--models", type=str, default="LSTM-GAT")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_override", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_q", action="store_true", default=True)
    p.add_argument("--no_save_q", action="store_false", dest="save_q")
    p.add_argument("--save_time", action="store_true", default=True)
    p.add_argument("--no_save_time", action="store_false", dest="save_time")

    # model init hyperparameters (plot defaults)
    p.add_argument("--dynamic_input_dim", type=int, default=3)
    p.add_argument("--forecast_input_dim", type=int, default=2)
    p.add_argument("--static_input_dim", type=int, default=-1)
    p.add_argument("--lstm_hidden_dim", type=int, default=128)
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--output_dim", type=int, default=1)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--lstm_dropout", type=float, default=0.2)
    p.add_argument("--gnn_dropout", type=float, default=0.2)
    p.add_argument("--cheb_k", type=int, default=3)
    p.add_argument("--num_hops", type=str, default="2")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[Warn] --device cuda requested but CUDA not available, fallback to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_models(models_str: str) -> list[str]:
    return [m.strip() for m in models_str.split(",") if m.strip()]


def parse_ckpt_override(s: str) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --ckpt_override: {e}")


def parse_int_list(s: str, *, name: str) -> list[int]:
    if s is None or str(s).strip() == "":
        raise ValueError(f"{name} is empty")
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid {name} value: {p}") from e
    return out


def parse_ckpt_tag(ckpt_path: Path) -> str:
    stem = ckpt_path.stem.lower()
    lead_m = re.search(r"lead[_-]?(\d+)", stem)
    hop_m = re.search(r"hop[_-]?(\d+)", stem)
    lead_tag = f"lead{lead_m.group(1)}" if lead_m else ""
    hop_tag = f"hop{hop_m.group(1)}" if hop_m else ""

    if "best" in stem:
        base = "best"
    else:
        m = re.search(r"epoch[_-]?(\d+)", stem)
        if m:
            base = f"epoch{m.group(1)}"
        elif "epoch" in stem:
            base = "epoch"
        elif "last" in stem:
            base = "last"
        else:
            base = "ckpt"

    extra = "_".join([t for t in [lead_tag, hop_tag] if t])
    return f"{base}_{extra}" if extra else base


def load_hparams_from_ckpt(ckpt_path: Path) -> dict:
    json_path = ckpt_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_ckpt_path(ckpt_dir: Path, base_name: str, lead_days: int, num_hops: int) -> Path:
    """
    Prefer lead/hop-specific ckpt if it exists, else fallback to base ckpt.
    base_name example: lstm_gat_model_epoch.pth
    """
    base_path = ckpt_dir / base_name
    stem = base_path.stem
    tagged = ckpt_dir / f"{stem}_lead{lead_days}_hop{num_hops}{base_path.suffix}"
    if tagged.exists():
        return tagged
    return base_path


def select_edge_weight(model_name: str, edge_weight, edge_weight_raw):
    if model_name in RAW_EDGE_MODELS:
        if edge_weight_raw is None:
            print("[Warn] edge_weight_raw not found in cache, fallback to normalized edge_weight.")
            return edge_weight, "norm"
        return edge_weight_raw, "raw"
    return edge_weight, "norm"


def predict_on_loader(model, dataloader, edge_index, edge_weight, device, amp: bool):
    model.eval()
    preds, targets, masks = [], [], []

    edge_index = edge_index.to(device, non_blocking=True)
    edge_weight = edge_weight.to(device, non_blocking=True)

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            static = batch["static"].to(device, non_blocking=True)
            target = batch["target"]
            mask = batch["mask"]

            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred = model(
                    dynamic_features=dynamic,
                    forecast_features=forecast,
                    static_features=static,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

            preds.append(pred.cpu())
            targets.append(target)
            masks.append(mask)

    obs_log = torch.cat(targets, dim=0).numpy()
    pred_log = torch.cat(preds, dim=0).numpy()
    mask_all = torch.cat(masks, dim=0).numpy().astype(bool)
    return obs_log, pred_log, mask_all


def resolve_split_list(split: str) -> list[str]:
    if split == "all":
        return ["train", "val", "test"]
    return [split]


def main():
    args = parse_args()
    if args.lead_day:
        args.lead_days = args.lead_day
    device = resolve_device(args.device)

    cache_file = Path(args.cache_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models = parse_models(args.models)
    ckpt_override = parse_ckpt_override(args.ckpt_override)
    lead_days_list = parse_int_list(args.lead_days, name="lead_days")
    num_hops_list = parse_int_list(args.num_hops, name="num_hops")

    # ----- load cache -----
    cache = load_lamah_from_cache(cache_file)
    precip_df = cache["precip_df"]
    temp_df = cache["temp_df"]
    soil_df = cache["soil_df"]
    runoff_df = cache["runoff_df"]
    static_df = cache["static_df"]
    edge_index = cache["edge_index"]
    edge_weight = cache["edge_weight"]
    edge_weight_raw = cache.get("edge_weight_raw", None)
    split_dict = cache["split"]

    basin_ids = list(static_df.index)
    runoff_median = (
        cache["meta"]["scalers"]["runoff"]["median_per_basin"]
        .loc[basin_ids]
        .to_numpy(dtype=np.float64)
    )

    # ----- build datasets/loaders per split/lead -----
    split_to_loader = {}
    for lead_days in lead_days_list:
        split_to_loader[lead_days] = {}
        for sp in resolve_split_list(args.split):
            indices = split_dict[sp]
            dataset = LamaHDataset(
                precip_df,
                temp_df,
                soil_df,
                runoff_df,
                static_df,
                seq_len=args.seq_len,
                lead_days=lead_days,
                indices=indices,
                sample_weights=None,
            )
            loader = create_dataloader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            indices_used = np.array(dataset.valid_t, dtype=np.int64)
            time_list = None
            if args.save_time:
                time_index = cache["meta"]["time_index"]
                time_index = np.asarray(time_index)
                idx = indices_used + int(lead_days)
                idx = idx[idx < len(time_index)]
                time_list = time_index[idx]
            split_to_loader[lead_days][sp] = (dataset, loader, indices_used, time_list)

    # ----- loop over splits and models -----
    for sp in resolve_split_list(args.split):
        for lead_days in lead_days_list:
            dataset, loader, indices_used, time_list = split_to_loader[lead_days][sp]
            for model_name in models:
                if model_name not in MODEL_REGISTRY:
                    raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

                cfg = MODEL_REGISTRY[model_name]

                for num_hops in num_hops_list:
                    override_key = f"{model_name}@lead{lead_days}_hop{num_hops}"
                    ckpt_path = ckpt_override.get(override_key, ckpt_override.get(model_name, ""))
                    if ckpt_path:
                        ckpt_path = Path(ckpt_path)
                    else:
                        ckpt_path = build_ckpt_path(
                            Path(args.ckpt_dir),
                            cfg["ckpt"],
                            int(lead_days),
                            int(num_hops),
                        )

                    if not ckpt_path.exists():
                        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                    ckpt_tag = parse_ckpt_tag(ckpt_path)
                    edge_weight_use, edge_mode = select_edge_weight(model_name, edge_weight, edge_weight_raw)

                    static_input_dim = static_df.shape[1] if args.static_input_dim < 0 else args.static_input_dim
                    base_hparams = {
                        "dynamic_input_dim": args.dynamic_input_dim,
                        "static_input_dim": static_input_dim,
                        "forecast_input_dim": args.forecast_input_dim,
                        "lstm_hidden_dim": args.lstm_hidden_dim,
                        "gnn_hidden_dim": args.gnn_hidden_dim,
                        "output_dim": args.output_dim,
                        "lstm_layers": args.lstm_layers,
                        "gat_heads": args.gat_heads,
                        "lstm_dropout": args.lstm_dropout,
                        "gnn_dropout": args.gnn_dropout,
                        "cheb_k": args.cheb_k,
                        "num_hops": int(num_hops),
                    }

                    hparams = load_hparams_from_ckpt(ckpt_path)
                    for k, v in hparams.items():
                        if k in base_hparams:
                            base_hparams[k] = v

                    print(
                        f"\n[Infer] model={model_name} split={sp} lead={lead_days} hop={num_hops} "
                        f"device={device} edge={edge_mode} ckpt={ckpt_path}"
                    )
                    t0 = time.time()

                    model = cfg["class"](**base_hparams).to(device)

                    ckpt = torch.load(ckpt_path, map_location=device)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        model.load_state_dict(ckpt["state_dict"])
                    else:
                        model.load_state_dict(ckpt)

                    obs_log, pred_log, mask = predict_on_loader(
                        model, loader, edge_index, edge_weight_use, device, args.amp
                    )

                    obs_q = pred_q = None
                    if args.save_q:
                        obs_q = positive_robust_log_per_basin_inverse(obs_log, runoff_median)
                        pred_q = positive_robust_log_per_basin_inverse(pred_log, runoff_median)

                    out_name = f"{model_name}_lead{lead_days}_hop{num_hops}.pt"
                    out_path = save_dir / out_name
                    if out_path.exists() and not args.overwrite:
                        print(f"[Skip] {out_path} exists (use --overwrite to replace).")
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    meta = {
                        "model_name": model_name,
                        "ckpt_path": str(ckpt_path),
                        "ckpt_tag": ckpt_tag,
                        "tag": args.tag,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device": str(device),
                        "batch_size": int(args.batch_size),
                        "num_workers": int(args.num_workers),
                        "amp": bool(args.amp),
                        "dynamic_input_dim": int(base_hparams["dynamic_input_dim"]),
                        "forecast_input_dim": int(base_hparams["forecast_input_dim"]),
                        "static_input_dim": int(base_hparams["static_input_dim"]),
                        "lstm_hidden_dim": int(base_hparams["lstm_hidden_dim"]),
                        "gnn_hidden_dim": int(base_hparams["gnn_hidden_dim"]),
                        "output_dim": int(base_hparams["output_dim"]),
                        "lstm_layers": int(base_hparams["lstm_layers"]),
                        "gat_heads": int(base_hparams["gat_heads"]),
                        "lstm_dropout": float(base_hparams["lstm_dropout"]),
                        "gnn_dropout": float(base_hparams["gnn_dropout"]),
                        "cheb_k": int(base_hparams["cheb_k"]),
                        "num_hops": int(base_hparams["num_hops"]),
                        "cache_file": str(cache_file),
                        "code_ref": "infer_cache.py",
                    }

                    cache_out = {
                        "obs_log": obs_log,
                        "pred_log": pred_log,
                        "mask": mask,
                        "obs_q": obs_q,
                        "pred_q": pred_q,
                        "basin_ids": basin_ids,
                        "seq_len": int(args.seq_len),
                        "lead_days": int(lead_days),
                        "split": sp,
                        "indices": indices_used,
                        "time": time_list,
                        "edge_weight_mode": edge_mode,
                        "meta": meta,
                    }

                    torch.save(cache_out, out_path)
                    elapsed = time.time() - t0
                    valid_cnt = int(mask.sum())
                    print(f"[Saved] {out_path} | obs/pred shape={obs_log.shape} | valid={valid_cnt} | time={elapsed:.1f}s")

                    del model, obs_log, pred_log, mask, obs_q, pred_q
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
