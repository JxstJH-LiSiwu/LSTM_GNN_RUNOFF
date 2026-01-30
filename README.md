# LSTM–GNN Rainfall–Runoff Modeling (LamaH Danube)

## Overview
This repository implements a research-grade rainfall–runoff modeling framework that combines per-basin LSTM encoders with graph neural network (GNN) routing on a river network. The goal is to reproduce and extend the ideas in “A GNN Routing Module Is All You Need for LSTM Rainfall–Runoff Models” using LamaH (Danube) daily data.

The pipeline follows a hydrologically motivated decomposition: LSTM models runoff generation at each sub-basin from meteorological inputs, while the GNN propagates and routes signals along the river network using travel-time–informed edges. The code is designed for reproducible experiments and supports multiple GNN backbones with a unified training/evaluation workflow.

## Method & Architecture
**Core idea**
- **LSTM (per basin)**: learns runoff generation from dynamic forcings (precipitation, temperature, soil moisture).
- **GNN (river network)**: models runoff routing and spatial dependencies across basins via graph convolution over the river network.

**Supported variants**
- `LSTM` (no routing)
- `LSTM-GAT`
- `LSTM-GCN`
- `LSTM-GraphSAGE`
- `LSTM-ChebNet`

**High-level flow (per time step)**

```
Dynamic forcing (precip/temp/soil) ─┐
                                    ├─ LSTM (per basin) ─┐
Static attributes (59 features) ────┘                    ├─ Fusion → GNN routing → Q̂
                                                        └─ Residual multi-hop routing
```

Routing uses `edge_index`/`edge_weight` derived from river network connectivity and a travel-time proxy (Kirpich-based). Multi-hop routing is supported via the `HOP` environment variable (default `HOP=2`).

## Dataset
**Source**: LamaH (Danube) basin  
**Time span**: 1987–2017 (daily)

**Dynamic variables** (per basin, daily)
- precipitation (`prec`)
- 2 m mean temperature (`2m_temp_mean`)
- soil water content (`volsw_123`)

**Static attributes**
- 59 basin attributes from `Catchment_attributes.csv` (scaled via min–max)

**River network**
- `edge_index`: upstream → downstream connectivity
- `edge_weight`: inverse travel-time proxy derived from `Stream_dist.csv` (Kirpich formula)
- both normalized and raw weights are stored; some models use raw weights by design

Expected dataset structure (see `dataset/data_prepare.py`):
```
dataset/
  222.csv
  B_basins_diff_upstrm_all/
    1_attributes/Catchment_attributes.csv
    1_attributes/Stream_dist.csv
    2_timeseries/daily/*.csv
  D_gauges/2_timeseries/daily/*.csv
```

## Project Structure
```
.
├── train.py                    # training entry point (single model per run)
├── run_queue.sh                # FIFO multi-GPU scheduling helper
├── dataset/
│   ├── data_prepare.py         # raw read, split, transforms, cache
│   ├── lamah_dataset.py        # PyTorch Dataset
│   ├── dataloader.py           # dataloader helpers
│   ├── preprocess_lamah.py     # optional raw cache utility
│   └── check_data.py           # sanity checks for raw data
├── src/
│   ├── lstm_only.py            # LSTM-only model
│   ├── lstm_gat.py             # LSTM + GAT routing
│   ├── lstm_gcn.py             # LSTM + GCN routing
│   ├── lstm_sage.py            # LSTM + GraphSAGE routing
│   ├── lstm_cheb.py            # LSTM + ChebNet routing
│   ├── train_one_epoch.py      # training loop
│   ├── metrics.py              # NSE/KGE metrics
│   └── losses.py               # masked loss helpers
├── plot_nse_cdf.py             # NSE-CDF plotting
├── plot_figure6.py             # paper-style figure
├── plot_figure9_hydrograph.py  # hydrograph plotting
├── infer_cache.py              # offline inference cache (.pt)
├── analyze_forecast_impact.py  # forecast vs no-forecast analysis
├── select_hydrograph_stations.py # auto-select A,B->C triplets for hydrograph
├── eval_mlw_metrics.py         # evaluation across checkpoints
└── analyze_bad_nse_basins.py   # diagnostic analysis
```

## Environment & Dependencies
- **Python**: `>=3.12` (see `pyproject.toml`)
- **Core**: PyTorch, PyTorch-Geometric, NumPy, Pandas, SciPy, scikit-learn
- **Plotting/geo**: matplotlib, geopandas, shapely, pyproj

Example (uv):
```bash
uv sync
```

Example (pip/venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.lock
```

## Quick Start
1) **Prepare data cache**
```bash
python train.py
```
`train.py` will build a cache at `checkpoints/data_cache/lamah_daily.pt` on first run.

2) **Train a model**
```bash
python train.py --model LSTM-GAT
```
or via environment variable:
```bash
MODEL_NAME=LSTM-GCN python train.py
```

2b) **No-forecast training (still predicts t+1)**
```bash
python train.py --model LSTM --lead-days 0
```

3) **Control routing hops (optional)**
```bash
HOP=3 python train.py --model LSTM-Cheb
```
Default is `HOP=2`.

4) **Multi-GPU FIFO (optional)**
```bash
bash run_queue.sh
```

## Evaluation
**Metrics implemented**
- **NSE (Nash–Sutcliffe Efficiency)**: goodness of fit vs. mean-baseline
- **KGE (Kling–Gupta Efficiency)**: correlation, bias, variability
- MSE/RMSE variants appear in analysis scripts

NSE is defined as:
\[
\mathrm{NSE} = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
\]

Generate NSE-CDF plot:
```bash
python plot_nse_cdf.py
```

## Offline Inference Cache (for fast plotting)
Generate cached predictions to avoid re-running inference inside plotting scripts:
```bash
python infer_cache.py --models LSTM,LSTM-GAT --lead_days 1 --num_hops 3
```
Outputs are saved to `checkpoints/infer_cache/` with names like:
```
LSTM_lead1_hop3.pt
LSTM-GAT_lead1_hop3.pt
```

## Forecast vs No-Forecast Analysis
Analyze and visualize the impact of forecast inputs (lead=1 by default):
```bash
python analyze_forecast_impact.py \
  --infer_dir checkpoints/infer_cache \
  --out_dir checkpoints/infer_cache/analysis_forecast
```
This produces:
- NSE CDFs (forecast vs no-forecast)
- delta histograms/scatter plots
- summary CSV

## Auto-select Hydrograph Stations (A,B -> C)
Select triplets where two upstream stations drain into a downstream station C:
```bash
python select_hydrograph_stations.py --lead 1 --hop 3
```
Then plot hydrographs using cached `.pt`:
```bash
python plot_figure9_hydrograph.py \
  --triplet_file checkpoints/infer_cache/selected_triplets_lead1_hop3.txt \
  --infer_dir checkpoints/infer_cache \
  --lead 1 --hop 3 \
  --compare_noforecast
```

## Reproducibility Notes
- **Paper-aligned**: LSTM per basin + GNN routing on river network; daily LamaH (Danube) with 1987–2017 window.
- **Transforms**: per-basin robust log scaling for precipitation/runoff; min–max scaling for temperature, soil, and static features (train-only fit).
- **Splits**: 70/15/15 time split; test is the last contiguous segment; train/val are randomized within the early period.
- **Engineering choices**: cached preprocessing (`checkpoints/data_cache/lamah_daily.pt`), AMP optional, FIFO GPU scheduling script.
- **Randomness**: train/val split controlled by `SPLIT_SEED` in `train.py`.
- **Memory/compute**: LSTM+GNN models can be GPU-intensive; adjust batch size and `NUM_WORKERS` as needed.


## Disclaimer
all idea from this paper "A GNN Routing Module Is All You Need for LSTM Rainfall–Runoff Models",
This project is intended for research and educational use only. It is not designed or validated for operational hydrologic forecasting or flood decision support.
