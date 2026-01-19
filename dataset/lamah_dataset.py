# dataset/lamah_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class LamaHDataset(Dataset):
    """
    One sample = one time step (whole basin graph)

    IMPORTANT (per your requirement):
    - This Dataset DOES NOT apply any log / inverse-log / scaling.
    - All transformations are done in data_prepare.py and cached.
    """

    def __init__(
        self,
        precip_df,
        temp_df,
        soil_df,
        runoff_df,
        static_df,
        seq_len,
        indices=None,
        sample_weights=None,
    ):
        self.seq_len = seq_len
        self.sample_weights = sample_weights

        self.basin_ids = list(static_df.index)
        self.N = len(self.basin_ids)

        # All are already processed (scaled/transformed) in data_prepare.py
        self.precip = precip_df[self.basin_ids].values.astype(np.float32)
        self.temp   = temp_df[self.basin_ids].values.astype(np.float32)
        self.soil   = soil_df[self.basin_ids].values.astype(np.float32)
        self.runoff = runoff_df[self.basin_ids].values.astype(np.float32)  # transformed target (may contain NaN)

        self.static = static_df.values.astype(np.float32)
        self.T = self.precip.shape[0]

        if indices is None:
            self.valid_t = list(range(seq_len, self.T - 1))
        else:
            self.valid_t = list(indices)

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = self.valid_t[idx]

        x_dyn = np.stack(
            [
                self.precip[t - self.seq_len + 1 : t + 1],
                self.temp[t - self.seq_len + 1 : t + 1],
                self.soil[t - self.seq_len + 1 : t + 1],
            ],
            axis=-1,
        )  # (seq_len, N, 3)

        y = self.runoff[t + 1]  # (N,) already transformed
        mask = np.isfinite(y) # validity mask

        weight = (
            float(self.sample_weights[t])
            if self.sample_weights is not None
            else 1.0
        )

        return {
            "dynamic": torch.from_numpy(x_dyn),
            "static":  torch.from_numpy(self.static),
            "target":  torch.from_numpy(y),              # transformed target
            "mask":    torch.from_numpy(mask),
            "weight":  torch.tensor(weight, dtype=torch.float32),
        }
