import torch
from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    persistent_workers=False,
):
    """
    batch output shapes:
    dynamic: [B, N, seq_len, 3]
    forecast:[B, N, 2]
    static:  [B, N, 58]
    target:  [B, N]
    mask:    [B, N]
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
