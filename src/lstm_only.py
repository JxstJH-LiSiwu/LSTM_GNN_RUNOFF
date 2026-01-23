import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LSTM Encoder (Dynamic only) - Paper-aligned
#   Input : (B*N, T, F_dyn)
#   Output: (B*N, d_lstm)
# ============================================================

class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x_dyn: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_dyn)
        z = out[:, -1, :]  # last timestep
        return z


# ============================================================
# Static Encoder - Paper-aligned
# ============================================================

class StaticEncoder(nn.Module):
    def __init__(self, static_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(static_input_dim, out_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(s))


# ============================================================
# Fusion (Paper-aligned)
# ============================================================

class FusionLayer(nn.Module):
    def __init__(self, d_lstm: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(d_lstm * 2, d_lstm)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_dyn: torch.Tensor, s_tilde: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_dyn, s_tilde], dim=-1)
        h = F.relu(self.proj(h))
        h = self.drop(h)
        return h


# ============================================================
# Full Model: LSTM + Static Encoder + Fusion (NO GNN routing)
# - Keep forward signature identical to GNN variants (edge_index/edge_weight accepted but unused)
# ============================================================

class CombinedLSTMWithStatic2Hop(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        lstm_hidden_dim: int,
        gnn_hidden_dim: int,    # unused, kept for minimal intrusion
        output_dim: int,
        lstm_layers: int,
        gat_heads: int,         # unused, kept for minimal intrusion
        lstm_dropout: float,
        gnn_dropout: float,     # used as fusion dropout
        cheb_k: int = 3,        # unused, kept for API compatibility
        num_hops: int = 2,      # MODIFIED: accept HOP (ignored for LSTM-only)
    ):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(
            input_dim=dynamic_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        self.static_encoder = StaticEncoder(
            static_input_dim=static_input_dim,
            out_dim=lstm_hidden_dim,
        )

        self.fusion = FusionLayer(
            d_lstm=lstm_hidden_dim,
            dropout=gnn_dropout,
        )

        # NO routing: directly output from fused embedding
        self.out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(
        self,
        dynamic_features: torch.Tensor,  # (B, T, N, F_dyn)
        static_features: torch.Tensor,   # (B, N, F_static)
        edge_index: torch.Tensor,        # accepted but unused
        edge_weight: torch.Tensor = None # accepted but unused
    ) -> torch.Tensor:
        B, T, N, F_dyn = dynamic_features.shape

        dyn = dynamic_features.permute(0, 2, 1, 3).reshape(B * N, T, F_dyn)
        sta = static_features.reshape(B * N, -1)

        z_dyn = self.lstm_encoder(dyn)      # (B*N, d_lstm)
        s_tilde = self.static_encoder(sta)  # (B*N, d_lstm)

        node_embed = self.fusion(z_dyn, s_tilde)  # (B*N, d_lstm)

        pred = self.out(node_embed).view(B, N)
        return pred
