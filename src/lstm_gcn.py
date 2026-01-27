import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ============================================================
# LSTM Encoder (Dynamic only)
# ============================================================

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
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
        return out[:, -1, :]


# ============================================================
# Static Encoder
# ============================================================

class StaticEncoder(nn.Module):
    def __init__(self, static_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(static_input_dim, out_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(s))


# ============================================================
# Forecast Encoder - tomorrow's meteorological features
# ============================================================

class ForecastEncoder(nn.Module):
    def __init__(self, forecast_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(forecast_input_dim, out_dim)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(f))


# ============================================================
# Fusion
# ============================================================

class FusionLayer(nn.Module):
    def __init__(self, d_lstm: int, dropout: float, *, use_forecast: bool = False):
        super().__init__()
        self.use_forecast = use_forecast
        in_dim = d_lstm * (3 if use_forecast else 2)
        self.proj = nn.Linear(in_dim, d_lstm)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_dyn: torch.Tensor, s_tilde: torch.Tensor, f_tilde: torch.Tensor = None) -> torch.Tensor:
        if self.use_forecast:
            if f_tilde is None:
                raise ValueError("forecast features required but missing")
            h = torch.cat([z_dyn, s_tilde, f_tilde], dim=-1)
        else:
            h = torch.cat([z_dyn, s_tilde], dim=-1)
        h = F.relu(self.proj(h))
        return self.drop(h)


# ============================================================
# 2-hop GCN Routing Module
# - Uses edge_weight directly (GCNConv supports edge_weight)
# ============================================================

class GCNRouting2Hop(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_hops: int = 2):
        super().__init__()
        self.num_hops = num_hops

        # MODIFIED: multi-hop routing with residual + layernorm
        self.convs = nn.ModuleList()
        for hop in range(num_hops):
            in_channels = in_dim if hop == 0 else hidden_dim
            self.convs.append(
                GCNConv(in_channels, hidden_dim, add_self_loops=True, normalize=True)
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_hops)])
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        for hop, conv in enumerate(self.convs):
            delta = conv(x, edge_index, edge_weight)
            if hop != self.num_hops - 1:
                delta = F.relu(delta)
                delta = self.drop(delta)
            base = self.res_proj(x) if (hop == 0 and self.res_proj is not None) else x
            x = base + delta
            x = self.norms[hop](x)
        return x


# ============================================================
# Full Model: LSTM + Static + Fusion + 2-hop GCN
# ============================================================

class CombinedLSTMWithStatic2Hop(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        forecast_input_dim: int,
        lstm_hidden_dim: int,
        gnn_hidden_dim: int,
        output_dim: int,
        lstm_layers: int,
        gat_heads: int,     # unused in GCN, kept for minimal intrusion
        lstm_dropout: float,
        gnn_dropout: float,
        cheb_k: int = 3,    # unused, kept for API compatibility
        num_hops: int = 2,
    ):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(dynamic_input_dim, lstm_hidden_dim, lstm_layers, lstm_dropout)
        self.static_encoder = StaticEncoder(static_input_dim, lstm_hidden_dim)
        self.forecast_encoder = None
        if forecast_input_dim > 0:
            self.forecast_encoder = ForecastEncoder(
                forecast_input_dim=forecast_input_dim,
                out_dim=lstm_hidden_dim,
            )

        self.fusion = FusionLayer(lstm_hidden_dim, gnn_dropout, use_forecast=self.forecast_encoder is not None)

        self.gnn = GCNRouting2Hop(
            lstm_hidden_dim,
            gnn_hidden_dim,
            gnn_dropout,
            num_hops=num_hops,  # MODIFIED: multi-hop routing with residual + layernorm
        )
        self.out = nn.Linear(gnn_hidden_dim, output_dim)

    def forward(
        self,
        dynamic_features: torch.Tensor,  # (B, T, N, F_dyn)
        forecast_features: torch.Tensor, # (B, N, F_fcst)
        static_features: torch.Tensor,   # (B, N, F_static)
        edge_index: torch.Tensor,        # (2, E)
        edge_weight: torch.Tensor = None # (E,)
    ) -> torch.Tensor:
        B, T, N, F_dyn = dynamic_features.shape

        dyn = dynamic_features.permute(0, 2, 1, 3).reshape(B * N, T, F_dyn)
        sta = static_features.reshape(B * N, -1)
        fcst = None
        if self.forecast_encoder is not None:
            if forecast_features is None:
                raise ValueError("forecast_features is required when forecast_input_dim > 0")
            fcst = forecast_features.reshape(B * N, -1)

        z_dyn = self.lstm_encoder(dyn)         # (B*N, d_lstm)
        s_tilde = self.static_encoder(sta)     # (B*N, d_lstm)
        f_tilde = self.forecast_encoder(fcst) if self.forecast_encoder is not None else None
        node_embed = self.fusion(z_dyn, s_tilde, f_tilde)

        device = node_embed.device
        edge_index_big = []
        edge_weight_big = []

        for b in range(B):
            offset = b * N
            edge_index_big.append(edge_index + offset)
            if edge_weight is not None:
                edge_weight_big.append(edge_weight)

        edge_index_big = torch.cat(edge_index_big, dim=1).to(device)
        if edge_weight is not None:
            edge_weight_big = torch.cat(edge_weight_big, dim=0).to(device)
        else:
            edge_weight_big = None

        gnn_embed = self.gnn(node_embed, edge_index_big, edge_weight_big)
        pred = self.out(gnn_embed).view(B, N)
        return pred
