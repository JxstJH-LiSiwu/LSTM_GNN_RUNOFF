# src/lstm_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ============================================================
# LSTM Encoder (Dynamic only) - Paper-aligned
#   Input : (B*N, T, F_dyn)
#   Output: (B*N, d_lstm)
#   We take the last timestep hidden state as z_i.
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
        """
        x_dyn: (B*N, T, F_dyn)
        returns z: (B*N, hidden_dim)
        """
        out, _ = self.lstm(x_dyn)
        z = out[:, -1, :]  # last timestep
        return z


# ============================================================
# Static Encoder - Paper-aligned
#   s_i (59-dim) -> Linear -> ReLU -> s_tilde_i (d_lstm)
# ============================================================

class StaticEncoder(nn.Module):
    def __init__(self, static_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(static_input_dim, out_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B*N, static_dim)
        returns: (B*N, out_dim)
        """
        return F.relu(self.fc(s))


# ============================================================
# Forecast Encoder - tomorrow's meteorological features
#   f_i (F_fcst-dim) -> Linear -> ReLU -> f_tilde_i (d_lstm)
# ============================================================

class ForecastEncoder(nn.Module):
    def __init__(self, forecast_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(forecast_input_dim, out_dim)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B*N, forecast_dim)
        returns: (B*N, out_dim)
        """
        return F.relu(self.fc(f))


# ============================================================
# Fusion (Paper-aligned)
#   h_i = Dropout(ReLU(W_c [z_i || s_tilde_i || f_tilde_i] + b))
#   where z_i: d_lstm, s_tilde_i: d_lstm, f_tilde_i: d_lstm (optional)
# ============================================================

class FusionLayer(nn.Module):
    def __init__(self, d_lstm: int, dropout: float, *, use_forecast: bool = False):
        super().__init__()
        self.use_forecast = use_forecast
        in_dim = d_lstm * (3 if use_forecast else 2)
        self.proj = nn.Linear(in_dim, d_lstm)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_dyn: torch.Tensor, s_tilde: torch.Tensor, f_tilde: torch.Tensor = None) -> torch.Tensor:
        """
        z_dyn  : (B*N, d_lstm)
        s_tilde: (B*N, d_lstm)
        f_tilde: (B*N, d_lstm) or None
        returns h: (B*N, d_lstm)
        """
        if self.use_forecast:
            if f_tilde is None:
                raise ValueError("forecast features required but missing")
            h = torch.cat([z_dyn, s_tilde, f_tilde], dim=-1)  # (B*N, 3*d_lstm)
        else:
            h = torch.cat([z_dyn, s_tilde], dim=-1)  # (B*N, 2*d_lstm)
        h = F.relu(self.proj(h))                # (B*N, d_lstm)
        h = self.drop(h)
        return h


# ============================================================
# 2-hop GAT Routing Module
#   Input : node embeddings (B*N, d_lstm)
#   Output: routed embeddings (B*N, gnn_hidden_dim)
#
# NOTE:
# - edge_weight is passed as edge_attr with shape [E, 1]
# ============================================================

class GATRouting2Hop(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        num_hops: int = 2,
    ):
        super().__init__()
        self.num_hops = num_hops

        # MODIFIED: multi-hop routing with residual + layernorm
        self.convs = nn.ModuleList()
        for hop in range(num_hops):
            in_channels = in_dim if hop == 0 else hidden_dim
            self.convs.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,  # keep (N, hidden_dim)
                    dropout=dropout,
                    edge_dim=1,
                )
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_hops)])
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B*N, in_dim)
        edge_index: (2, E_big)
        edge_weight: (E_big,) or (E_big, 1)
        """
        if edge_weight is not None and edge_weight.dim() == 1:
            edge_weight = edge_weight.view(-1, 1)  # (E,) -> (E,1)
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
# Full Model: LSTM (dynamic) + Static Encoder + Fusion + 2-hop GAT
#   - Output is log-space discharge (to match your Dataset target = log1p(Q))
# ============================================================

class CombinedLSTMWithStatic2Hop(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        forecast_input_dim: int,
        lstm_hidden_dim: int,   # d_lstm, paper uses 128
        gnn_hidden_dim: int,    # paper uses 64
        output_dim: int,        # 1
        lstm_layers: int,
        gat_heads: int,
        lstm_dropout: float,
        gnn_dropout: float,
        cheb_k: int,
        num_hops: int = 2,
    ):
        super().__init__()

        # ----- LSTM encoder (dynamic only) -----
        self.lstm_encoder = LSTMEncoder(
            input_dim=dynamic_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # ----- Static encoder (paper-aligned) -----
        self.static_encoder = StaticEncoder(
            static_input_dim=static_input_dim,
            out_dim=lstm_hidden_dim,  # s_tilde in R^{d_lstm}
        )

        # ----- Forecast encoder (tomorrow's meteo) -----
        self.forecast_encoder = None
        if forecast_input_dim > 0:
            self.forecast_encoder = ForecastEncoder(
                forecast_input_dim=forecast_input_dim,
                out_dim=lstm_hidden_dim,
            )

        # ----- Fusion (paper-aligned) -----
        self.fusion = FusionLayer(
            d_lstm=lstm_hidden_dim,
            dropout=gnn_dropout,  # paper uses dropout after fusion
            use_forecast=self.forecast_encoder is not None,
        )

        # ----- GNN routing -----
        self.gnn = GATRouting2Hop(
            in_dim=lstm_hidden_dim,      # input is h_i in R^{d_lstm}
            hidden_dim=gnn_hidden_dim,
            heads=gat_heads,
            dropout=gnn_dropout,
            num_hops=num_hops,  # MODIFIED: multi-hop routing with residual + layernorm
        )

        # ----- Output head -----
        self.out = nn.Linear(gnn_hidden_dim, output_dim)

    def forward(
        self,
        dynamic_features: torch.Tensor,  # (B, T, N, F_dyn)
        forecast_features: torch.Tensor, # (B, N, F_fcst)
        static_features: torch.Tensor,   # (B, N, F_static)
        edge_index: torch.Tensor,        # (2, E)
        edge_weight: torch.Tensor = None # (E,)
    ) -> torch.Tensor:
        """
        returns pred: (B, N) in log-space (compatible with your Dataset target log1p(Q))
        """
        B, T, N, F_dyn = dynamic_features.shape

        # --------------------------------------------------------
        # reshape to (B*N, T, F_dyn) and (B*N, F_static)
        # --------------------------------------------------------
        dyn = dynamic_features.permute(0, 2, 1, 3).reshape(B * N, T, F_dyn)
        sta = static_features.reshape(B * N, -1)
        fcst = None
        if self.forecast_encoder is not None:
            if forecast_features is None:
                raise ValueError("forecast_features is required when forecast_input_dim > 0")
            fcst = forecast_features.reshape(B * N, -1)

        # --------------------------------------------------------
        # encoders
        # --------------------------------------------------------
        z_dyn = self.lstm_encoder(dyn)         # (B*N, d_lstm)
        s_tilde = self.static_encoder(sta)     # (B*N, d_lstm)
        f_tilde = self.forecast_encoder(fcst) if self.forecast_encoder is not None else None

        # --------------------------------------------------------
        # fusion (paper)
        # --------------------------------------------------------
        node_embed = self.fusion(z_dyn, s_tilde, f_tilde)  # (B*N, d_lstm)

        # --------------------------------------------------------
        # Build "big graph" for batched message passing
        # edge_index: (2, E)  -> (2, B*E) with node offsets
        # edge_weight: (E,)   -> (B*E,)
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # GNN routing
        # --------------------------------------------------------
        gnn_embed = self.gnn(node_embed, edge_index_big, edge_weight_big)  # (B*N, gnn_hidden)

        # --------------------------------------------------------
        # output
        # --------------------------------------------------------
        pred = self.out(gnn_embed).view(B, N)  # (B, N)
        return pred
