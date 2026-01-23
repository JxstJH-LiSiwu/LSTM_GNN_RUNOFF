import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
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


class StaticEncoder(nn.Module):
    def __init__(self, static_input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(static_input_dim, out_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(s))


class FusionLayer(nn.Module):
    def __init__(self, d_lstm: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(d_lstm * 2, d_lstm)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_dyn: torch.Tensor, s_tilde: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_dyn, s_tilde], dim=-1)
        h = F.relu(self.proj(h))
        return self.drop(h)


# ============================================================
# 2-layer ChebNet Routing (Chebyshev spectral conv)
# - edge_weight is supported: treated as edge weights in Laplacian construction
# ============================================================

class ChebNetRouting2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, K: int, dropout: float, num_hops: int = 2):
        super().__init__()
        self.num_hops = num_hops

        # MODIFIED: multi-hop routing with residual + layernorm
        self.convs = nn.ModuleList()
        for hop in range(num_hops):
            in_channels = in_dim if hop == 0 else hidden_dim
            self.convs.append(ChebConv(in_channels, hidden_dim, K=K))

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


class CombinedLSTMWithStatic2Hop(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        lstm_hidden_dim: int,
        gnn_hidden_dim: int,
        output_dim: int,
        lstm_layers: int,
        gat_heads: int,     # unused, kept for minimal intrusion
        lstm_dropout: float,
        gnn_dropout: float,
        cheb_k: int ,    # <-- ChebNet needs K; default won't break old train.py
        num_hops: int = 2,
    ):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(dynamic_input_dim, lstm_hidden_dim, lstm_layers, lstm_dropout)
        self.static_encoder = StaticEncoder(static_input_dim, lstm_hidden_dim)
        self.fusion = FusionLayer(lstm_hidden_dim, gnn_dropout)

        self.gnn = ChebNetRouting2Layer(
            lstm_hidden_dim,
            gnn_hidden_dim,
            K=cheb_k,
            dropout=gnn_dropout,
            num_hops=num_hops,  # MODIFIED: multi-hop routing with residual + layernorm
        )
        self.out = nn.Linear(gnn_hidden_dim, output_dim)

    def forward(
        self,
        dynamic_features: torch.Tensor,
        static_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, N, F_dyn = dynamic_features.shape

        dyn = dynamic_features.permute(0, 2, 1, 3).reshape(B * N, T, F_dyn)
        sta = static_features.reshape(B * N, -1)

        z_dyn = self.lstm_encoder(dyn)
        s_tilde = self.static_encoder(sta)
        node_embed = self.fusion(z_dyn, s_tilde)

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
