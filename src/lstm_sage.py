import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


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
# 2-hop GraphSAGE Routing
# - GraphSAGEConv does NOT take edge_weight in standard form; we ignore it (kept in signature)
# ============================================================

class GraphSAGERouting2Hop(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.drop(x)
        x = self.sage2(x, edge_index)
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
        cheb_k: int = 3,    # unused, kept for API compatibility
    ):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(dynamic_input_dim, lstm_hidden_dim, lstm_layers, lstm_dropout)
        self.static_encoder = StaticEncoder(static_input_dim, lstm_hidden_dim)
        self.fusion = FusionLayer(lstm_hidden_dim, gnn_dropout)

        self.gnn = GraphSAGERouting2Hop(lstm_hidden_dim, gnn_hidden_dim, gnn_dropout)
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
        # edge_weight is ignored by GraphSAGE, but keep big graph construction identical
        for b in range(B):
            offset = b * N
            edge_index_big.append(edge_index + offset)

        edge_index_big = torch.cat(edge_index_big, dim=1).to(device)

        gnn_embed = self.gnn(node_embed, edge_index_big, edge_weight=None)
        pred = self.out(gnn_embed).view(B, N)
        return pred
