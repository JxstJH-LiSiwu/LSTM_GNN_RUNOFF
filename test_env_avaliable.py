from torch_geometric.nn import GATConv
import torch

x = torch.randn(10, 16).cuda()
edge_index = torch.tensor([[0,1,2],[1,2,3]]).cuda()

gat = GATConv(16, 8).cuda()
out = gat(x, edge_index)

print(out.shape)

