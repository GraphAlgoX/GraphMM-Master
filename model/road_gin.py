import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, MLP
import torch.nn.functional as F


class RoadGIN(nn.Module):
    def __init__(self, emb_dim, depth=3, mlp_layers=2):
        super().__init__()
        self.depth = depth
        self.gins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.depth):
            mlp = MLP(in_channels=emb_dim,
                      hidden_channels=2 * emb_dim,
                      out_channels=emb_dim,
                      num_layers=mlp_layers)
            self.gins.append(GINConv(nn=mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, adj_t):
        layer_outputs = []
        for i in range(self.depth):
            x = self.gins[i](x, adj_t.to(x.device))
            x = F.relu(self.batch_norms[i](x))
            layer_outputs.append(x)
        x = torch.stack(layer_outputs, dim=0)
        x = torch.max(x, dim=0)[0]
        return x
