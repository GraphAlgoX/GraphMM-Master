import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.gcnconv = GCNConv(in_channels=in_feats,
                               out_channels=out_feats,
                               add_self_loops=False,
                               bias=bias)

    def forward(self, x, adj_t, edge_weight):
        hl = self.linear(x)
        hr = self.gcnconv(x, adj_t, edge_weight)
        return hl + hr


class DiGCN(nn.Module):
    def __init__(self, embed_dim, depth=2):
        super(DiGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            self.convs.append(GCNLayer(embed_dim, embed_dim))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, adj_t, edge_weight):
        for idx in range(self.depth):
            x = self.convs[idx](x, adj_t, edge_weight)
            x = self.bns[idx](x)
            if idx != self.depth - 1:
                x = F.relu(x)
        return x


class TraceGCN(torch.nn.Module):
    def __init__(self, emb_dim):
        super(TraceGCN, self).__init__()
        self.emb_dim = emb_dim
        self.gcn1 = DiGCN(self.emb_dim)
        self.gcn2 = DiGCN(self.emb_dim)

    def forward(self, feats, in_adj_t, out_adj_t, edge_weight):
        emb_ind = self.gcn1(feats, in_adj_t, edge_weight)
        emb_oud = self.gcn2(feats, out_adj_t, edge_weight)
        ans = torch.cat([emb_ind, emb_oud], 1)
        return ans


if __name__ == "__main__":
    pass