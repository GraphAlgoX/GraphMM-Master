import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class UnDiGCN(torch.nn.Module):
    def __init__(self, embed_dim, depth=2):
        super(UnDiGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            self.convs.append(
                GCNConv(in_channels=embed_dim, out_channels=embed_dim))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, adj_t):
        for idx in range(self.depth):
            x = self.convs[idx](x, adj_t.to(x.device))
            x = self.bns[idx](x)
            if idx != self.depth - 1:
                x = F.relu(x)
        return x


class RoadGCN(torch.nn.Module):
    def __init__(self, emb_dim):
        super(RoadGCN, self).__init__()
        self.emb_dim = emb_dim
        self.gcn = UnDiGCN(self.emb_dim)

    def forward(self, x_feat, adj_t):
        """
        adj_t: SparseTensor
        """
        emb = self.gcn(x_feat, adj_t)  # n, embdim
        return emb

    # def forward_minibatch(self, x_feat, batchs, tgtid):
    #     tgtid = tgtid.numpy().tolist()
    #     tgtnum = len(tgtid)
    #     batch_size, n_id, adjs = batchs
    #     x_feat = x_feat[n_id]
    #     nid2idx = {}
    #     for idx, id in enumerate(n_id):
    #         nid2idx[int(id)] = idx
    #     x = self.location_encoder(x_feat)
    #     emb = self.gcn(x, adjs)
    #     out = {}
    #     for i in tgtid:
    #         out[i] = emb[nid2idx[i]]
    #     return out
