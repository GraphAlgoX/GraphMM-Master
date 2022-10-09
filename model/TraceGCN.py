
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


class DiGCN(torch.nn.Module):
    def __init__(self, embed_dim, depth=2):
        super(DiGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            self.convs.append(
                GCNConv(embed_dim, embed_dim, normalize=False, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, adj_t):
        for idx in range(self.depth):
            x = self.convs[idx](x, adj_t)
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

    def forward(self, feats, in_adj_t, out_adj_t):
        emb_ind = self.gcn1(feats, in_adj_t)
        emb_oud = self.gcn2(feats, out_adj_t)
        ans = torch.cat([emb_ind, emb_oud], 1)
        return ans

    # def forward_minibatch(self, feats, inweight,outweight,ins,outs,tgtid):
    #     tgtid = tgtid.cpu().numpy().tolist()
    #     tgtnum = len(tgtid)
    #     inbatch_size, inn_id, inadjs = ins
    #     outbatch_size, outn_id, outadjs = outs
    #     x_in = feats[inn_id]
    #     x_out = feats[outn_id]
    #     inid2idx = {}
    #     outid2idx = {}
    #     for idx, id in enumerate(inn_id):
    #         inid2idx[int(id)] = idx
    #     for idx, id in enumerate(outn_id):
    #         outid2idx[int(id)] = idx

    #     emb_ind = self.gcn1(x_in, inadjs, inweight)
    #     emb_oud = self.gcn2(x_out, outadjs, outweight)
    #     out = {}

    #     for i in tgtid:
    #         out[i] = torch.cat(
    #             [emb_ind[inid2idx[i]], emb_oud[outid2idx[i]]], 0)
    #     return out  # batch*2*embeddim
