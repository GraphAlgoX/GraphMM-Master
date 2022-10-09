import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):

    def __init__(self, in_dim, loc_dim):
        super(FeatureEncoder, self).__init__()
        
        self.embedding_list = nn.ModuleList()

        for i in range(in_dim):
            emb = nn.Linear(1, loc_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

    def forward(self, x):
        x_emb = []
        for i in range(x.shape[1]):
            x_emb.append(self.embedding_list[i](x[:, i].view(-1, 1)))

        return torch.cat(x_emb, dim=-1)


if __name__ == "__main__":
    pass
