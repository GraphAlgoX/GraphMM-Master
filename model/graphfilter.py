import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphFilter(nn.Module):
    """
    Graph Filter Layer f(A_R)
    """
    def __init__(self, emb_dim):
        super(GraphFilter, self).__init__()
        # self.layers = layers
        self.emb_dim = emb_dim
        # used for compute loss
        self.w = nn.Parameter(torch.randn(1, emb_dim * 2))

    def binary_loss(self, id1, id2, emb1, emb2, A_list):  # eq 15, 16
        concat_emb = torch.cat([emb1, emb2], 0).view(-1, 1)
        r = F.relu(self.w @ concat_emb)
        ans = (A_list[:, id1, id2].T @ r)
        return ans

    def binary_loss_batch(self, id1, id2, emb1, emb2, A_list):  # eq 15, 16
        """
        id1, id2:(batch_size, )
        emb1, emb2: (batch_size, emb_size)
        """
        # (2 * emb_size, batch_size)
        concat_emb = torch.cat([emb1, emb2], dim=1).T
        # (L, batch_size)
        r = F.relu(torch.mm(self.w, concat_emb))
        ans = torch.sum(A_list[:, id1, id2] * r, dim=0)
        return ans

    def get_hard_filter(self, emb_ls, id, A_list):
        N, E = emb_ls.shape
        D = 1
        ans = torch.zeros(1, N).to(emb_ls.device)
        idx = A_list[:, id, :].sum(dim=0).nonzero().squeeze(1)

        emb_copy = torch.cat([emb_ls[[id], :]] * len(idx), dim=0)

        emb_concat = torch.cat([emb_ls[idx], emb_copy],
                               dim=1).reshape(-1, 2 * E)  # [,2E]
        # [D,N] D=depth (5)
        r = torch.mm(self.w.reshape(D, 2 * E), emb_concat.T).reshape(D, -1)
        r = F.relu(r)

        ans[:, idx] = torch.sum(A_list[:, id, idx] * r, dim=0).reshape(1, -1)

        del emb_copy, emb_concat
        return ans


if __name__ == "__main__":
    pass
