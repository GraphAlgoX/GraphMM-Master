import torch.nn as nn
import torch
import random
import math
from model.RoadGIN import RoadGIN
from model.TraceGCNV1 import TraceGCN
from model.seq2seqV1 import Seq2Seq
from model.graphfilter import GraphFilter
from model.crf import CRF
import torch.nn.functional as F


def mask_log_softmax(x, mask, log_flag=False):
    # [B, N]
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        # output_custom = torch.log((x_exp + 1e-10) / x_exp_sum)
        output_custom = torch.log((x_exp) / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return int(idx)


def log_sum_exp(vec):
    B, N = vec.shape
    max_score, _ = torch.max(vec, 1)
    # max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(B, -1).expand(B, N)
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


def min_max_norm(x):
    min_val = torch.min(x)
    max_val = torch.max(x)
    return (x - min_val) / (max_val - min_val)


def l2_norm(x):
    return x / torch.sqrt(torch.sum(x ** 2, axis=-1, keepdims=True))


class GMM(nn.Module):
    def __init__(self,
                 emb_dim,
                 device,
                 target_size,
                 beam_size,
                 atten_flag=True,
                 drop_prob=0.5,
                 bi=True) -> None:
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.target_size = target_size
        self.beam_size = beam_size
        self.atten_flag = atten_flag
        # self.road_gcn = RoadGCN(4 * loc_dim)
        self.road_gin = RoadGIN(emb_dim)
        self.trace_gcn = TraceGCN(emb_dim)
        self.seq2seq = Seq2Seq(input_size=2 * emb_dim,
                               hidden_size=emb_dim,
                               atten_flag=atten_flag,
                               drop_prob=drop_prob,
                               bi=bi)
        self.graphfilter = GraphFilter(emb_dim=emb_dim)
        self.road_feat_fc = nn.Linear(28, emb_dim) # 3*8 + 4
        self.trace_feat_fc = nn.Linear(4, emb_dim)
        self.fc_input = nn.Linear(2*self.emb_dim+3, 2*self.emb_dim)
        self.crf = CRF(num_tags=target_size,
                       emb_dim=emb_dim,
                       beam_size=beam_size,
                       device=device)
        # self.feat_encoder = FeatureEncoder(4, loc_dim)
        # self.exp_fc_road = nn.Linear(4, 4 * loc_dim)
        # self.exp_fc_traj = nn.Linear(2, 4)
        # self.norm_road = nn.BatchNorm1d(4)
        # self.norm_traj = nn.BatchNorm1d(2)
        # self.proj_out = nn.Sequential(
        #     nn.Linear(4 * loc_dim, 6 * loc_dim),
        #     nn.ReLU(),
        #     nn.Linear(6 * loc_dim, 4 * loc_dim)
        # )
        # self.proj_road = nn.Sequential(
        #     nn.Linear(4 * loc_dim, 6 * loc_dim),
        #     nn.ReLU(),
        #     nn.Linear(6 * loc_dim, 4 * loc_dim)
        # )
        # self.road_mlp = nn.Sequential(
        #     nn.Linear(4 * loc_dim, 8 * loc_dim),
        #     nn.BatchNorm1d(8 * loc_dim),
        #     nn.ReLU(),
        #     nn.Linear(8 * loc_dim, 8 * loc_dim),
        #     nn.BatchNorm1d(8 * loc_dim),
        #     nn.ReLU(),
        #     nn.Linear(8 * loc_dim, 4 * loc_dim)
        # )
        # self.classification = nn.Linear(4 * loc_dim, target_size)

    def forward(self, grid_traces, tgt_roads, traces_gps, traces_lens,
                road_lens, gdata, sample_Idx, tf_ratio):
        """
        grid_traces: (batch_size, seq_len, emb_size)
        tgt_roads: (batch_size, seq_len1, emb_size)
        traces_lens, road_lens: (batch_size, )
        """
        full_road_emb, full_grid_emb = self.get_emb(gdata)
        # B, RL = tgt_roads.shape
        # B, TL = grid_traces.shape
        emissions = self.get_probs(grid_traces=grid_traces,
                                   tgt_roads=tgt_roads,
                                   traces_gps=traces_gps,
                                   trace_lens=traces_lens,
                                   road_lens=road_lens,
                                   tf_ratio=tf_ratio,
                                   full_grid_emb=full_grid_emb,
                                   gdata=gdata,
                                   sample_Idx=sample_Idx,
                                   easy_filter_cache=gdata.A_list.squeeze(0),
                                   full_road_emb=full_road_emb)
        # B, L, N
        # full_loss = 0

        # # build mask
        # mask = torch.arange(0, RL).long().unsqueeze(0).expand(B, RL)
        # mask = (mask < road_lens.unsqueeze(1).expand(B, RL)).to(self.device)
        # gold_score = self._score_sentence_batch(emissions, tgt_roads, mask,
        #                                         full_road_emb, gdata.A_list)
        # for idx, feat in enumerate(emissions):
        #     feat = feat[:road_lens[idx], :]  # mask
        #     forward_score = self.negative_sample_sum_single(
        #         feat, full_road_emb, gdata.A_list)
        #     full_loss += forward_score
        # full_loss -= sum(gold_score)
        # avg_loss = full_loss / B
        # return avg_loss
        tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
        for i in range(len(road_lens)):
            tgt_mask[i][:road_lens[i]] = 1.
        tgt_mask = tgt_mask.bool().to(self.device)
        loss = -self.crf(emissions, tgt_roads, full_road_emb.detach(), gdata.A_list.squeeze(0), tgt_mask)
        return loss

    def infer(self, grid_traces, traces_gps, traces_lens, road_lens, sample_Idx, gdata,
              tf_ratio):
        """
        grid_traces: (batch_size, seq_len, emb_size)
        traces_lens, road_lens: (batch_size, )
        """
        full_road_emb, full_grid_emb = self.get_emb(gdata)

        emissions = self.get_probs(grid_traces=grid_traces,
                                   tgt_roads=None,
                                   traces_gps=traces_gps,
                                   trace_lens=traces_lens,
                                   road_lens=road_lens,
                                   tf_ratio=tf_ratio,
                                   full_grid_emb=full_grid_emb,
                                   sample_Idx=sample_Idx,
                                   gdata=gdata,
                                   easy_filter_cache=gdata.A_list.squeeze(0),
                                   full_road_emb=full_road_emb)
        # max_RL = max(road_lens)
        # B = len(road_lens)
        # infer_output = torch.zeros(B, max_RL).to(self.device)
        # for idx, feat in enumerate(emissions):
        #     feat = feat[:road_lens[idx], :]
        #     score, tag_seq = self._viterbi_decode_single_line(
        #         feat, full_road_emb, gdata.A_list)
        #     infer_output[idx, :road_lens[idx]] = tag_seq
        # return score, infer_output
        tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
        for i in range(len(road_lens)):
            tgt_mask[i][:road_lens[i]] = 1.
        tgt_mask = tgt_mask.bool().to(self.device)
        preds = self.crf.decode(emissions, full_road_emb, gdata.A_list.squeeze(0), tgt_mask)
        return None, preds

    def get_emb(self, gdata):
        """
        gain road embedding and grid embedding
        """
        # road_x = self.feat_encoder(gdata.road_x)
        road_x = self.road_feat_fc(gdata.road_x)
        # road_x = self.feat_fc(F.normalize(gdata.road_x))
        # road_x = self.exp_fc_road(self.norm_road(gdata.road_x))
        # full_road_emb = self.road_gcn(road_x, gdata.road_adj)
        full_road_emb = self.road_gin(road_x, gdata.road_adj)
        # full_road_emb1 = self.road_mlp(full_road_emb.detach())
        # [num_of_grid, 4*loc]
        pure_grid_feat = torch.mm(gdata.map_matrix, full_road_emb)
        # pure_grid_feat[gdata.singleton_grid_mask] = self.exp_fc_road(singleton_x)
        # pure_grid_feat[gdata.singleton_grid_mask] = self.feat_encoder(gdata.singleton_grid_location)
        pure_grid_feat[gdata.singleton_grid_mask] = self.trace_feat_fc(gdata.singleton_grid_location)
        full_grid_emb = torch.zeros(gdata.num_grids + 1, 2 * self.emb_dim).to(self.device)
        full_grid_emb[1:, :] = self.trace_gcn(pure_grid_feat,
                                              gdata.trace_inadj,
                                              gdata.trace_outadj,
                                              gdata.trace_weight)
        return full_road_emb, full_grid_emb

    def rnnhidden2prob(self,
                       lst_road_id,
                       rnn_out,
                       gdata,
                       full_road_emb,
                       easy_filter_cache,
                       first_constraint=False,
                       tmp_grid=None):
        """
        hidden similarity computation
        """
        
        # batchsize = lst_road_id.shape[0]
        # constraint = torch.zeros(batchsize, gdata.num_roads).to(self.device)
        # if first_constraint:
        #     tmp = torch.zeros(batchsize, gdata.num_grids).to(self.device)
        #     for idx in range(batchsize):
        #         gridx, gridy = gdata.traceid2grid_dict[int(tmp_grid[idx]) - 1]
        #         for i in range(gridx - 3, gridx + 4):
        #             for j in range(gridy - 3, gridy + 4):
        #                 if gdata.grid2traceid_dict.get((i, j)) is not None:
        #                     tmp[idx, gdata.grid2traceid_dict[(i, j)]] = \
        #                         1 / (abs(i - gridx) + abs(j - gridy) + 1)
        #     constraint = tmp @ gdata.map_matrix
        # else:
        #     constraint = easy_filter_cache[lst_road_id.squeeze(1)]  # [B, N]
        #     mask = (lst_road_id.squeeze(1) == -1)
        #     constraint[mask] = 1
        # h_iH_R \odot f(A_R)
        
        prob = (rnn_out @ full_road_emb.detach().T).squeeze(0)
        # prob = mask_log_softmax(prob, constraint, log_flag=False)
        # prob = self.classification(rnn_out)
        # prob = (l2_norm(rnn_out) @ l2_norm(full_road_emb.detach()).T).squeeze(0)

        return prob

    def get_probs(self, grid_traces, tgt_roads, traces_gps, sample_Idx, trace_lens,
                  road_lens, tf_ratio, full_road_emb, full_grid_emb,
                  easy_filter_cache, gdata):
        """
        decode max_trace_lens times for tgt
        grid_traces: (batch_size, max_trace_lens)
        tgt_roads: (batch_size, max_road_lens)
        return (batch_size, max_road_lens, num_roads)
        """
        if tgt_roads is not None:
            B, max_RL = tgt_roads.shape
        else:
            B = grid_traces.shape[0]
            max_RL = int(max(road_lens))

        rnn_input = full_grid_emb[grid_traces]
        # trace_grids = gps2grid_batch(trace_grids)
        # traces_gps = self.norm_traj(traces_gps.permute(0, 2, 1)).permute(0, 2, 1)
        # traces_gps = self.exp_fc_traj(traces_gps)
        rnn_input = torch.cat([rnn_input, traces_gps, sample_Idx.unsqueeze(-1)], dim=-1)
        rnn_input = self.fc_input(rnn_input)
        encoder_outputs, hiddens = self.seq2seq.encode(rnn_input, trace_lens)
        # start decode
        probs = torch.zeros(B, max_RL, gdata.num_roads).to(self.device)
        inputs = torch.zeros(B, 1, self.seq2seq.hidden_size).to(self.device)
        attn_mask = None
        if self.atten_flag:
            attn_mask = torch.zeros(B, int(max(trace_lens)))
            for i in range(len(trace_lens)):
                attn_mask[i][:trace_lens[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs,
                                              attn_mask)
        probs[:, 0, :] = inputs.squeeze(1) @ full_road_emb.detach().T
        teacher_force = random.random() < tf_ratio
        if teacher_force:
            lst_road_id = tgt_roads[:, 0]
        else:
            lst_road_id = probs[:, 0, :].argmax(1)
        # penality entry
        # tgt_mask = torch.zeros(B, int(max(road_lens)))
        # for i in range(len(road_lens)):
        #     tgt_mask[i][:road_lens[i]] = 1.
        # tgt_mask = tgt_mask.to(self.device)
        # penality_loss, count = 0., 1
        for t in range(1, max_RL):
            if teacher_force:
                inputs = full_road_emb[lst_road_id].view(B, 1, -1)
            inputs, hiddens = self.seq2seq.decode(inputs, hiddens,
                                                  encoder_outputs, attn_mask)
            probs[:, t, :] = inputs.squeeze(1) @ full_road_emb.detach().T
            teacher_force = random.random() < tf_ratio
            # if self.training:
            #     cur_p = torch.norm(inputs.squeeze(1) - full_road_emb.detach()[tgt_roads[:, t]], p=2, dim=1) * tgt_mask[:, t]
            #     penality_loss += cur_p.sum()
            #     count += torch.sum(tgt_mask[:, t] != 0)
            if teacher_force:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs

    ##################functions for CRF##################
    def transition(self, tag1, tag2, full_road_emb, A_list):
        """
        tag1, tag2: (batch_size,)
        """
        # (batch_size, road_emb_size)
        hidden1 = full_road_emb[tag1]
        hidden2 = full_road_emb[tag2]
        return self.graphfilter.binary_loss_batch(tag1, tag2, hidden1, hidden2,
                                                  A_list)

    def transitions(self, next_tag, full_road_emb, A_list):
        transitions = self.graphfilter.get_hard_filter(full_road_emb, next_tag,
                                                       A_list)
        return transitions

    def negative_sample_sum_single(self, emissions, full_road_emb, A_list):
        """
        reduce the transition states
        """
        # Do the forward algorithm to compute the partition function
        forward_var = torch.full((1, self.target_size), 0.).to(self.device)
        # Wrap in a variable so that we will get automatic backprop
        forward_var[0] = emissions[0]
        # Iterate through the sentence
        for feat in emissions[1:, :]:
            alphas_t = (torch.ones(1, self.target_size) * float('-inf')).to(
                self.device)  # The forward tensors at this timestep

            _, index = forward_var.topk(self.beam_size, largest=True)
            next_tag_set = A_list[:, index.squeeze(0), :].sum(dim=0).sum(dim=0).nonzero().squeeze(1)

            for next_tag in next_tag_set:
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions(next_tag, full_road_emb, A_list)

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t[:, next_tag] = log_sum_exp(next_tag_var).view(1)
                # alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = alphas_t

        terminal_var = forward_var

        value, _ = terminal_var.topk(self.beam_size, largest=True)
        # * n/beamsize
        alpha = log_sum_exp(value) + math.log(
            self.target_size / self.beam_size)
        del forward_var, alphas_t
        return alpha

    def _score_sentence_batch(self, emissions, tags, mask, full_road_emb,
                              A_list):
        """
        S(X,y)
        emissions: (batch_size, seq_len, num_tags)
        tags: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        return (batch_size, )
        """
        batch_size, seq_len, _ = emissions.shape
        # (batch_size, )
        score = torch.zeros(batch_size).to(self.device)

        for i in range(0, seq_len - 1):
            # (batch_size, )
            score += self.transition(tags[:, i], tags[:, i + 1], full_road_emb,
                                     A_list)

        areg = torch.arange(seq_len).to(self.device)
        for i in range(batch_size):
            score[i] += sum(emissions[i, areg, tags[i, :]] * mask[i, :])

        return score

    def _viterbi_decode_single_line(self, emissions, full_road_emb, A_list):
        L, N = emissions.shape

        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, N), 0).to(self.device)
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        forward_var[0] = emissions[0]

        for feat in emissions[1:, :]:
            _, index = forward_var.topk(self.beam_size, largest=True)
            next_tag_set = A_list[:, index.squeeze(0), :].sum(dim=0).sum(
                dim=0).nonzero().squeeze(1)

            bptrs_t = torch.zeros(self.target_size,
                                  dtype=torch.long).to(self.device)

            # bptrs_t = []  # holds the backpointers for this step
            # viterbivars_t = []  # holds the viterbi variables for this step
            viterbivars_t = (torch.ones(1, self.target_size) *
                             float('-inf')).to(self.device)
            for next_tag in next_tag_set:
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions(
                    next_tag, full_road_emb, A_list)
                best_tag_id = argmax(next_tag_var)
                bptrs_t[next_tag] = best_tag_id
                viterbivars_t[0, next_tag] = next_tag_var[0][best_tag_id]
                # viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat.view(1, -1)
            # choose the beam-size th max id
            # compute the available candidates
            backpointers.append(bptrs_t)
        # dp
        # Transition to STOP_TAG
        terminal_var = forward_var
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        # assert start == start_tag  # Sanity check
        best_path.reverse()
        # print(best_path, path_score)
        best_path = torch.tensor(best_path)
        return path_score, best_path


if __name__ == "__main__":
    pass
