import torch.nn as nn
import torch
import random
import math
from model.RoadGCN import RoadGCN
from model.TraceGCN import TraceGCN
from model.seq2seq import Seq2Seq
from model.graphfilter import GraphFilter
import torch.nn.functional as F


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


class GMM(nn.Module):
    def __init__(self,
                 loc_dim,
                 device,
                 target_size,
                 beam_size,
                 atten_flag=True) -> None:
        super().__init__()
        self.device = device
        self.loc_dim = loc_dim
        self.target_size = target_size
        self.beam_size = beam_size
        self.atten_flag = atten_flag
        self.road_gcn = RoadGCN(loc_dim)
        self.trace_gcn = TraceGCN(loc_dim)
        self.seq2seq = Seq2Seq(input_size=8 * loc_dim,
                               hidden_size=4 * loc_dim,
                               atten_flag=atten_flag)
        self.graphfilter = GraphFilter(emb_dim=4 * loc_dim)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, grid_traces, tgt_roads, traces_gps, traces_lens,
                road_lens, gdata, tf_ratio):
        """
        grid_traces: (batch_size, seq_len, emb_size)
        tgt_roads: (batch_size, seq_len1, emb_size)
        traces_lens, road_lens: (batch_size, )
        """
        full_road_emb, full_grid_emb = self.get_emb(gdata)
        easy_filter_cache = self.graphfilter.init_easy_filter_cache(gdata.A_list)
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
                                   easy_filter_cache=easy_filter_cache,
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
        return emissions

    def infer(self, grid_traces, traces_gps, traces_lens, road_lens, gdata,
              tf_ratio):
        """
        grid_traces: (batch_size, seq_len, emb_size)
        traces_lens, road_lens: (batch_size, )
        """
        full_road_emb, full_grid_emb = self.get_emb(gdata)
        easy_filter_cache = self.graphfilter.init_easy_filter_cache(gdata.A_list)

        emissions = self.get_probs(grid_traces=grid_traces,
                                   tgt_roads=None,
                                   traces_gps=traces_gps,
                                   trace_lens=traces_lens,
                                   road_lens=road_lens,
                                   tf_ratio=tf_ratio,
                                   full_grid_emb=full_grid_emb,
                                   gdata=gdata,
                                   easy_filter_cache=easy_filter_cache,
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
        return None, F.softmax(emissions, dim=-1)

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

    def get_emb(self, gdata):
        full_road_emb = self.road_gcn(self.norm(gdata.road_x), gdata.road_adj)
        # [num_of_grid, 4*loc]
        pure_grid_feat = torch.mm(gdata.map_matrix, full_road_emb)
        pure_grid_feat[gdata.singleton_grid_mask] = self.norm(gdata.singleton_grid_location)
        full_grid_emb = torch.zeros(gdata.num_grids + 1, 8 * self.loc_dim).to(self.device)
        full_grid_emb[1:, :] = self.trace_gcn(pure_grid_feat,
                                              gdata.trace_inadj,
                                              gdata.trace_outadj)
        return full_road_emb, full_grid_emb

    def rnnhidden2prob(self,
                       lst_road_id,
                       rnn_out,
                       gdata,
                       full_road_emb,
                       easy_filter_cache,
                       first_constraint=False,
                       tmp_grid=None):
        batchsize = lst_road_id.shape[0]
        constraint = torch.zeros(batchsize, gdata.num_roads).to(self.device)
        if first_constraint:
            tmp = torch.zeros(batchsize, gdata.num_grids).to(self.device)
            for idx in range(batchsize):
                gridx, gridy = gdata.traceid2grid_dict[int(tmp_grid[idx]) - 1]
                for i in range(gridx - 3, gridx + 4):
                    for j in range(gridy - 3, gridy + 4):
                        if gdata.grid2traceid_dict.get((i, j)) is not None:
                            tmp[idx, gdata.grid2traceid_dict[(i, j)]] = \
                                1 / (abs(i - gridx) + abs(j - gridy) + 1)
            constraint = tmp @ gdata.map_matrix
        else:
            constraint = easy_filter_cache[lst_road_id.squeeze(1)]  # [B, N]
        # h_iH_R \odot f(A_R)
        prob = (rnn_out @ full_road_emb.detach().T).squeeze(0) * constraint
        # prob = mask_log_softmax(prob, constraint, log_flag=False)

        return prob

    def get_probs(self, grid_traces, tgt_roads, traces_gps, trace_lens,
                  road_lens, tf_ratio, full_road_emb, full_grid_emb,
                  easy_filter_cache, gdata):
        """
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
        probs[:, 0, :] = self.rnnhidden2prob(
            lst_road_id=torch.tensor([[0]] * B).to(self.device),
            rnn_out=inputs.squeeze(1),
            gdata=gdata,
            full_road_emb=full_road_emb,
            easy_filter_cache=easy_filter_cache,
            first_constraint=True,
            tmp_grid=grid_traces[:, 0])  # to be finished by set mask = grid 1
        teacher_force = random.random() < tf_ratio
        if teacher_force:
            lst_road_id = tgt_roads[:, 0]
        else:
            lst_road_id = probs[:, 0, :].argmax(1)
        for t in range(1, max_RL):
            if teacher_force:
                inputs = full_road_emb[lst_road_id].view(B, 1, -1)
            inputs, hiddens = self.seq2seq.decode(inputs, hiddens,
                                                  encoder_outputs, attn_mask)

            probs[:, t, :] = self.rnnhidden2prob(
                lst_road_id=lst_road_id.view(-1, 1),
                rnn_out=inputs.squeeze(1),
                gdata=gdata,
                full_road_emb=full_road_emb,
                easy_filter_cache=easy_filter_cache,
                first_constraint=False,
                tmp_grid=grid_traces[:, 0])
            teacher_force = random.random() < tf_ratio
            if teacher_force:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs

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

            bptrs_t = torch.zeros(self.tagset_size,
                                  dtype=torch.long).to(self.device)

            # bptrs_t = []  # holds the backpointers for this step
            # viterbivars_t = []  # holds the viterbi variables for this step
            viterbivars_t = (torch.ones(1, self.tagset_size) *
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
