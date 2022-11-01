import torch.nn as nn
import torch
import random
from model.road_gin import RoadGIN
from model.trace_gcn import TraceGCN
from model.seq2seq import Seq2Seq
from model.crf import CRF
import torch.nn.functional as F


class GMM(nn.Module):
    def __init__(self,
                 emb_dim,
                 target_size,
                 topn,
                 neg_nums,
                 atten_flag=True,
                 drop_prob=0.5,
                 bi=True,
                 use_crf=True,
                 device="cpu") -> None:
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.target_size = target_size
        self.atten_flag = atten_flag
        self.use_crf = use_crf
        self.road_gin = RoadGIN(emb_dim)
        self.trace_gcn = TraceGCN(emb_dim)
        self.seq2seq = Seq2Seq(input_size=2 * emb_dim,
                               hidden_size=emb_dim,
                               atten_flag=atten_flag,
                               bi=bi,
                               drop_prob=drop_prob)
        self.road_feat_fc = nn.Linear(28, emb_dim)  # 3*8 + 4
        self.trace_feat_fc = nn.Linear(4, emb_dim)
        self.fc_input = nn.Linear(2 * self.emb_dim + 3, 2 * self.emb_dim)
        if self.use_crf:
            self.crf = CRF(num_tags=target_size,
                           emb_dim=emb_dim,
                           topn=topn,
                           neg_nums=neg_nums,
                           device=device)

    def forward(self, grid_traces, tgt_roads, traces_gps, traces_lens,
                road_lens, gdata, sample_Idx, tf_ratio):
        """
        grid_traces: id of traj points, (batch_size, seq_len)
        tgt_roads: ground truth, (batch_size, seq_len1)
        traces_gps: gps location of traj points, (batch_size, seq_len, 2)
        sample_Idx: (batch_size, seq_len)
        traces_lens, road_lens: list, real length of traj/ground truth
        """
        full_road_emb, full_grid_emb = self.get_emb(gdata)
        emissions = self.get_probs(grid_traces=grid_traces,
                                   tgt_roads=tgt_roads,
                                   traces_gps=traces_gps,
                                   trace_lens=traces_lens,
                                   road_lens=road_lens,
                                   tf_ratio=tf_ratio,
                                   full_grid_emb=full_grid_emb,
                                   gdata=gdata,
                                   sample_Idx=sample_Idx,
                                   full_road_emb=full_road_emb)
        if self.use_crf:
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool().to(self.device)
            loss = -self.crf(emissions, tgt_roads, full_road_emb.detach(), gdata.A_list, tgt_mask)
        else:
            mask = (tgt_roads.view(-1) != -1)
            loss = F.cross_entropy(emissions.view(-1, self.target_size)[mask], tgt_roads.view(-1)[mask])
        return loss

    def infer(self, grid_traces, traces_gps, traces_lens, road_lens,
              sample_Idx, gdata, tf_ratio):
        """
        make predictions
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
                                   full_road_emb=full_road_emb)
        if self.use_crf:
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool().to(self.device)
            preds = self.crf.decode(emissions, full_road_emb, gdata.A_list, tgt_mask)
        else:
            preds = F.softmax(emissions, dim=-1)
        return preds

    def get_emb(self, gdata):
        """
        get road embedding and grid embedding
        """
        # road embedding, (num_roads, embed_dim)
        road_x = self.road_feat_fc(gdata.road_x)
        full_road_emb = self.road_gin(road_x, gdata.road_adj)
        # (num_grids, emb_dim)
        pure_grid_feat = torch.mm(gdata.map_matrix, full_road_emb)
        pure_grid_feat[gdata.singleton_grid_mask] = self.trace_feat_fc(gdata.singleton_grid_location)
        # (num_grids, 2 * emb_dim)
        full_grid_emb = torch.zeros(gdata.num_grids + 1, 2 * self.emb_dim).to(self.device)
        full_grid_emb[1:, :] = self.trace_gcn(pure_grid_feat,
                                              gdata.trace_in_edge_index,
                                              gdata.trace_out_edge_index,
                                              gdata.trace_weight)
        return full_road_emb, full_grid_emb

    def get_probs(self, grid_traces, tgt_roads, traces_gps, sample_Idx,
                  trace_lens, road_lens, tf_ratio, full_road_emb,
                  full_grid_emb, gdata):
        """
        decode max_trace_lens times for tgt
        return (batch_size, max_road_lens, num_roads)
        """
        if tgt_roads is not None:
            B, max_RL = tgt_roads.shape
        else:
            B = grid_traces.shape[0]
            max_RL = int(max(road_lens))

        rnn_input = full_grid_emb[grid_traces]
        # concat rnn ouput with sampleIdx and traces_gps
        rnn_input = torch.cat([rnn_input, traces_gps, sample_Idx.unsqueeze(-1)], dim=-1)
        rnn_input = self.fc_input(rnn_input)
        # start encode
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
        inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs, attn_mask)
        probs[:, 0, :] = inputs.squeeze(1) @ full_road_emb.detach().T
        teacher_force = random.random() < tf_ratio
        if teacher_force:
            lst_road_id = tgt_roads[:, 0]
        else:
            lst_road_id = probs[:, 0, :].argmax(1)
        for t in range(1, max_RL):
            if teacher_force:
                inputs = full_road_emb[lst_road_id].view(B, 1, -1)
            inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs, attn_mask)
            probs[:, t, :] = inputs.squeeze(1) @ full_road_emb.detach().T
            teacher_force = random.random() < tf_ratio
            if teacher_force:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs
