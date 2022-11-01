import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CRF(nn.Module):
    """
    Conditional random field.
    """

    def __init__(self,
                 num_tags,
                 emb_dim,
                 topn,
                 neg_nums,
                 device='cpu',
                 batch_first=True) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.device = device
        self.topn = topn
        self.neg_nums = neg_nums
        self.W = nn.Linear(emb_dim, emb_dim, bias=False)

    def get_transitions(self, full_road_emb, A_list):
        # (num_tags, num_tags)
        r = self.W(full_road_emb) @ full_road_emb.T
        energy = A_list * F.relu(r)
        return energy

    def forward(self, emissions, tags, full_road_emb, A_list, mask):
        """
        Compute the conditional log likelihood of a sequence of tags given emission scores.
        emissions: (batch_size, seq_length, num_tags)
        tags: (batch_size, seq_length)
        mask: (batch_size, seq_length)
        Returns: 
            The log likelihood.
        """
        batch_size = mask.size(0)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        # get trainsition matrix
        transitions = self.get_transitions(full_road_emb, A_list)
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, transitions, mask)
        # sample neg_nums satus
        seq_ends = mask.long().sum(dim=0) - 1
        neg_tag_sets = set()
        for i in range(batch_size):
            neg_tag_sets |= set(tags[:seq_ends[i] + 1, i].detach().cpu().numpy().tolist())
        assert len(neg_tag_sets) < self.neg_nums
        remain_nums = self.neg_nums - len(neg_tag_sets)
        # sample from topk
        if remain_nums > 0:
            _, indices = torch.topk(emissions, dim=-1, k=3)
            tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
            cand_set = [i for i in tag_sets if i not in neg_tag_sets]
            cand_num = len(cand_set)
            neg_tag_sets |= set(np.random.choice(cand_set, min(remain_nums, cand_num), replace=False).tolist())
        neg_tag_sets = sorted(list(neg_tag_sets))
        trans = transitions[neg_tag_sets, :]
        trans = trans[:, neg_tag_sets]
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, trans, neg_tag_sets, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        return llh.sum() / mask.float().sum()

    def decode(self, emissions, full_road_emb, A_list, mask):
        """
        Find the most likely tag sequence using Viterbi algorithm.
        emissions: (batch_size, seq_length, num_tags)
        mask: (batch_size, seq_length)
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        transitions = self.get_transitions(full_road_emb, A_list)
        return self._viterbi_decode(emissions, transitions, mask)

    def _compute_score(self, emissions, tags, transitions, mask):
        """
        S(X,y)
        emissions: (seq_length, batch_size, num_tags)
        tags: (seq_length, batch_size)
        mask: (seq_length, batch_size)
        transitions: (num_tags, num_tags)
        return: (batch_size, )
        """

        seq_length, batch_size = tags.shape
        mask = mask.float()
        # Start transition score and first emission
        # shape: (batch_size,)
        score = torch.zeros(batch_size).to(self.device)
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        return score

    def _compute_normalizer(self, emissions, trans, neg_tag_sets, mask):
        """
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        trans: (k, k), k = |neg_tag_sets|
        """
        seq_length = emissions.size(0)
        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = emissions[0, :, neg_tag_sets]
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i, :, neg_tag_sets].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + trans + broadcast_emissions
            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        # # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, transitions, mask):
        """
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        transitions: (num_tags, num_tags)
        """

        seq_length, batch_size = mask.shape

        # gain topk
        _, indices = torch.topk(emissions, dim=-1, k=self.topn)
        tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
        tag_sets = sorted(tag_sets)
        tag_map = {i: tag for i, tag in enumerate(tag_sets)}
        # gain sub transition prob matrix
        trans = transitions[tag_sets, :]
        trans = trans[:, tag_sets]
        # Start transition and first emission
        # shape: (batch_size, k)
        score = emissions[0, :, tag_sets]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        # next_score = torch.zeros(batch_size, self.num_tags).to(self.device)
        # indices = torch.zeros(batch_size, self.num_tags).int()
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, k, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, k)
            broadcast_emission = emissions[i, :, tag_sets].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, k, k)
            next_score = broadcast_score + trans + broadcast_emission
            # for j in range(batch_size):
            #     cur_score, cur_indices = torch.max(score[j].unsqueeze(1) + trans + emissions[i,j,:].unsqueeze(0), dim=0)
            #     next_score[j] = cur_score
            #     indices[j] = cur_indices.detach().cpu()
            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)
            # print(next_score.shape)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Now, compute the best path for each sample
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags = [tag_map[t] for t in best_tags]
            tags_len = len(best_tags)
            best_tags_list.append(best_tags + [-1] * (seq_length - tags_len))
        return best_tags_list
