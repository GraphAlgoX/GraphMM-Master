import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    """
    Conditional random field.
    """

    def __init__(self, num_tags, device, batch_first=True, beamsize=5) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.device = device
        self.beamsize = beamsize

    def transition(self, tag1, tag2, full_road_emb, A_list):
        """
        tag1, tag2: (batch_size,)
        """
        # (batch_size, 1, emb_dim)
        emb1 = full_road_emb[tag1].unsqueeze(1)
        # (batch_size, emb_dim, 1)
        emb2 = full_road_emb[tag2].unsqueeze(-1)
        # (batch_size, )
        r = F.relu(torch.bmm(emb1, emb2)).squeeze()
        energy = A_list[tag1, tag2] * r
        return energy.flatten()

    def transitions(self, full_road_emb, A_list):
        # (num_tags, num_tags)
        attention = full_road_emb @ full_road_emb.T
        energy = A_list * attention
        return F.relu(energy)

    def forward(self, emissions, tags, full_road_emb, A_list, mask):
        """
        Compute the conditional log likelihood of a sequence of tags given emission scores.
        emissions: (batch_size, seq_length, num_tags)
        tags: (batch_size, seq_length)
        mask: (batch_size, seq_length)
        Returns: 
            The log likelihood.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, full_road_emb, A_list, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, full_road_emb, A_list, mask)
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
        return self._viterbi_decode(emissions, full_road_emb, A_list, mask)

    def _compute_score(self, emissions, tags, full_road_emb, A_list, mask):
        """
        S(X,y)
        emissions: (seq_length, batch_size, num_tags)
        tags: (seq_length, batch_size)
        mask: (seq_length, batch_size)
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
            score += self.transition(tags[i - 1], tags[i], full_road_emb, A_list) * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        return score

    def _compute_normalizer(self, emissions, full_road_emb, A_list, mask):
        """
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        """
        seq_length, batch_size, num_tags = emissions.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = emissions[0]
        trans = self.transitions(full_road_emb, A_list)
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)

            # broadcast_score = score.unsqueeze(2)
            for bs in range(batch_size):
                alphas_t = (torch.ones(1, num_tags) * float('-inf')).to(self.device)

                _, index = score[bs].topk(self.beamsize, largest=True)
                next_tag_set = A_list[index.squeeze(0), :].sum(dim=0).nonzero().squeeze(1)
                broadcast_emissions = emissions[i, bs, next_tag_set].unsqueeze(-1)
                next_score = score[bs].unsqueeze(0) + trans[next_tag_set, :] + broadcast_emissions
                alphas_t[0, next_tag_set] = torch.logsumexp(next_score, dim=-1)

                # for next_tag in next_tag_set:

                #     # Broadcast emission score for every possible current tag
                #     # shape: (batch_size, 1, num_tags)
                #     broadcast_emissions = emissions[i,bs,next_tag].unsqueeze(-1)

                #     # Compute the score tensor of size (batch_size, num_tags, num_tags) where
                #     # for each sample, entry at row i and column j stores the sum of scores of all
                #     # possible tag sequences so far that end with transitioning from tag i to tag j
                #     # and emitting
                #     # shape: (batch_size, num_tags, num_tags)
                #     print(score[bs].shape, trans[next_tag,:].shape)
                #     next_score = score[bs] + trans[next_tag,:] + broadcast_emissions
                #     alphas_t[0, next_tag] = torch.logsumexp(next_score, dim=-1).view(1)
                    # if mask[i, bs] != False:
                    #     score[i, next_tag] = alphas_t[0, next_tag]

                next_score = alphas_t
                # next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
                if mask[i, bs] == True:
                    score[bs] = next_score

        value, _ = score.topk(self.beamsize, largest=True)
        alpha = torch.logsumexp(value, dim=-1) + math.log(num_tags / self.beamsize)
        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return alpha#torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, full_road_emb, A_list, mask):
        """
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        """

        seq_length, batch_size, num_tags = emissions.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = emissions[0]
        history = []

        trans = self.transitions(full_road_emb, A_list)

        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            # broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            # broadcast_emission = emissions[i].unsqueeze(1)
            indices = torch.zeros(batch_size, num_tags).to(self.device)
            for bs in range(batch_size):
                alphas_t = (torch.ones(1, num_tags) * float('-inf')).to(self.device)

                _, index = score[bs].topk(self.beamsize, largest=True)
                next_tag_set = A_list[index.squeeze(0), :].sum(dim=0).nonzero().squeeze(1)
                broadcast_emissions = emissions[i, bs, next_tag_set].unsqueeze(-1)
                next_score = score[bs].unsqueeze(0) + trans[next_tag_set, :] + broadcast_emissions
                alphas_t[0, next_tag_set] = torch.logsumexp(next_score, dim=-1)
                next_score = alphas_t
                # next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
                if mask[i, bs] == True:
                    score[bs] = next_score

                next_score, indices[bs] = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            # score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [int(best_last_tag.item())]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(int(best_last_tag.item()))

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            pad_lens = seq_length - len(best_tags)
            for i in range(pad_lens):
                best_tags.append(-1)
            best_tags_list.append(best_tags)

        return best_tags_list