import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        """
        hidden: (1, batch_szie, hidden_dim)
        encooder_outputs: (batch_size, src_len, hidden_dim)
        """
        src_len = encoder_outputs.shape[1]
        # batch_size, hidden_dim
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, hid dim * num directions]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elems.

        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 atten_flag=True,
                 bi=True,
                 drop_prob=0.5) -> None:
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.atten_flag = atten_flag
        self.drop_prob = drop_prob
        self.bi = bi
        self.D = 2 if self.bi else 1
        self.encoder = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=self.bi)
        dec_input_dim = hidden_size * self.D
        if self.atten_flag:
            self.attn = Attention(enc_hid_dim=hidden_size * self.D,
                                  dec_hid_dim=hidden_size)
            dec_input_dim += hidden_size
        self.decoder = nn.GRU(input_size=dec_input_dim,
                              hidden_size=hidden_size,
                              batch_first=True)

    def encode(self, src, src_len):
        self.encoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)
        # encode traj sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            src, src_len, batch_first=True, enforce_sorted=False)
        # (batch_size, seq_len, hidden_size)
        packed_outputs, hiddens = self.encoder(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs,
                                                      batch_first=True)
        if self.bi:
            hiddens = torch.sum(hiddens, dim=0, keepdims=True)
        # outputs (batch_size, seq_len, hidden_size * 2)
        # hiddens (1, batch_size, hidden_size)
        return outputs, hiddens

    def decode(self, src, hidden, encoder_outputs, attn_mask):
        """
        src: (batch_size, 1, emb_dim)
        hidden: (1, batch_size, hidden_size)
        encoder_outputs: (batch_size, src_len, num_directions * hidden_size)
        """
        self.decoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)
        if self.atten_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # (batch size, 1, src len)
            a = a.unsqueeze(1)
            # (batch_size, 1, num_directions * hidden_size)
            weighted = torch.bmm(a, encoder_outputs)
            src = torch.cat((weighted, src), dim=2)

        outputs, hiddens = self.decoder(src, hidden)
        return outputs, hiddens


if __name__ == "__main__":
    pass
