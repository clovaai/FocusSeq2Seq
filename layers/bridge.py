"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import torch.nn as nn


class LinearBridge(nn.Module):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, rnn='GRU', activation='tanh'):
        super().__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.linear = nn.Linear(enc_hidden_size, dec_hidden_size)

        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.linear_h = nn.Linear(enc_hidden_size, dec_hidden_size)
            self.linear_c = nn.Linear(enc_hidden_size, dec_hidden_size)

        if activation.lower() == 'tanh':
            self.act = nn.Tanh()
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()

    def forward(self, hidden):
        """
           [2, B, enc_hidden_size // 2]
        => [B, 2, enc_hidden_size // 2] (transpose)
        => [B, enc_hidden_size]         (view)
        => [B, dec_hidden_size]         (linear)
        """

        if self.rnn_type == 'GRU':
            h = hidden

            B = h.size(1)

            h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)

            return self.act(self.linear(h))

        elif self.rnn_type == 'LSTM':
            h, c = hidden

            B = h.size(1)

            h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
            c = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)

            h = self.act(self.linear_h(h))
            c = self.act(self.linear_c(c))
            h = (h, c)

            return h
