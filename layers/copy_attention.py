"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import math

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self,
                 enc_hidden_size=512, dec_hidden_size=256,
                 attention_size=700,
                 coverage=False,
                 weight_norm=False,
                 bias=True,
                 pointer_end_bias=False):
        """Bahdanau Attention (+ Coverage)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.coverage = coverage
        self.bias = bias
        self.weight_norm = weight_norm
        self.end_bias = pointer_end_bias

        # encoder state
        self.Wh = nn.Linear(enc_hidden_size, attention_size, bias=False)
        # decoder state
        self.Ws = nn.Linear(dec_hidden_size, attention_size, bias=False)
        # coverage
        if coverage:
            self.Wc = nn.Linear(1, attention_size, bias=False)

        if bias:
            self.b = nn.Parameter(torch.randn(1, 1, attention_size))

        # to scalar
        if weight_norm:
            v = nn.Linear(attention_size, 1, bias=False)
            self.v = nn.utils.weight_norm(v)
        else:
            self.v = nn.Linear(attention_size, 1, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        if self.end_bias:
            self.end_energy = nn.Parameter(torch.randn(1, 1))

    def forward(self, encoder_outputs, decoder_state, mask, coverage=None):
        """
        Args:
            encoder_outputs [B, source_len, hidden_size]
            decoder_state [B, hidden_size]
            mask [B, source_len]
            coverage [B, source_len] (optional)
        Return:
            attention [B, source_len]

        e = v T tanh (Wh @ h + Ws @ s + b)
        a = softmax(e)

        e = v T tanh (Wh @ h + Ws @ s + Wc @ c + b) <= coverage
        a = softmax(e; bias) <= bias
        """
        B, source_len, _ = encoder_outputs.size()

        # Attention Energy
        # [B, source_len, hidden_size]
        enc_out_energy = self.Wh(encoder_outputs)
        # [B, 1, hidden_size]
        dec_state_energy = self.Ws(decoder_state).unsqueeze(1)
        # [B, source_len, hidden_size]
        energy = enc_out_energy + dec_state_energy

        if self.coverage:
            # [B, source_len] => [B, source_len, 1] => [B, source_len, hidden_size]
            try:
                cov_energy = self.Wc(coverage.unsqueeze(2))
                energy = energy + cov_energy
            except RuntimeError:
                print(energy.size())
                print(coverage.size())

        if self.bias:
            energy = energy + self.b

        energy = self.tanh(energy)  # [B, source_len, hidden_size]
        energy = self.v(energy).squeeze(2)  # [B, source_len]

        # mask out attention outside of answer sentence
        energy.masked_fill_(mask, -math.inf)

        if self.end_bias:
            # [B, 1]
            end_energy = self.end_energy.expand(B, 1)
            # [B, source_len+1]
            energy = torch.cat([energy, end_energy], dim=1)

        attention = self.softmax(energy)  # [B, source_len]

        return attention


class CopySwitch(nn.Module):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=256):
        """Pointing the Unknown Words (ACL 2016)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        # self.W = nn.Linear(hidden_size,  1)
        # self.U = nn.Linear(hidden_size,  1, bias=False)

        self.W = nn.Linear(enc_hidden_size + dec_hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder_state, context):
        """
        Args:
            decoder_state [B, hidden_size]
            context [B, hidden_size]
        Return:
            p [B, 1]

        p = sigmoid(W @ s + U @ c + b)
        """
        # [B, 1]
        # p = self.W(decoder_state) + self.U(context)
        p = self.W(torch.cat([decoder_state, context], dim=1))
        p = self.sigmoid(p)

        return p


class PointerGenerator(nn.Module):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=128, rnn_type='LSTM'):
        """Estimation of Word Generation (vs Copying) Probability
        Get To The Point: Summarization with Pointer-Generator Networks (ACL 2017)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.rnn_type = rnn_type

        # # context vector
        # self.wh = nn.Linear(hidden_size, 1, bias=False)
        # # decoder state
        # self.ws = nn.Linear(hidden_size, 1, bias=False)
        # # target embedding
        # self.wx = nn.Linear(embed_size,  1, bias=False)
        #
        # self.b = nn.Parameter(torch.randn(1, 1))

        self.W = nn.Linear(enc_hidden_size + dec_hidden_size + embed_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, context, decoder_state, decoder_input):
        """
        Args:
            context [B, hidden_size]
            decoder_state [B, hidden_size]
            decoder_input [B, embed_size]
        Return:
            p_gen [B, 1]

        p = sigmoid(wh @ h + ws @ s + wx @ x + b)
        """
        # p_gen = self.wh(context) \
        #     + self.ws(decoder_state) \
        #     + self.wx(decoder_input) \
        #     + self.b # [batch, 1]
        p_gen = self.W(torch.cat([context, decoder_state, decoder_input], dim=1))

        p_gen = self.sigmoid(p_gen)  # [batch, 1]

        return p_gen
