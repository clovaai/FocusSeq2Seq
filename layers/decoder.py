"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .copy_attention import BahdanauAttention, CopySwitch, PointerGenerator
from .beam_search import Beam


def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]

    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
        tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out


class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(
                x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] //
                      self._pool_size, self._pool_size).max(-1)
        return m


class NQGReadout(nn.Module):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=300, vocab_size=30000, dropout_p=0.5,
                 pool_size=2, tie=False):
        super().__init__()

        self.tie = tie

        if tie:
            self.W = nn.Linear(embed_size + enc_hidden_size + dec_hidden_size, embed_size)
            self.Wo = nn.Linear(embed_size, vocab_size, bias=False)
            self.dropout = nn.Dropout(dropout_p)

        else:
            # self.Wr = nn.Linear(embed_size,      hidden_size, bias=False)
            # self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
            # self.Vr = nn.Linear(hidden_size,     hidden_size, bias=False)
            self.W = nn.Linear(embed_size + enc_hidden_size + dec_hidden_size, dec_hidden_size)
            self.maxout = Maxout(pool_size=pool_size)
            self.dropout = nn.Dropout(dropout_p)
            self.Wo = nn.Linear(dec_hidden_size // pool_size, vocab_size, bias=False)

    def forward(self, word_emb, context, decoder_state):
        # [B, 2 * hidden_size]

        if self.tie:
            r = self.W(torch.cat([word_emb, context, decoder_state], dim=1))

            r = torch.tanh(r)
            r = self.dropout(r)
            energy = self.Wo(r)

        else:
            r = self.W(torch.cat([word_emb, context, decoder_state], dim=1))
            # r = self.Wr(word_emb) + self.Ur(context) + self.Vr(decoder_state)

            # [B, hidden_size // 2]
            r = self.maxout(r)
            r = self.dropout(r)

            # [B, vocab_size]
            energy = self.Wo(r)

        return energy


class PGReadout(nn.Module):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=128, vocab_size=50000, dropout_p=0.5,
                 tie=False):
        super().__init__()

        self.tie = tie

        if tie:
            self.W1 = nn.Linear(enc_hidden_size + dec_hidden_size, embed_size)
            self.dropout = nn.Dropout(dropout_p)
            self.tanh = nn.Tanh()
            self.Wo = nn.Linear(embed_size, vocab_size, bias=False)

        else:
            if dropout_p > 0:
                self.mlp = nn.Sequential(
                    nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size),
                    nn.Dropout(dropout_p),
                    nn.Linear(dec_hidden_size, vocab_size)
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size),
                    # nn.Dropout(dropout_p),
                    nn.Linear(dec_hidden_size, vocab_size)
                )

    def forward(self, context, decoder_state):
        # [B, hidden_size] | [B, hidden_size]

        if self.tie:
            # => [B, hidden_size]
            r = self.W1(torch.cat([context, decoder_state], dim=1))
            r = self.dropout(r)
            r = self.tanh(r)

            # => [B, vocab_size]
            energy = self.Wo(r)
        else:
            energy = self.mlp(torch.cat([context, decoder_state], dim=1))

        return energy


class ASReadout(nn.Module):
    def __init__(self, enc_hidden_size=350, dec_hidden_size=350, embed_size=300,
                 vocab_size=34000, dropout_p=0.4, tie=True):
        super().__init__()

        self.tie = tie

        self.Wq = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size, bias=False)
        # self.dropout = nn.Dropout(dropout_p)
        self.tanh = nn.Tanh()
        self.Wa = nn.Linear(dec_hidden_size, embed_size, bias=False)
        self.Wo = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, context, decoder_state):
        # [B, hidden_size] | [B, hidden_size]
        # => [B, hidden_size]
        q = self.Wq(torch.cat([context, decoder_state], dim=1))
        q = self.tanh(q)
        # q = self.dropout(q)

        # => [B, embed_size]
        r = self.Wa(q)

        # => [B, vocab_size]
        energy = self.Wo(r)

        return energy


class NQGDecoder(nn.Module):
    def __init__(self, embed_size=300, enc_hidden_size=512, dec_hidden_size=512,
                 attention_size=512,
                 vocab_size=20000, dropout_p=0.5,
                 rnn='GRU', tie=False, n_mixture=None):
        """Neural Question Generation from Text: A Preliminary Study (2017)
        incorporated with copying mechanism of Pointing the Unknown Words (ACL 2016)"""
        super().__init__()

        input_size = embed_size + dec_hidden_size

        self.hidden_size = dec_hidden_size
        self.vocab_size = vocab_size

        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.rnncell = nn.GRUCell(input_size=input_size,
                                      hidden_size=dec_hidden_size)
        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnncell = nn.LSTMCell(input_size=input_size,
                                       hidden_size=dec_hidden_size)

        # Attention
        self.attention = BahdanauAttention(
            enc_hidden_size, dec_hidden_size, attention_size=attention_size)
        self.copy_switch = CopySwitch(enc_hidden_size, dec_hidden_size)

        self.readout = NQGReadout(enc_hidden_size=enc_hidden_size,
                                  dec_hidden_size=dec_hidden_size,
                                  embed_size=embed_size,
                                  vocab_size=vocab_size,
                                  dropout_p=dropout_p, tie=tie)

        self.tie = tie

        self.n_mixture = n_mixture
        if n_mixture:
            self.mixture_embedding = nn.Embedding(n_mixture, embed_size)

    def forward(self,
                enc_outputs,
                s,
                source_WORD_encoding,
                answer_WORD_encoding=None,
                mixture_id=None,
                target_WORD_encoding=None,
                source_WORD_encoding_extended=None,
                train=True,
                decoding_type='beam',
                K=10,
                max_dec_len=30,
                temperature=1.0,
                diversity_lambda=0.5):

        device = enc_outputs.device
        B, max_source_len = source_WORD_encoding.size()
        V = self.vocab_size + max_source_len

        PAD_ID = 0  # word2id['<pad>']
        UNK_ID = 1  # word2id['<unk>']
        SOS_ID = 2  # word2id['<sos>']
        EOS_ID = 3  # word2id['<eos>']

        # Initial input for decoder (Start of sequence token; SOS)
        dec_input_word = torch.tensor([SOS_ID] * B, dtype=torch.long, device=device)

        # attention mask [B, max_source_len]
        pad_mask = (source_WORD_encoding == PAD_ID)

        if train:
            # Number of decoding iteration
            max_dec_len = target_WORD_encoding.size(1)
        else:
            # Number of decoding iteration
            max_dec_len = max_dec_len

            # Repeat with beam/group size
            if decoding_type in ['beam', 'diverse_beam', 'topk_sampling']:
                # [B] => [B*K]
                dec_input_word = repeat(dec_input_word, K)
                # [B, hidden_size] => [B*K, hidden_size]
                if self.rnn_type == 'GRU':
                    s = repeat(s, K)
                elif self.rnn_type == 'LSTM':
                    s = (repeat(s[0], K), repeat(s[1], K))
                # [B*K, max_source_len, hidden*2]
                enc_outputs = repeat(enc_outputs, K)
                # [B*K, max_source_len]
                pad_mask = repeat(pad_mask, K)

                if decoding_type in ['beam', 'diverse_beam']:
                    # Initialize log probability scores [B, K]
                    score = torch.zeros(B, K, device=device)
                    # Avoid duplicated beams
                    score[:, 1:] = -math.inf
                    # Initialize Beam
                    beam = Beam(B, K, EOS_ID)
                    n_finished = torch.zeros(B, dtype=torch.long, device=device)

                elif decoding_type == 'topk_sampling':
                    # Initialize log probability scores [B, K]
                    score = torch.zeros(B * K, device=device)
                    finished = torch.zeros(B * K, dtype=torch.uint8, device=device)

            else:
                finished = torch.zeros(B, dtype=torch.uint8, device=device)
                score = torch.zeros(B, device=device)

        # Outputs will be concatenated
        out_log_p = []
        output_sentence = []

        # for attention visualization
        self.attention_list = []

        for i in range(max_dec_len):

            if i == 0 and self.n_mixture:
                dec_input_word_embed = self.mixture_embedding(mixture_id)
            else:
                dec_input_word_embed = self.word_embed(dec_input_word)

            # Feed context vector to decoder rnn
            if i == 0:
                # [B, hidden_size]
                context = torch.zeros_like(enc_outputs[:, 0])
            dec_input = torch.cat([dec_input_word_embed, context], dim=1)

            s = self.rnncell(dec_input, s)

            # ------ Attention ------#
            # Bahdanau Attention (Softmaxed)
            # [B, max_source_len]
            if self.rnn_type == 'GRU':
                attention = self.attention(enc_outputs, s, pad_mask)
            if self.rnn_type == 'LSTM':
                attention = self.attention(enc_outputs, s[0], pad_mask)

            self.attention_list.append(attention)

            # Context vector: attention-weighted sum of enc outputs
            # [B, 1, max_source_len] @ [B, max_source_len, hidden]
            # => [B, 1, hidden]
            context = torch.bmm(attention.unsqueeze(1), enc_outputs)
            # => [B, hidden]
            context = context.squeeze(1)

            # ------ Copy Switch ------#
            # [B, 1]
            if self.rnn_type == 'GRU':
                p_copy = self.copy_switch(s, context)
            if self.rnn_type == 'LSTM':
                p_copy = self.copy_switch(s[0], context)

            # ------ Output Word Probability (Before Softmax) ------#
            # [B, vocab_size]
            if self.rnn_type == 'GRU':
                p_vocab = self.readout(dec_input_word_embed, context, s)
            if self.rnn_type == 'LSTM':
                p_vocab = self.readout(dec_input_word_embed, context, s[0])

            # suppress PAD / UNK tokens in the decoding
            if not train:
                p_vocab[:, PAD_ID] = -math.inf
                p_vocab[:, UNK_ID] = -math.inf

            p_vocab = F.softmax(p_vocab, dim=1)

            # V = vocab_size + max_source_len
            # [B, V]
            p_out = torch.cat([(1 - p_copy) * p_vocab,
                               p_copy * attention], dim=1)
            # if train:
            p_out = p_out + 1e-12

            log_p = p_out.log()
            out_log_p.append(log_p)

            # ------ Training ------#
            if train:
                # ------ Teacher forcing ------#
                # dec_input_word = question[:, i]
                # dec_input_word.masked_fill_(dec_input_word >= self.vocab_size, UNK_ID)

                dec_input_word = target_WORD_encoding[:, i]
                unk = torch.full_like(dec_input_word, UNK_ID)
                dec_input_word = torch.where(
                    dec_input_word >= self.vocab_size, unk, dec_input_word)

            # ------ Decoding ------#
            else:
                if decoding_type in ['beam', 'diverse_beam']:
                    # [B*K, V] => [B, K, V]
                    current_score = log_p.view(B, K, V)

                    # if i+1 < min_dec_len:
                    #     current_score[:, EOS_ID] = -math.inf

                    # Reduce Branching Size from Vocab size => K
                    # [B, K, K]
                    # current_score_topk, current_score_topk_idx = current_score.topk(K, dim=2)

                    # [B, K] => [B, K, 1] => [B, K, K]
                    # score = score.view(B, K, 1) + current_score_topk

                    if decoding_type == 'diverse_beam':
                        diversity_penalty = torch.zeros(B, V, device=device)
                        for k in range(K):
                            # [B, V]
                            current_beam_score = current_score[:, k]
                            if k > 0:
                                diversity_penalty.scatter_add_(
                                    1, beam_word_id, torch.ones(B, V, device=device))
                                current_beam_score -= diversity_lambda * diversity_penalty
                            # [B, 1]
                            beam_word_id = current_beam_score.argmax(dim=1, keepdim=True)

                    # [B, K] => [B, K, 1] => [B, K, V]
                    score = score.view(B, K, 1) + current_score

                    # [B, K, V] => [B, K*V]
                    score = score.view(B, -1)

                    # Select top k 'vocab' candidates (range: 0 ~ K*V-1)
                    # [B, K]
                    topk_score, topk_idx = score.topk(K, dim=1)

                    # [B, K]
                    # topk_idx = current_score_topk_idx.view(B, K*K).gather(1, topk_idx)

                    # [B, K], (0 ~ K-1)
                    topk_beam_idx = topk_idx // V
                    # [B, K], (0 ~ V-1)
                    topk_word_id = topk_idx % V

                    beam.back_pointers.append(topk_beam_idx.clone())  # [B, K]
                    beam.token_ids.append(topk_word_id.clone())  # [B, K]
                    beam.scores.append(topk_score.clone())  # [B, K]

                    # Top-k scored sequence index in each batch (which batch + which beam inside batch)
                    # batch_indices [B, K]
                    #   [[0,         ...     0        ],
                    #    [K,         ...     K        ],
                    #    [K * 2,     ...     K * 2    ],
                    #                ...
                    #    [K * (B-1), ...,    K * (B-1)]
                    batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
                    batch_indices = batch_starting_indices.unsqueeze(1)
                    topk_beam_idx_flat = (batch_indices + topk_beam_idx).flatten()

                    # Prepare next iteration
                    # [B*K, hidden_size]
                    # [B, hidden_size] => [B*K, hidden_size]
                    if self.rnn_type == 'GRU':
                        s = s.index_select(0, topk_beam_idx_flat)
                    elif self.rnn_type == 'LSTM':
                        s = (s[0].index_select(0, topk_beam_idx_flat),
                             s[1].index_select(0, topk_beam_idx_flat))

                    score = topk_score  # [B, K]
                    where_EOS = topk_word_id == EOS_ID
                    score.masked_fill_(where_EOS, -math.inf)

                    predicted_word_id = topk_word_id.flatten()
                    where_oov = predicted_word_id >= self.vocab_size
                    dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)

                    # [B, K]
                    generated_eos = topk_word_id == EOS_ID
                    if generated_eos.any():
                        # [B]
                        n_finished += generated_eos.long().sum(dim=1)
                        if n_finished.min().item() >= K:
                            break

                elif decoding_type in ['greedy', 'topk_sampling']:
                    if decoding_type == 'greedy':
                        # [B]
                        log_p_sampled, predicted_word_id = log_p.max(dim=1)

                    # Hierarchical Neural Story Generation (ACL 2018)
                    elif decoding_type == 'topk_sampling':
                        topk = 10
                        # [B*K, topk]
                        log_p_topk, predicted_word_id_topk = log_p.topk(
                            topk, dim=1)

                        temperature_scaled_score = (log_p_topk / temperature).exp()

                        # [B*K, 1]
                        sampled_idx = temperature_scaled_score.multinomial(1)

                        # [B*K]
                        log_p_sampled = temperature_scaled_score.gather(
                            1, sampled_idx).squeeze(1)
                        predicted_word_id = predicted_word_id_topk.gather(
                            1, sampled_idx).squeeze(1)

                    log_p_sampled.masked_fill_(finished, 0)
                    score += log_p_sampled

                    where_oov = predicted_word_id >= self.vocab_size
                    dec_input_word = predicted_word_id.masked_fill(
                        where_oov, UNK_ID)

                    output_sentence.append(predicted_word_id)

                    # [B]
                    generated_eos = predicted_word_id == EOS_ID
                    if generated_eos.any():
                        finished += generated_eos
                        finished.clamp_(0, 1)

                        if finished.min().item() > 0:
                            break

            # ------ End-of-Decoder! ------#

        if train:
            # [B, max_dec_len, V]
            log_p = torch.stack(out_log_p, dim=1)
            return log_p

        else:
            if decoding_type in ['beam', 'diverse_beam']:
                # Top-K sentence / score
                output_sentence, score = beam.backtrack()  # [B, K, max_dec_len]
                # [B, K, max_dec_len], [B, K]
                return output_sentence, score
            else:
                output_sentence = torch.stack(output_sentence, dim=-1).view(B, 1, -1)
                # [B, 1, max_dec_len], [B, 1]
                return output_sentence, score


class PGDecoder(nn.Module):
    def __init__(self, embed_size=128, enc_hidden_size=512, dec_hidden_size=256,
                 attention_size=700,
                 vocab_size=50000, dropout_p=0.0,
                 rnn='LSTM', tie=False, n_mixture=None):
        """Get To The Point: Summarization with Pointer-Generator Networks (ACL 2017)"""
        super().__init__()

        input_size = embed_size
        self.input_linear = nn.Linear(embed_size + enc_hidden_size, input_size)

        self.hidden_size = dec_hidden_size
        self.vocab_size = vocab_size

        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.rnncell = nn.GRUCell(input_size=input_size,
                                      hidden_size=dec_hidden_size)
        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnncell = nn.LSTMCell(input_size=input_size,
                                       hidden_size=dec_hidden_size)

        # Attention
        self.attention = BahdanauAttention(enc_hidden_size=enc_hidden_size,
                                           dec_hidden_size=dec_hidden_size if rnn == 'GRU' else 2 * dec_hidden_size,
                                           attention_size=attention_size,
                                           coverage=True)
        self.pointer_switch = PointerGenerator(enc_hidden_size=enc_hidden_size,
                                               dec_hidden_size=dec_hidden_size if rnn == 'GRU' else 2 * dec_hidden_size,
                                               embed_size=embed_size)

        self.readout = PGReadout(enc_hidden_size=enc_hidden_size,
                                 dec_hidden_size=dec_hidden_size,
                                 embed_size=embed_size,
                                 vocab_size=vocab_size,
                                 dropout_p=dropout_p,
                                 tie=tie)

        self.tie = tie

        self.n_mixture = n_mixture
        if n_mixture:
            self.mixture_embedding = nn.Embedding(n_mixture, embed_size)

    def forward(self,
                enc_outputs,
                s,
                source_WORD_encoding,
                answer_WORD_encoding=None,
                mixture_id=None,
                target_WORD_encoding=None,
                source_WORD_encoding_extended=None,
                train=True,
                decoding_type='beam',
                K=10,
                max_dec_len=100,
                temperature=1.0,
                diversity_lambda=0.5):

        device = enc_outputs.device
        B, max_source_len = source_WORD_encoding.size()

        PAD_ID = 0  # word2id['<pad>']
        UNK_ID = 1  # word2id['<unk>']
        SOS_ID = 2  # word2id['<sos>']
        EOS_ID = 3  # word2id['<eos>']

        # Initial input for decoder (Start of sequence token; SOS)
        dec_input_word = torch.tensor([SOS_ID] * B, dtype=torch.long, device=device)

        # attention mask [B, max_source_len]
        pad_mask = (source_WORD_encoding == PAD_ID)

        # coverage [B, max_source_len]
        coverage = torch.zeros(B, max_source_len, device=device)

        max_n_oov = source_WORD_encoding_extended.max().item() - self.vocab_size + 1
        max_n_oov = max(max_n_oov, 1)
        V = self.vocab_size + max_n_oov

        if train:
            # Number of decoding iteration
            max_dec_len = target_WORD_encoding.size(1)

        else:
            # Number of decoding iteration
            max_dec_len = max_dec_len
            # Repeat with beam/group size
            if decoding_type in ['beam', 'diverse_beam', 'topk_sampling']:
                # [B] => [B*K]
                dec_input_word = repeat(dec_input_word, K)
                # [B, hidden_size] => [B*K, hidden_size]
                if self.rnn_type == 'GRU':
                    s = repeat(s, K)
                elif self.rnn_type == 'LSTM':
                    s = (repeat(s[0], K), repeat(s[1], K))
                # [B*K, max_source_len, hidden*2]
                enc_outputs = repeat(enc_outputs, K)
                # [B*K, max_source_len]
                pad_mask = repeat(pad_mask, K)
                # [B*K, max_source_len]
                source_WORD_encoding_extended = repeat(source_WORD_encoding_extended, K)
                # [B*K, max_source_len]
                coverage = repeat(coverage, K)

                if decoding_type in ['beam', 'diverse_beam']:
                    # Initialize log probability scores [B, K]
                    score = torch.zeros(B, K, device=device)
                    # Avoid duplicated beams
                    score[:, 1:] = -math.inf
                    # Initialize Beam
                    beam = Beam(B, K, EOS_ID)
                    n_finished = torch.zeros(B, dtype=torch.long, device=device)

                elif decoding_type == 'topk_sampling':
                    # Initialize log probability scores [B, K]
                    score = torch.zeros(B * K, device=device)
                    finished = torch.zeros(B * K, dtype=torch.uint8, device=device)

            else:
                finished = torch.zeros(B, dtype=torch.uint8, device=device)
                score = torch.zeros(B, device=device)

        # Outputs will be concatenated
        out_log_p = []
        output_sentence = []
        coverage_loss_list = []

        # for attention visualization
        self.attention_list = []

        for i in range(max_dec_len):

            if i == 0 and self.n_mixture:
                dec_input_word_embed = self.mixture_embedding(mixture_id)
            else:
                dec_input_word_embed = self.word_embed(dec_input_word)

            # Feed context vector to decoder rnn
            if i == 0:
                # [B, hidden_size]
                context = torch.zeros_like(enc_outputs[:, 0])
            dec_input = self.input_linear(torch.cat([dec_input_word_embed, context], dim=1))

            s = self.rnncell(dec_input, s)

            if self.rnn_type == 'LSTM':
                s_cat = torch.cat([s[0], s[1]], dim=1)

            # ------ Attention ------#
            # Bahdanau Attention (Softmaxed)
            # [B, max_source_len]
            if self.rnn_type == 'GRU':
                attention = self.attention(enc_outputs, s, pad_mask, coverage)
            if self.rnn_type == 'LSTM':
                attention = self.attention(enc_outputs, s_cat, pad_mask, coverage)

            self.attention_list.append(attention)

            # Context vector: attention-weighted sum of enc outputs
            # [B, 1, max_source_len] * [B, max_source_len, hidden]
            # => [B, 1, hidden]
            context = torch.bmm(attention.unsqueeze(1), enc_outputs)
            # => [B, hidden]
            context = context.squeeze(1)

            # #------ Coverage ------#
            if train:
                # [B]
                step_coverage_loss = torch.sum(torch.min(attention, coverage), dim=1)
                coverage_loss_list.append(step_coverage_loss)

            # Update coverage vector by adding attention
            coverage = coverage + attention
            # assert coverage.min().item() >= 0

            # ------ Copy Switch ------#
            # P(Generation)
            # [B, 1]
            if self.rnn_type == 'GRU':
                p_gen = self.pointer_switch(context, s, dec_input)
            if self.rnn_type == 'LSTM':
                p_gen = self.pointer_switch(context, s_cat, dec_input)

            # ------ Output Word Probability ------#
            # 1) In-Vocab Probability: P(Vocab)
            # [B, Vocab_size]
            if self.rnn_type == 'GRU':
                p_vocab = self.readout(context, s)
            if self.rnn_type == 'LSTM':
                p_vocab = self.readout(context, s[0])

            # suppress PAD / UNK tokens in the decoding
            if not train:
                p_vocab[:, PAD_ID] = -math.inf
                p_vocab[:, UNK_ID] = -math.inf
            p_vocab = F.softmax(p_vocab, dim=1)

            # 2) Copy from answer sentence: P(Copy)
            # If attended to
            #     In-Vocab Words  => Add to In-Vocab Probability
            #     OOV Words (UNK) => Pointer candidates
            # p_copy = torch.zeros(p_vocab.size(0), V, device=device)

            # p_copy.scatter_add_(1, source_WORD_encoding, attention)
            # if not train:
            #     p_copy[:, UNK_ID] = 0

            # 3) Total Prob: Mix P(Vocab) and P(Copy) with P(Generation)
            # [B, V]
            # = [B, vocab_size + n_max_oovs]
            # p_vocab = torch.cat([
            #     p_vocab,
            #     torch.zeros(p_vocab.size(0), max_n_oov, device=device)], dim=1)

            # ext_zero_size = (p_vocab.size(0), max_n_oov)
            # print(ext_zero_size)
            ext_zeros = torch.zeros(p_vocab.size(0), max_n_oov, device=device)
            p_out = torch.cat([p_vocab, ext_zeros], dim=1)
            p_out = p_out * p_gen

            # p_out = torch.cat(
            #     [p_gen * p_vocab, torch.zeros(p_vocab.size(0), max_n_oov, device=device)], dim=1)
            p_out.scatter_add_(1, source_WORD_encoding_extended, (1 - p_gen) * attention)
            if not train:
                p_out[:, UNK_ID] = 0

            # p_out = p_gen * p_vocab + (1 - p_gen) * p_copy

            # if train:
            p_out = p_out + 1e-12

            log_p = p_out.log()
            out_log_p.append(log_p)

            # ------ Training ------#
            if train:
                # ------ Teacher forcing ------#
                # dec_input_word = question[:, i]
                # dec_input_word.masked_fill_(dec_input_word >= self.vocab_size, UNK_ID)

                dec_input_word = target_WORD_encoding[:, i]
                unk = torch.full_like(dec_input_word, UNK_ID)
                dec_input_word = torch.where(
                    dec_input_word >= self.vocab_size, unk, dec_input_word)

            # ------ Decoding ------#
            else:
                if decoding_type in ['beam', 'diverse_beam']:

                    # [B*K, V] => [B, K, V]
                    current_score = log_p.view(B, K, V)

                    # if decoding_type == 'diverse_beam':
                    #
                    #     diversity_score = 0
                    #
                    #     current_score = current_score + diversity_score

                    if max_dec_len > 30:
                        min_dec_len = 35
                        if i + 1 < min_dec_len:
                            current_score[:, :, EOS_ID] = -math.inf
                            # current_score[:, EOS_ID] = -1e7

                    # Reduce Branching Size from Vocab size => K
                    # [B, K, K]
                    # current_score_topk, current_score_topk_idx = current_score.topk(K, dim=2)

                    # [B, K] => [B, K, 1] => [B, K, K]
                    # score = score.view(B, K, 1) + current_score_topk

                    if decoding_type == 'diverse_beam':
                        diversity_penalty = torch.zeros(B, V, device=device)
                        for k in range(K):
                            # [B, V]
                            current_beam_score = current_score[:, k]
                            if k > 0:
                                diversity_penalty.scatter_add_(
                                    1, beam_word_id, torch.ones(B, V, device=device))
                                current_beam_score -= diversity_lambda * diversity_penalty
                            # [B, 1]
                            beam_word_id = current_beam_score.argmax(dim=1, keepdim=True)

                    # [B, K] => [B, K, 1] => [B, K, V]
                    score = score.view(B, K, 1) + current_score

                    # [B, K, K] => [B, K*K]
                    score = score.view(B, -1)

                    # Select top k 'vocab' candidates (range: 0 ~ K*2K-1)
                    # [B, K]
                    topk_score, topk_idx = score.topk(K, dim=1)

                    # [B, K]
                    # topk_idx = current_score_topk_idx.view(B, K * K).gather(1, topk_idx)

                    # [B, K], (0 ~ K-1)
                    topk_beam_idx = topk_idx // V
                    # [B, K], (0 ~ V-1)
                    topk_word_id = topk_idx % V

                    beam.back_pointers.append(topk_beam_idx)  # [B, K]
                    beam.token_ids.append(topk_word_id)  # [B, K]
                    beam.scores.append(topk_score)  # [B, K]

                    # Top-k scored sequence index in each batch (which batch + which beam inside batch)
                    # batch_indices [B, K]
                    #   [[0,         ...     0        ],
                    #    [K,         ...     K        ],
                    #    [K * 2,     ...     K * 2    ],
                    #                ...
                    #    [K * (B-1), ...,    K * (B-1)]
                    batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
                    batch_indices = batch_starting_indices.unsqueeze(1)
                    topk_beam_idx_flat = (batch_indices + topk_beam_idx).flatten()

                    # Prepare next iteration
                    # [B*K, hidden_size]
                    if self.rnn_type == 'GRU':
                        s = s.index_select(0, topk_beam_idx_flat)
                    elif self.rnn_type == 'LSTM':
                        s = (s[0].index_select(0, topk_beam_idx_flat),
                             s[1].index_select(0, topk_beam_idx_flat))
                    attention = attention.index_select(0, topk_beam_idx_flat)
                    coverage = coverage.index_select(0, topk_beam_idx_flat)
                    score = topk_score  # [B, K]
                    where_EOS = topk_word_id == EOS_ID
                    score.masked_fill_(where_EOS, -math.inf)

                    predicted_word_id = topk_word_id.flatten()
                    where_oov = predicted_word_id >= self.vocab_size
                    dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)

                    # [B, K]
                    generated_eos = topk_word_id == EOS_ID
                    if generated_eos.any():
                        # [B]
                        n_finished += generated_eos.long().sum(dim=1)
                        if n_finished.min().item() >= K:
                            break

                else:
                    if decoding_type == 'greedy':
                        # [B]
                        log_p_sampled, predicted_word_id = log_p.max(dim=1)

                    # Hierarchical Neural Story Generation (ACL 2018)
                    elif decoding_type == 'topk_sampling':
                        topk = 10
                        # [B*K, topk]
                        log_p_topk, predicted_word_id_topk = log_p.topk(
                            topk, dim=1)

                        temperature_scaled_score = (log_p_topk / temperature).exp()

                        # [B*K, 1]
                        sampled_idx = temperature_scaled_score.multinomial(1)

                        # [B*K]
                        log_p_sampled = temperature_scaled_score.gather(
                            1, sampled_idx).squeeze(1)
                        predicted_word_id = predicted_word_id_topk.gather(
                            1, sampled_idx).squeeze(1)

                    log_p_sampled.masked_fill_(finished, 0)
                    score += log_p_sampled

                    where_oov = predicted_word_id >= self.vocab_size
                    dec_input_word = predicted_word_id.masked_fill(
                        where_oov, UNK_ID)

                    output_sentence.append(predicted_word_id)

                    # [B]
                    generated_eos = predicted_word_id == EOS_ID
                    if generated_eos.any():
                        finished += generated_eos
                        finished.clamp_(0, 1)

                        if finished.min().item() > 0:
                            break

            # ------ End-of-Decoder! ------#

        if train:
            # [B, max_dec_len, V]
            log_p = torch.stack(out_log_p, dim=1)

            # [B, max_dec_len]
            coverage_loss = torch.stack(coverage_loss_list, dim=1)

            # [B, max_dec_len, V], [B, max_dec_len]
            return log_p, coverage_loss

        else:
            if decoding_type in ['beam', 'diverse_beam']:
                # Top-K sentence / score
                output_sentence, score = beam.backtrack()  # [B, K, max_dec_len]
                # [B, K, max_dec_len], [B, K]
                return output_sentence, score
            else:
                output_sentence = torch.stack(output_sentence, dim=-1).view(B, 1, -1)
                # [B, 1, max_dec_len], [B, 1]
                return output_sentence, score
