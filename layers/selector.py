"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUEncoder(nn.Module):
    def __init__(self,
                 word_embed_size=300,
                 answer_position_embed_size=16, ner_embed_size=16, pos_embed_size=16, case_embed_size=16,
                 hidden_size=300, dropout_p=0.2, task='QG'):
        super().__init__()

        self.task = task
        if task == 'QG':
            input_size = word_embed_size + answer_position_embed_size + \
                         ner_embed_size + pos_embed_size + case_embed_size
        else:
            input_size = word_embed_size

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size // 2,
                          batch_first=True,
                          num_layers=1,
                          bidirectional=True)

        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                source_WORD_encoding,
                answer_position_BIO_encoding=None,
                ner_encoding=None,
                pos_encoding=None,
                case_encoding=None,
                PAD_ID=0):

        pad_mask = source_WORD_encoding == PAD_ID

        # ---- Sort by length (decreasing order) ----#
        source_len = (~pad_mask).long().sum(dim=1)  # [B]
        source_len_sorted, idx_sorted = torch.sort(source_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sorted, dim=0)

        if self.task == 'QG':
            source_WORD_encoding = source_WORD_encoding[idx_sorted]
            answer_position_BIO_encoding = answer_position_BIO_encoding[idx_sorted]
            ner_encoding = ner_encoding[idx_sorted]
            pos_encoding = pos_encoding[idx_sorted]
            case_encoding = case_encoding[idx_sorted]

            word_embedding = self.word_embed(source_WORD_encoding)  # [B, L, word_embed_size]
            answer_position_embedding = self.answer_position_embed(
                answer_position_BIO_encoding)  # [B, L, answer_position_embed_size]
            ner_embedding = self.ner_embed(ner_encoding)  # [B, L, ner_embed_size]
            pos_embedding = self.pos_embed(pos_encoding)  # [B, L, pos_embed_size]
            case_embedding = self.case_embed(case_encoding)  # [B, L, case_embed_size]

            enc_input = torch.cat([word_embedding, answer_position_embedding,
                                   ner_embedding, pos_embedding, case_embedding], dim=2)
        else:
            source_WORD_encoding = source_WORD_encoding[idx_sorted]
            word_embedding = self.word_embed(source_WORD_encoding)
            enc_input = word_embedding

        # ---- Input Dropout ----#
        enc_input = self.dropout(enc_input)

        # ---- RNN ----#
        # [B, L, hidden_size]
        enc_input = pack_padded_sequence(enc_input, source_len_sorted, batch_first=True)
        enc_outputs, h = self.rnn(enc_input)
        enc_outputs, _ = pad_packed_sequence(enc_outputs, batch_first=True)

        # ---- Unsort ----#
        enc_outputs = enc_outputs[idx_unsort]
        h = h.index_select(1, idx_unsort)

        return enc_outputs, h


class ParallelDecoder(nn.Module):
    def __init__(self, embed_size=300, enc_hidden_size=512, dec_hidden_size=512,
                 n_mixture=5, threshold=0.15, task='QG'):
        """Parallel Decoder for Focus Selector"""
        super().__init__()

        self.mixture_embedding = nn.Embedding(n_mixture, embed_size)

        # input_size = embed_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.out_mlp = nn.Sequential(
            nn.Linear(enc_hidden_size + dec_hidden_size + embed_size, dec_hidden_size),
            nn.Tanh(),
            nn.Linear(dec_hidden_size, 1),
        )

        self.threshold = threshold

    def forward(self,
                enc_outputs,
                s,
                source_WORD_encoding,
                mixture_id,
                focus_input=None,
                train=True,
                max_decoding_len=None):

        B, max_source_len = source_WORD_encoding.size()

        # [B, embed_size]
        mixture_embedding = self.mixture_embedding(mixture_id)

        # [B, max_source_len, enc_hidden_size + dec_hidden_size + embed_size]
        concat_h = torch.cat([enc_outputs,
                              s.unsqueeze(1).expand(-1, max_source_len, -1),
                              mixture_embedding.unsqueeze(1).expand(-1, max_source_len, -1)], dim=2)

        # [B, max_source_len]
        focus_logit = self.out_mlp(concat_h).squeeze(2)

        if train:
            # [B, max_source_len]
            return focus_logit

        else:
            focus_p = torch.sigmoid(focus_logit)

            # [B, max_source_len]
            return focus_p


class ParallelSelector(nn.Module):

    def __init__(self,
                 word_embed_size=300,
                 answer_position_embed_size=16,
                 ner_embed_size=16,
                 pos_embed_size=16,
                 case_embed_size=16,
                 enc_hidden_size=300,
                 dec_hidden_size=300,
                 num_layers=1,
                 dropout_p=0.2,
                 n_mixture=5,
                 task="QG",
                 threshold=0.15):
        super().__init__()

        self.task = task
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.n_mixture = n_mixture

        self.encoder = GRUEncoder(
            word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size,
            enc_hidden_size, task=task)

        self.dec_init = nn.Sequential(
            nn.Linear(enc_hidden_size, dec_hidden_size),
            nn.LeakyReLU())

        self.decoder = ParallelDecoder(
            word_embed_size, enc_hidden_size, dec_hidden_size,
            n_mixture=n_mixture, task=task, threshold=threshold)

    def forward(self,
                source_WORD_encoding,
                answer_position_BIO_encoding=None,
                ner_encoding=None,
                pos_encoding=None,
                case_encoding=None,
                mixture_id=None,
                focus_input=None,
                train=True,
                max_decoding_len=None):

        enc_outputs, h = self.encoder(
            source_WORD_encoding,
            answer_position_BIO_encoding,
            ner_encoding,
            pos_encoding,
            case_encoding)

        # (num_layers * num_directions, B, hidden_size)
        B = h.size(1)
        h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
        s = self.dec_init(h)  # [B, hidden_size]

        # Proceed with all mixtures
        if mixture_id is None:
            enc_outputs = repeat(enc_outputs, self.n_mixture)
            s = repeat(s, self.n_mixture)
            source_WORD_encoding = repeat(source_WORD_encoding, self.n_mixture)
            focus_input = repeat(focus_input, self.n_mixture)
            mixture_id = torch.arange(self.n_mixture, dtype=torch.long,
                                      device=s.device).unsqueeze(0).repeat(B, 1).flatten()
        else:
            assert mixture_id.size(0) == B

        dec_output = self.decoder(
            enc_outputs,
            s,
            source_WORD_encoding,
            mixture_id,
            focus_input,
            train,
            max_decoding_len=None)

        if train:
            focus_logit = dec_output
            return focus_logit

        else:
            focus_p = dec_output
            return focus_p


def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]

    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        # repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
        # tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
        expand_size = B, K, *size
        tensor = tensor.unsqueeze(1).expand(*expand_size).contiguous().view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out
