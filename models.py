"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn

from layers.encoder import FocusedEncoder
from layers.decoder import NQGDecoder, PGDecoder
from layers.selector import ParallelSelector
from layers.bridge import LinearBridge


class Model(nn.Module):
    def __init__(self, seq2seq, selector=None):
        super().__init__()
        self.selector = selector
        self.seq2seq = seq2seq


class FocusSelector(nn.Module):
    """Sample focus (sequential binary masks) from source sequence"""

    def __init__(self,
                 word_embed_size: int = 300,
                 answer_position_embed_size: int = 16,
                 ner_embed_size: int = 16,
                 pos_embed_size: int = 16,
                 case_embed_size: int = 16,
                 focus_embed_size: int = 16,
                 enc_hidden_size: int = 300,
                 dec_hidden_size: int = 300,
                 num_layers: int = 1,
                 dropout_p: float = 0.2,
                 rnn: str = 'GRU',
                 n_mixture: int = 1,
                 seq2seq_model: str = 'NQG',
                 task: str = 'QG',
                 threshold: float = 0.15,
                 feature_rich: bool = False):

        super().__init__()

        self.task = task
        self.seq2seq_model = seq2seq_model
        self.feature_rich = feature_rich

        self.selector = ParallelSelector(
            word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size,
            enc_hidden_size, dec_hidden_size,
            num_layers=num_layers, dropout_p=dropout_p, n_mixture=n_mixture, task=task,
            threshold=threshold)

    def add_embedding(self, word_embed, answer_position_embed=None, ner_embed=None, pos_embed=None, case_embed=None):
        if self.feature_rich:
            self.selector.encoder.word_embed = word_embed
            self.selector.encoder.answer_position_embed = answer_position_embed
            self.selector.encoder.ner_embed = ner_embed
            self.selector.encoder.pos_embed = pos_embed
            self.selector.encoder.case_embed = case_embed

        else:
            self.selector.encoder.word_embed = word_embed

    def forward(self,
                source_WORD_encoding,
                answer_position_BIO_encoding=None,
                ner_encoding=None,
                pos_encoding=None,
                case_encoding=None,
                focus_POS_prob=None,
                mixture_id=None,
                focus_input=None,
                train=True,
                max_decoding_len=30):

        out = self.selector(
            source_WORD_encoding,
            answer_position_BIO_encoding,
            ner_encoding,
            pos_encoding,
            case_encoding,
            mixture_id,
            focus_input,
            train,
            max_decoding_len)

        if train:
            focus_logit = out
            return focus_logit

        else:
            generated_focus_mask = out
            return generated_focus_mask


class Seq2Seq(nn.Module):
    def __init__(self,
                 vocab_size: int = 20000,
                 word_embed_size: int = 300,
                 answer_position_embed_size: int = 16,
                 ner_embed_size: int = 16,
                 pos_embed_size: int = 16,
                 case_embed_size: int = 16,
                 position_embed_size: int = 16,
                 focus_embed_size: int = 16,
                 enc_hidden_size: int = 512,
                 dec_hidden_size: int = 256,
                 num_layers: int = 1,
                 dropout_p: float = 0.5,
                 tie: bool = False,
                 rnn: str = 'GRU',
                 use_focus: bool = False,
                 task: str = 'QG',
                 model: str = 'NQG',
                 feature_rich: bool = False,
                 n_mixture=None):
        """Neural Question Generation from Text: A Preliminary Study (Zhou et al. NLPCC 2017)
        Get To The Point: Summarization with Pointer-Generator Networks (See et al. ACL 2017)
        """
        super().__init__()

        # ------ Hyperparameters ------#
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.vocab_size = vocab_size
        self.tie = tie
        self.rnn_type = rnn
        self.use_focus = use_focus
        self.n_mixture = n_mixture
        self.model = model
        self.task = task
        self.feature_rich = feature_rich

        if model == 'NQG':
            assert feature_rich == True

        # ------ Modules ------#
        # [0,1,2,3] => [PAD, UNK, SOS, EOS]
        self.word_embed = nn.Embedding(
            vocab_size, word_embed_size, padding_idx=0)

        self.encoder = FocusedEncoder(
            word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size,
            focus_embed_size, enc_hidden_size, num_layers, dropout_p,
            rnn_type=self.rnn_type, use_focus=use_focus, model=model, feature_rich=feature_rich)
        self.encoder.word_embed = self.word_embed

        if feature_rich:
            self.answer_position_embed = nn.Embedding(
                4, answer_position_embed_size, padding_idx=-1)  # BIO encoding
            self.ner_embed = nn.Embedding(13, ner_embed_size, padding_idx=-1)
            self.pos_embed = nn.Embedding(46, pos_embed_size, padding_idx=-1)
            self.case_embed = nn.Embedding(
                3, case_embed_size, padding_idx=-1)  # Binary

            self.encoder.answer_position_embed = self.answer_position_embed
            self.encoder.ner_embed = self.ner_embed
            self.encoder.pos_embed = self.pos_embed
            self.encoder.case_embed = self.case_embed
        else:
            self.answer_position_embed = None
            self.ner_embed = None
            self.pos_embed = None
            self.case_embed = None

        if use_focus:
            self.focus_embed = nn.Embedding(
                3, focus_embed_size, padding_idx=-1)  # Binary
            self.encoder.focus_embed = self.focus_embed

        # for using encoder's hidden state as decoder input
        if model == 'NQG':
            self.bridge = LinearBridge(
                enc_hidden_size, dec_hidden_size, rnn, 'tanh')
        elif model == 'PG':
            self.bridge = LinearBridge(
                enc_hidden_size, dec_hidden_size, rnn, 'ReLU')

        if model == 'NQG':
            self.decoder = NQGDecoder(word_embed_size,
                                      enc_hidden_size=enc_hidden_size,
                                      dec_hidden_size=dec_hidden_size,
                                      attention_size=700,
                                      # attention_size=dec_hidden_size,
                                      vocab_size=vocab_size,
                                      dropout_p=dropout_p,
                                      rnn=rnn, tie=tie, n_mixture=n_mixture)
        elif model == 'PG':
            self.decoder = PGDecoder(word_embed_size,
                                     enc_hidden_size=enc_hidden_size,
                                     dec_hidden_size=dec_hidden_size,
                                     # attention_size=dec_hidden_size,
                                     vocab_size=vocab_size,
                                     dropout_p=0,
                                     rnn=rnn, tie=tie, n_mixture=n_mixture)

        self.decoder.word_embed = self.word_embed
        if tie:
            self.decoder.readout.Wo.weight = self.word_embed.weight

    def forward(self,
                source_WORD_encoding,
                answer_WORD_encoding=None,
                answer_position_BIO_encoding=None,
                ner_encoding=None,
                pos_encoding=None,
                case_encoding=None,
                focus_mask=None,
                mixture_id=None,
                target_WORD_encoding=None,
                source_WORD_encoding_extended=None,
                train=True,
                decoding_type='beam',
                beam_k=12,
                max_dec_len=30,
                temperature=1.0,
                diversity_lambda=0.5):

        enc_outputs, h = self.encoder(
            source_WORD_encoding,
            answer_position_BIO_encoding=answer_position_BIO_encoding,
            ner_encoding=ner_encoding,
            pos_encoding=pos_encoding,
            case_encoding=case_encoding,
            focus_mask=focus_mask)

        B = enc_outputs.size(0)

        s = self.bridge(h)

        if self.n_mixture:
            # Proceed with all mixtures
            if mixture_id is None:
                enc_outputs = repeat(enc_outputs, self.n_mixture)

                if self.rnn_type == 'GRU':
                    s = repeat(s, self.n_mixture)
                    device = s.device
                elif self.rnn_type == 'LSTM':
                    s = (repeat(s[0], self.n_mixture),
                         repeat(s[1], self.n_mixture))
                    device = s[0].device

                source_WORD_encoding = repeat(
                    source_WORD_encoding, self.n_mixture)
                if self.model == 'PG':
                    source_WORD_encoding_extended = repeat(
                        source_WORD_encoding_extended, self.n_mixture)
                mixture_id = torch.arange(self.n_mixture, dtype=torch.long,
                                          device=device).unsqueeze(0).repeat(B, 1).flatten()
                if train:
                    target_WORD_encoding = repeat(
                        target_WORD_encoding, self.n_mixture)
            else:
                assert mixture_id.size(0) == B

        dec_output = self.decoder(
            enc_outputs,
            s,
            source_WORD_encoding,
            answer_WORD_encoding=answer_WORD_encoding,
            mixture_id=mixture_id,
            target_WORD_encoding=target_WORD_encoding,
            source_WORD_encoding_extended=source_WORD_encoding_extended,
            train=train,
            decoding_type=decoding_type,
            K=beam_k,
            max_dec_len=max_dec_len,
            temperature=temperature,
            diversity_lambda=diversity_lambda)

        if train:
            return dec_output

        else:
            output_sentence, score = dec_output
            return output_sentence, score


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
        tensor = tensor.unsqueeze(1).expand(
            *expand_size).contiguous().view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out
