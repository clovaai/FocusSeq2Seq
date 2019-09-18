"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import torch


class Beam(object):
    def __init__(self, batch_size, beam_size, EOS_ID=3):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.beam_size = beam_size

        self.EOS_ID = EOS_ID

        self.back_pointers = []  # [(B, K)] * sequence_length
        self.token_ids = []  # [(B, K)] * sequence_length
        self.scores = []  # [(B, K)] * sequence_length

    def backtrack(self):
        """Backtracks over batch to generate optimal k-sequences

        back_pointer [B, K]
        token_id [B, K]
        attention [B, K, source_L]

        Returns:
            prediction ([B, K, max_unroll])
                A list of Tensors containing predicted sequence
        """

        B = self.batch_size
        K = self.beam_size
        device = self.token_ids[0].device
        max_unroll = len(self.back_pointers)

        # Sum of score (sorted) [B, K]
        score = self.scores[-1].clone()

        n_eos_found = [0] * B

        # Initialize Back-pointer [B, K]
        back_pointer = torch.arange(0, K).unsqueeze(0).repeat(B, 1).to(device)

        # max_unroll * [B, K]
        prediction = []
        for t in reversed(range(max_unroll)):
            token_id = self.token_ids[t].gather(1, back_pointer)
            back_pointer = self.back_pointers[t].gather(1, back_pointer)

            where_EOS = self.token_ids[t] == self.EOS_ID

            if where_EOS.any():
                for eos_idx in reversed(where_EOS.nonzero().tolist()):
                    batch_idx, beam_idx = eos_idx
                    back_pointer[batch_idx, K - 1 - (n_eos_found[batch_idx] %
                                                     K)] = self.back_pointers[t][batch_idx, beam_idx]
                    token_id[batch_idx, K - 1 - (n_eos_found[batch_idx] %
                                                 K)] = self.token_ids[t][batch_idx, beam_idx]
                    score[batch_idx, K - 1 - (n_eos_found[batch_idx] % K)
                    ] = self.scores[t][batch_idx, beam_idx]
                    n_eos_found[batch_idx] += 1

            prediction.append(token_id)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in the reverse order
        # max_unroll * [B x K]
        prediction = list(reversed(prediction))

        # [B, K, max_unroll]
        prediction = torch.stack(prediction, 2)

        # import ipdb
        # ipdb.set_trace()

        # Re-order (Score orders might have been changed during EOS handling)
        # [B, K]
        score, score_idx = score.topk(K, dim=1)

        batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
        batch_indices = (batch_starting_indices.unsqueeze(1) + score_idx).flatten()
        prediction = prediction.view(B * K, max_unroll).index_select(0,
                                                                     batch_indices).view(B, K, max_unroll)

        return prediction, score
