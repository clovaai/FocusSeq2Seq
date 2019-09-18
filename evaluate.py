"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import time
import multiprocessing

import numpy as np
import torch

from utils import bleu, rouge
from utils.tensor_utils import repeat
from utils.data_utils import split_sentences

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

n_cpus = multiprocessing.cpu_count()


def evaluate(loader, model, epoch, config, test=False):
    start = time.time()
    print('Evaluation start!')
    model.eval()
    if config.task == 'QG':
        references = loader.dataset.df.target_WORD.tolist()

    elif config.task == 'SM':
        # references = loader.dataset.df.target_tagged.tolist()
        references = loader.dataset.df.target_multiref.tolist()
        # references = loader.dataset.df.target.tolist()
    hypotheses = [[] for _ in range(max(config.n_mixture, config.decode_k))]
    hyp_focus = [[] for _ in range(max(config.n_mixture, config.decode_k))]
    hyp_attention = [[] for _ in range(max(config.n_mixture, config.decode_k))]

    if config.n_mixture > 1:
        assert config.decode_k == 1
        use_multiple_hypotheses = True
        best_hypothesis = []
    elif config.decode_k > 1:
        assert config.n_mixture == 1
        use_multiple_hypotheses = True
        best_hypothesis = []
    else:
        use_multiple_hypotheses = False
        best_hypothesis = None

    word2id = model.word2id
    id2word = model.id2word

    # PAD_ID = word2id['<pad>']
    vocab_size = len(word2id)

    n_iter = len(loader)
    temp_time_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if config.task == 'QG':
                source_WORD_encoding, source_len, \
                target_WORD_encoding, target_len, \
                source_WORD, target_WORD, \
                answer_position_BIO_encoding, answer_WORD, \
                ner, ner_encoding, \
                pos, pos_encoding, \
                case, case_encoding, \
                focus_WORD, focus_mask, \
                focus_input, answer_WORD_encoding, \
                source_WORD_encoding_extended, oovs \
                    = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            elif config.task == 'SM':
                source_WORD_encoding, source_len, \
                target_WORD_encoding, target_len, \
                source_WORD, target_WORD, \
                focus_WORD, focus_mask, \
                focus_input, \
                source_WORD_encoding_extended, oovs \
                    = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                answer_position_BIO_encoding = answer_WORD = ner_encoding = pos_encoding = case_encoding = None
                answer_WORD_encoding = None

            B, L = source_WORD_encoding.size()

            if config.use_focus:
                if config.eval_focus_oracle:

                    generated_focus_mask = focus_mask
                    input_mask = focus_mask

                else:
                    # [B * n_mixture, L]
                    focus_p = model.selector(
                        source_WORD_encoding,
                        answer_position_BIO_encoding,
                        ner_encoding,
                        pos_encoding,
                        case_encoding,
                        # mixture_id=mixture_id,
                        # focus_input=focus_input,
                        train=False)

                    generated_focus_mask = (focus_p > config.threshold).long()

                    # Repeat for Focus Selector
                    if config.n_mixture > 1:
                        source_WORD_encoding = repeat(
                            source_WORD_encoding, config.n_mixture)
                        if config.feature_rich:
                            answer_position_BIO_encoding = repeat(
                                answer_position_BIO_encoding, config.n_mixture)
                            ner_encoding = repeat(ner_encoding, config.n_mixture)
                            pos_encoding = repeat(pos_encoding, config.n_mixture)
                            case_encoding = repeat(case_encoding, config.n_mixture)
                        if config.model == 'PG':
                            source_WORD_encoding_extended = repeat(
                                source_WORD_encoding_extended, config.n_mixture)
                            assert source_WORD_encoding.size(0) \
                                   == source_WORD_encoding_extended.size(0)

                    input_mask = generated_focus_mask

            else:
                input_mask = None
                generated_focus_mask = focus_mask

            # [B*n_mixturre, K, max_len]
            prediction, score = model.seq2seq(
                source_WORD_encoding,
                answer_WORD_encoding=answer_WORD_encoding,
                answer_position_BIO_encoding=answer_position_BIO_encoding,
                ner_encoding=ner_encoding,
                pos_encoding=pos_encoding,
                case_encoding=case_encoding,
                focus_mask=input_mask,
                target_WORD_encoding=None,
                source_WORD_encoding_extended=source_WORD_encoding_extended,
                train=False,
                decoding_type=config.decoding,
                beam_k=config.beam_k,
                max_dec_len=30 if config.task == 'QG' else 120 if config.task == 'SM' else exit(),
                temperature=config.temperature,
                diversity_lambda=config.diversity_lambda)

            prediction = prediction.view(B, config.n_mixture, config.beam_k, -1)
            prediction = prediction[:, :, 0:config.decode_k, :].tolist()

            if use_multiple_hypotheses:
                score = score.view(B, config.n_mixture, config.beam_k)
                score = score[:, :, :config.decode_k].view(B, -1)
                # [B]
                best_hyp_idx = score.argmax(dim=1).tolist()

            # Word IDs => Words
            for batch_j, (predicted_word_ids, source_words, target_words) \
                in enumerate(zip(prediction, source_WORD, target_WORD)):
                if config.n_mixture > 1:
                    assert config.decode_k == 1
                    for n in range(config.n_mixture):
                        predicted_words = []
                        # [n_mixture, decode_k=1, dec_len]
                        for word_id in predicted_word_ids[n][0]:
                            # Generate
                            if word_id < vocab_size:
                                word = id2word[word_id]
                                # End of sequence
                                if word == '<eos>':
                                    break
                            # Copy
                            else:
                                pointer_idx = word_id - vocab_size
                                if config.model == 'NQG':
                                    word = source_words[pointer_idx]
                                elif config.model == 'PG':
                                    try:
                                        word = oovs[batch_j][pointer_idx]
                                    except IndexError:
                                        import ipdb
                                        ipdb.set_trace()
                            predicted_words.append(word)

                        hypotheses[n].append(predicted_words)

                        if use_multiple_hypotheses and best_hyp_idx[batch_j] == n:
                            best_hypothesis.append(predicted_words)

                elif config.n_mixture == 1:
                    for k in range(config.decode_k):
                        predicted_words = []
                        # [n_mixture=1, decode_k, dec_len]
                        for word_id in predicted_word_ids[0][k]:
                            # Generate
                            if word_id < vocab_size:
                                word = id2word[word_id]
                                # End of sequence
                                if word == '<eos>':
                                    break
                            # Copy
                            else:
                                pointer_idx = word_id - vocab_size
                                if config.model == 'NQG':
                                    word = source_words[pointer_idx]
                                elif config.model == 'PG':
                                    try:
                                        word = oovs[batch_j][pointer_idx]
                                    except IndexError:
                                        import ipdb
                                        ipdb.set_trace()
                            predicted_words.append(word)

                        hypotheses[k].append(predicted_words)

                        if use_multiple_hypotheses and best_hyp_idx[batch_j] == k:
                            best_hypothesis.append(predicted_words)

            # For visualization
            if config.use_focus:
                # [B * n_mixture, L] => [B, n_mixture, L]
                focus_p = focus_p.view(B, config.n_mixture, L)
                generated_focus_mask = generated_focus_mask.view(B, config.n_mixture, L)
                # target_L x [B * n_mixture, L]
                # => [B * n_mixture, L, target_L]
                # => [B, n_mixture, L, target_L]
                attention_list = torch.stack(model.seq2seq.decoder.attention_list, dim=2).view(
                    B, config.n_mixture, L, -1)

                # n_mixture * [B, L]
                for n, focus_n in enumerate(focus_p.split(1, dim=1)):
                    # [B, 1, L] => [B, L]
                    focus_n = focus_n.squeeze(1).tolist()
                    # B x [L]
                    for f_n in focus_n:
                        hyp_focus[n].append(f_n)  # [L]

                # n_mixture * [B, L, target_L]
                for n, attention in enumerate(attention_list.split(1, dim=1)):
                    # [B, 1, L, target_L] => [B, L, target_L]
                    attention = attention.squeeze(1).tolist()
                    # B x [L, target_L]
                    for at in attention:
                        hyp_attention[n].append(np.array(at))  # [L, target_L]

            if (not test) and batch_idx == 0:
                # if batch_idx > 260:
                n_samples_to_print = min(10, len(source_WORD))
                for i in range(n_samples_to_print):
                    s = source_WORD[i]  # [L]
                    g_m = generated_focus_mask[i].tolist()  # [n_mixture, L]

                    f_p = focus_p[i].tolist()  # [n_mixture, L]

                    print(f'[{i}]')

                    print(f"Source Sequence: {' '.join(source_WORD[i])}")
                    if config.task == 'QG':
                        print(f"Answer: {' '.join(answer_WORD[i])}")
                    if config.use_focus:
                        print(f"Oracle Focus: {' '.join(focus_WORD[i])}")
                    if config.task == 'QG':
                        print(f"Target Question: {' '.join(target_WORD[i])}")
                    elif config.task == 'SM':
                        print(f"Target Summary: {' '.join(target_WORD[i])}")
                    if config.n_mixture > 1:
                        for n in range(config.n_mixture):
                            if config.use_focus:
                                print(f'(focus {n})')

                                print(
                                    f"Focus Prob: {' '.join([f'({w}/{p:.2f})' for (w, p) in zip(s, f_p[n])])}")
                                print(
                                    f"Generated Focus: {' '.join([w for w, m in zip(s, g_m[n]) if m == 1])}")
                            if config.task == 'QG':
                                print(
                                    f"Generated Question: {' '.join(hypotheses[n][B * batch_idx + i])}\n")
                            elif config.task == 'SM':
                                print(
                                    f"Generated Summary: {' '.join(hypotheses[n][B * batch_idx + i])}\n")
                    else:
                        for k in range(config.decode_k):
                            if config.use_focus:
                                print(f'(focus {k})')

                                print(
                                    f"Focus Prob: {' '.join([f'({w}/{p:.2f})' for (w, p) in zip(s, f_p[k])])}")
                                print(
                                    f"Generated Focus: {' '.join([w for w, m in zip(s, g_m[k]) if m == 1])}")
                            if config.task == 'QG':
                                print(
                                    f"Generated Question: {' '.join(hypotheses[k][B * batch_idx + i])}\n")
                            elif config.task == 'SM':
                                print(
                                    f"Generated Summary: {' '.join(hypotheses[k][B * batch_idx + i])}\n")

            if batch_idx % 100 == 0 or (batch_idx + 1) == n_iter:
                log_str = f'Evaluation | Epoch [{epoch}/{config.epochs}]'
                log_str += f' | Iteration [{batch_idx}/{n_iter}]'
                time_taken = time.time() - temp_time_start
                log_str += f' | Time taken: : {time_taken:.2f}'
                print(log_str)
                temp_time_start = time.time()

    time_taken = time.time() - start
    print(f"Generation Done! It took {time_taken:.2f}s")

    if test:
        print('Test Set Evaluation Result')

    score_calc_start = time.time()

    if not config.eval_focus_oracle and use_multiple_hypotheses:
        if config.task == 'QG':
            nested_references = [[r] for r in references]
            flat_hypothesis = best_hypothesis
            # bleu_4 = bleu.corpus_bleu(nested_references, flat_hypothesis,
            #                           smoothing_function=bleu.cm.method2) * 100
            bleu_4 = bleu.corpus_bleu(nested_references, flat_hypothesis) * 100
            print(f"BLEU-4: {bleu_4:.3f}")

            oracle_bleu_4 = bleu.oracle_bleu(hypotheses, references,
                                             n_process=min(4, n_cpus)) * 100
            print(f"Oracle BLEU-4: {oracle_bleu_4:.3f}")

            self_bleu = bleu.self_bleu(hypotheses,
                                       n_process=min(4, n_cpus)) * 100
            print(f"Self BLEU-4: {self_bleu:.3f}")

            avg_bleu = bleu.avg_bleu(hypotheses, references) * 100
            print(f"Average BLEU-4: {avg_bleu:.3f}")

            metric_result = {
                'BLEU-4': bleu_4,
                'Oracle_BLEU-4': oracle_bleu_4,
                'Self_BLEU-4': self_bleu,
                'Average_BLEU-4': avg_bleu}
        elif config.task == 'SM':
            flat_hypothesis = best_hypothesis

            # summaries = [split_sentences(remove_tags(words))
            #              for words in flat_hypothesis]
            summaries = [split_sentences(words)
                         for words in flat_hypothesis]
            # references = [split_tagged_sentences(ref) for ref in references]

            # summaries = [[" ".join(words)]
            #              for words in flat_hypothesis]
            # references = [[ref] for ref in references]

            rouge_eval_start = time.time()
            rouge_dict = rouge.corpus_rouge(summaries, references,
                                            n_process=min(4, n_cpus))
            print(f'ROUGE calc time: {time.time() - rouge_eval_start:.3f}s')
            for metric_name, score in rouge_dict.items():
                print(f"{metric_name}: {score * 100:.3f}")

            ##################

            hypotheses_ = [[split_sentences(words) for words in hypothesis]
                           for hypothesis in hypotheses]
            # references = [split_tagged_sentences(ref) for ref in references]
            # hypotheses_ = [[[" ".join(words)] for words in hypothesis]
            #                for hypothesis in hypotheses]
            # references = [[ref] for ref in references]

            oracle_rouge_eval_start = time.time()
            oracle_rouge = rouge.oracle_rouge(hypotheses_, references,
                                              n_process=min(4, n_cpus))
            print(f'Oracle ROUGE calc time: {time.time() - oracle_rouge_eval_start:.3f}s')
            for metric_name, score in oracle_rouge.items():
                print(f"Oracle {metric_name}: {score * 100:.3f}")

            self_rouge_eval_start = time.time()
            self_rouge = rouge.self_rouge(hypotheses_,
                                          n_process=min(4, n_cpus))
            print(f'Self ROUGE calc time: {time.time() - self_rouge_eval_start:.3f}s')
            for metric_name, score in self_rouge.items():
                print(f"Self {metric_name}: {score * 100:.3f}")

            avg_rouge_eval_start = time.time()
            avg_rouge = rouge.avg_rouge(hypotheses_, references,
                                        n_process=min(4, n_cpus))
            print(f'Average ROUGE calc time: {time.time() - avg_rouge_eval_start:.3f}s')
            for metric_name, score in avg_rouge.items():
                print(f"Average {metric_name}: {score * 100:.3f}")

            metric_result = {
                'ROUGE-1': rouge_dict['ROUGE-1'],
                'ROUGE-2': rouge_dict['ROUGE-2'],
                'ROUGE-L': rouge_dict['ROUGE-L'],
                'Oracle_ROUGE-1': oracle_rouge['ROUGE-1'],
                'Oracle_ROUGE-2': oracle_rouge['ROUGE-2'],
                'Oracle_ROUGE-L': oracle_rouge['ROUGE-L'],
                'Self_ROUGE-1': self_rouge['ROUGE-1'],
                'Self_ROUGE-2': self_rouge['ROUGE-2'],
                'Self_ROUGE-L': self_rouge['ROUGE-L'],
                'Average_ROUGE-1': avg_rouge['ROUGE-1'],
                'Average_ROUGE-2': avg_rouge['ROUGE-2'],
                'Average_ROUGE-L': avg_rouge['ROUGE-L'],
            }
            metric_result = {k: v * 100 for k, v in metric_result.items()}

    else:
        if config.task == 'QG':
            nested_references = [[r] for r in references]
            flat_hypothesis = hypotheses[0]
            # bleu_4 = bleu.corpus_bleu(nested_references, flat_hypothesis,
            #                           smoothing_function=bleu.cm.method2) * 100
            bleu_4 = bleu.corpus_bleu(nested_references, flat_hypothesis)
            # print(f"BLEU-4: {100 * bleu_4:.3f}")
            metric_result = {'BLEU-4': bleu_4}

            metric_result = {k: v * 100 for k, v in metric_result.items()}
            for metric_name, score in metric_result.items():
                print(f"{metric_name}: {score:.3f}")

        elif config.task == 'SM':
            flat_hypothesis = hypotheses[0]

            # summaries = [split_sentences(remove_tags(words))
            #              for words in flat_hypothesis]
            summaries = [split_sentences(words)
                         for words in flat_hypothesis]
            # references = [split_tagged_sentences(ref) for ref in references]

            # summaries = [[" ".join(words)]
            #              for words in flat_hypothesis]
            # references = [[ref] for ref in references]

            metric_result = rouge.corpus_rouge(summaries, references,
                                               n_process=min(4, n_cpus))

            metric_result = {k: v * 100 for k, v in metric_result.items()}

            for metric_name, score in metric_result.items():
                print(f"{metric_name}: {score:.3f}")

    score_calc_time_taken = time.time() - score_calc_start
    print(f'Score calculation Done! It took {score_calc_time_taken:.2f}s')

    return metric_result, hypotheses, best_hypothesis, hyp_focus, hyp_attention


if __name__ == '__main__':
    pass
