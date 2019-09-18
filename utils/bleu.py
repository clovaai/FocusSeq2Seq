from multiprocessing import Pool
from itertools import chain

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm


cm = SmoothingFunction()


def flatten(list_of_lists):
    return list(chain(*list_of_lists))


def get_sent_bleu(h_r):
    h, r = h_r
    # smoothing method: Chin-Yew Lin and Franz Josef Och (COLING 2004)
    return sentence_bleu([r], h, smoothing_function=cm.method2)


def get_sent_bleu_list(hyp, ref, n_process=4):
    assert len(hyp) == len(ref)

    if n_process > len(hyp):
        n_process = len(hyp)

    with Pool(n_process) as pool:
        sent_bleu_list = list(tqdm(pool.imap(get_sent_bleu, zip(hyp, ref)),
                                   total=len(hyp)
                                   ))

    return sent_bleu_list


def oracle_bleu(hyp_list, ref, n_process=4):

    assert len(set([len(h) for h in hyp_list])) == 1

    all_hyp_sentence_bleu_list = [get_sent_bleu_list(hyp, ref, n_process=n_process)
                                  for hyp in hyp_list]

    if n_process > len(hyp_list[0]):
        n_process = len(hyp_list[0])

    with Pool(n_process) as pool:
        max_hyp_index_list = list(tqdm(pool.imap(np.argmax, zip(*all_hyp_sentence_bleu_list)),
                                       total=len(all_hyp_sentence_bleu_list)))

    best_hyp_list = []
    for i, max_hyp_index in enumerate(max_hyp_index_list):
        best_hyp = hyp_list[max_hyp_index][i]
        best_hyp_list.append(best_hyp)

    return corpus_bleu([[r] for r in ref], best_hyp_list, smoothing_function=cm.method2)


def avg_bleu(hyp_list, ref):
    return corpus_bleu([[r] for r in ref * len(hyp_list)], flatten(hyp_list), smoothing_function=cm.method2)


def get_self_bleu(hyps):
    sent_bleu_list = []
    for i in range(len(hyps)):
        h = hyps[i]
        r = hyps[:i] + hyps[i + 1:]
        sent_bleu_list.append(sentence_bleu(r, h, smoothing_function=cm.method2))
    return np.mean(sent_bleu_list)


def self_bleu(hyp_list, n_process=4):

    assert len(set([len(h) for h in hyp_list])) == 1

    if n_process > len(hyp_list):
        n_process = len(hyp_list)

    with Pool(n_process) as pool:
        self_bleu_list = list(tqdm(pool.imap(get_self_bleu, zip(*hyp_list)),
                                   total=len(hyp_list)))

    return np.mean(self_bleu_list)


if __name__ == '__main__':

    hyp1 = [['the', 'dog', 'is', 'found', 'under', 'the', 'big', 'funny', 'bed'],
            ['dog', 'was', 'found', 'under', 'bed']]
    hyp2 = [['the', 'cat', 'was', 'found', 'under', 'the', 'big', 'funny', 'bed'],
            ['cat', 'was', 'found', 'under', 'bed']]
    hyp3 = [['the', 'dragon', 'was', 'found', 'under', 'the', 'big', 'funny', 'bed'],
            ['cat', 'was', 'found', 'under', 'house']]

    ref = [['that', 'cat', 'was', 'under', 'the', 'bed', '.'],
           ['this', 'bat', 'was', 'under', 'the', 'bed', '.']]

    print('Corpus BLEU with hyp 1')
    print(corpus_bleu([[r] for r in ref], hyp1, smoothing_function=cm.method2))
    print('Corpus BLEU with hyp 2')
    print(corpus_bleu([[r] for r in ref], hyp2, smoothing_function=cm.method2))
    print('Corpus BLEU with hyp 3')
    print(corpus_bleu([[r] for r in ref], hyp3, smoothing_function=cm.method2))

    print('\nOracle BLEU - Manual')
    print(corpus_bleu([[r] for r in ref], hyp2, smoothing_function=cm.method2))
    print('Oracle BLEU')
    print(oracle_bleu([hyp2], ref))

    # print(np.mean([sentence_bleu([hyp1[0], hyp2[0]], hyp3[0], smoothing_function=cm.method2),
    #                sentence_bleu([hyp2[0], hyp3[0]], hyp1[0], smoothing_function=cm.method2),
    #                sentence_bleu([hyp3[0], hyp1[0]], hyp2[0], smoothing_function=cm.method2)]))
    # print(np.mean([sentence_bleu([hyp1[1], hyp2[1]], hyp3[1], smoothing_function=cm.method2),
    #                sentence_bleu([hyp2[1], hyp3[1]], hyp1[1], smoothing_function=cm.method2),
    #                sentence_bleu([hyp3[1], hyp1[1]], hyp2[1], smoothing_function=cm.method2)]))
    print('\nSELF BLEU - Manual')
    print(np.mean([np.mean([sentence_bleu([hyp1[0], hyp2[0]], hyp3[0], smoothing_function=cm.method2),
                            sentence_bleu([hyp2[0], hyp3[0]], hyp1[0],
                                          smoothing_function=cm.method2),
                            sentence_bleu([hyp3[0], hyp1[0]], hyp2[0], smoothing_function=cm.method2)]),
                   np.mean([sentence_bleu([hyp1[1], hyp2[1]], hyp3[1], smoothing_function=cm.method2),
                            sentence_bleu([hyp2[1], hyp3[1]], hyp1[1],
                                          smoothing_function=cm.method2),
                            sentence_bleu([hyp3[1], hyp1[1]], hyp2[1], smoothing_function=cm.method2)])]))
    print('SELF BLEU')
    print(self_bleu([hyp1, hyp2, hyp3]))

    print('\nAverage BLEU')
    print(avg_bleu([hyp1, hyp2, hyp3], ref))
