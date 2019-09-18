import tempfile
import os
import shutil
from multiprocessing import Pool
from itertools import chain
import numpy as np
from tqdm import tqdm

from pyrouge import Rouge155

rouge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RELEASE-1.5.5/')
rouge_path = os.path.join(rouge_dir, 'ROUGE-1.5.5.pl')
assert os.path.exists(rouge_path)
os.system('chmod a+x ' + rouge_path)
_ = Rouge155(rouge_dir, log_level=0)


def split_list(lst, n):
    splitted = []
    for i in reversed(range(1, n + 1)):
        split_point = len(lst) // i
        splitted.append(lst[:split_point])
        lst = lst[split_point:]
    return splitted


def run_rouge(summaries_references):
    """Run Perl ROUGE 1.5.5 script"""

    summaries, references = summaries_references

    temp_dir = tempfile.mkdtemp()
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)

    for i, (summary, ref_candidates) in enumerate(zip(summaries, references)):

        for j, ref_candidate in enumerate(ref_candidates):
            with open(os.path.join(model_dir, f'{i}.{j}.txt'), encoding='utf-8', mode='w') as f:
                f.write('\n'.join(ref_candidate))

        with open(os.path.join(system_dir, f'{i}.txt'), encoding='utf-8', mode='w') as f:
            f.write('\n'.join(summary))

    rouge_args = ['-e', os.path.join(rouge_dir, "data"),
                  '-a',
                  '-c', 95,
                  '-m',
                  '-n', 2,
                  # '-w', 1.2,
                  ]

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_dir=rouge_dir, rouge_args=args_str, log_level=0)
    # rouge = Rouge155(rouge_args=args_str, log_level=0)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'
    output = rouge.convert_and_evaluate()

    output_dict = rouge.output_to_dict(output)

    output_dict = {
        'ROUGE-1': output_dict['rouge_1_f_score'],
        'ROUGE-2': output_dict['rouge_2_f_score'],
        'ROUGE-L': output_dict['rouge_l_f_score'],
        'len': len(summaries)
    }

    # remove the created temporary files
    shutil.rmtree(temp_dir)

    return output_dict


def corpus_rouge(summaries, references, n_process=4):

    assert len(summaries) == len(references)

    # summaries =  [[sentence]]
    # references = [[[sentence]]]
    references = [[r] for r in references]  # [[[sentence]]]

    if n_process > len(summaries):
        n_process = len(summaries)

    split_summaries = split_list(summaries, n_process)
    split_references = split_list(references, n_process)
    split_len = [len(s) for s in split_summaries]
    assert min(split_len) > 0
    # assert sum(split_len) == len(summaries)

    with Pool(n_process) as pool:
        output_dicts = list(tqdm(pool.imap(run_rouge, zip(split_summaries, split_references)),
                                 total=len(split_summaries)))

    final_output_dict = {
        'ROUGE-1': np.average(
            [o_d['ROUGE-1'] for o_d in output_dicts],
            weights=[o_d['len'] for o_d in output_dicts]),
        'ROUGE-2': np.average(
            [o_d['ROUGE-2'] for o_d in output_dicts],
            weights=[o_d['len'] for o_d in output_dicts]),
        'ROUGE-L': np.average(
            [o_d['ROUGE-L'] for o_d in output_dicts],
            weights=[o_d['len'] for o_d in output_dicts]),
    }

    return final_output_dict


def get_sent_rouge(summary_reference):
    summary, reference = summary_reference
    output_dict = run_rouge(([summary], [reference]))
    return output_dict
# import sys
# from pathlib import Path
# rouge_lib_dir = str(Path(__file__).parent.resolve())
# sys.path.append(rouge_lib_dir)
# from google_rouge.rouge_scorer import RougeScorer
# google_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#
#
# def get_sent_rouge(summary_reference):
#     summary, reference = summary_reference
#
#     rouge_1_list = []
#     rouge_2_list = []
#     rouge_l_list = []
#     for s in summary:
#         temp_rouge_1_list = []
#         temp_rouge_2_list = []
#         temp_rouge_l_list = []
#         for r in reference:
#             output_dict = google_scorer.score(r, s)
#             temp_rouge_1_list.append(output_dict['rouge1'].fmeasure)
#             temp_rouge_2_list.append(output_dict['rouge2'].fmeasure)
#             temp_rouge_l_list.append(output_dict['rougeL'].fmeasure)
#         rouge_1_list.append(max(temp_rouge_1_list))
#         rouge_2_list.append(max(temp_rouge_2_list))
#         rouge_l_list.append(max(temp_rouge_l_list))
#
#     output_dict = {
#         'ROUGE-1': np.mean(rouge_1_list),
#         'ROUGE-2': np.mean(rouge_2_list),
#         'ROUGE-L': np.mean(rouge_l_list)
#     }
#     return output_dict


def get_sent_rouge_list(summaries, references, n_process=4):
    assert len(summaries) == len(references)

    # summaries =  [[sentence]]
    # references = [[[sentence]]]
    references = [[r] for r in references]  # [[[sentence]]]

    if n_process > len(summaries):
        n_process = len(summaries)

    with Pool(n_process) as pool:
        output_dicts = list(tqdm(pool.imap(get_sent_rouge,
                                           zip(summaries, references)),
                                 total=len(summaries)))

    return output_dicts


def argmax_rouge(rouge_dicts):
    rouge_1_list = [r_d['ROUGE-1'] for r_d in rouge_dicts]
    rouge_2_list = [r_d['ROUGE-2'] for r_d in rouge_dicts]
    rouge_l_list = [r_d['ROUGE-L'] for r_d in rouge_dicts]
    return np.argmax(rouge_1_list), np.argmax(rouge_2_list), np.argmax(rouge_l_list)


def oracle_rouge(list_of_summaries, references, n_process=4):
    all_sum_sentence_rouge_list = []
    for summaries in list_of_summaries:
        sent_rouge_list = get_sent_rouge_list(summaries, references, n_process=n_process)
        all_sum_sentence_rouge_list.append(sent_rouge_list)

    best_rouge_1_summaries = []
    best_rouge_2_summaries = []
    best_rouge_l_summaries = []
    for i, rouge_dicts in enumerate(zip(*all_sum_sentence_rouge_list)):
        max_rouge_1_index, max_rouge_2_index, max_rouge_l_index = argmax_rouge(rouge_dicts)
        best_rouge_1_summaries.append(list_of_summaries[max_rouge_1_index][i])
        best_rouge_2_summaries.append(list_of_summaries[max_rouge_2_index][i])
        best_rouge_l_summaries.append(list_of_summaries[max_rouge_l_index][i])

    if n_process > len(summaries):
        n_process = len(summaries)

    oracle_rouge_1 = corpus_rouge(best_rouge_1_summaries, references,
                                  n_process=n_process)['ROUGE-1']
    oracle_rouge_2 = corpus_rouge(best_rouge_2_summaries, references,
                                  n_process=n_process)['ROUGE-2']
    oracle_rouge_l = corpus_rouge(best_rouge_l_summaries, references,
                                  n_process=n_process)['ROUGE-L']

    return {
        'ROUGE-1': oracle_rouge_1,
        'ROUGE-2': oracle_rouge_2,
        'ROUGE-L': oracle_rouge_l
    }


def flatten(list_of_lists):
    return list(chain(*list_of_lists))


def avg_rouge(list_of_summaries, references, n_process=4):
    return corpus_rouge(flatten(list_of_summaries),
                        references * len(list_of_summaries),
                        n_process=n_process)


def get_self_rouge_dict(hyps):
    sent_rouge_1_list = []
    sent_rouge_2_list = []
    sent_rouge_l_list = []

    for i in range(len(hyps)):
        h = hyps[i]
        r_list = hyps[:i] + hyps[i + 1:]
        # rouge_dict = run_rouge(([h], [r]))
        # for r in r_list:
        rouge_dict = get_sent_rouge((h, r_list))
        sent_rouge_1_list.append(rouge_dict['ROUGE-1'])
        sent_rouge_2_list.append(rouge_dict['ROUGE-2'])
        sent_rouge_l_list.append(rouge_dict['ROUGE-L'])

    self_rouge_1 = np.mean(sent_rouge_1_list)
    self_rouge_2 = np.mean(sent_rouge_2_list)
    self_rouge_l = np.mean(sent_rouge_l_list)

    return {
        'ROUGE-1': self_rouge_1,
        'ROUGE-2': self_rouge_2,
        'ROUGE-L': self_rouge_l
    }


def self_rouge(list_of_summaries, n_process=4):

    with Pool(processes=n_process) as pool:
        self_rouge_dict_list = list(tqdm(pool.imap(get_self_rouge_dict,
                                                   zip(*list_of_summaries)),
                                         total=len(list_of_summaries)))

    self_rouge_1_list = []
    self_rouge_2_list = []
    self_rouge_l_list = []

    for rouge_dict in self_rouge_dict_list:
        self_rouge_1_list.append(rouge_dict['ROUGE-1'])
        self_rouge_2_list.append(rouge_dict['ROUGE-2'])
        self_rouge_l_list.append(rouge_dict['ROUGE-L'])

    return {
        'ROUGE-1': np.mean(self_rouge_1_list),
        'ROUGE-2': np.mean(self_rouge_2_list),
        'ROUGE-L': np.mean(self_rouge_l_list)
    }


if __name__ == '__main__':

    def split_sentences(words):
        # First, divide decoded output into sentences
        sents = []
        while len(words) > 0:
            try:
                fst_period_idx = words.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(words)
            sent = words[:fst_period_idx + 1]  # sentence up to and including the period
            words = words[fst_period_idx + 1:]  # everything else
            sents.append(' '.join(sent))
        return sents

    import re

    def split_tagged_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
        bare_sents = re.findall(r'%s (.+?) %s' % (sentence_start_tag, sentence_end_tag), article)
        return bare_sents

    summaries_1 = [
        'the dog was found . It was under the bed .'.split(),
        'the dog was found under bed .'.split(),
        'the dog was not found .'.split(),
        'the dog was finally found . It is dangerous dog .'.split(),
    ]

    summaries_2 = [
        'cat was found . It was under the bed'.split(),
        'cat was found under bed. It is still missing .'.split(),
        'where is the cat .'.split(),
        'Cat is on the bed . It is sleeping .'.split()
    ]

    summaries_3 = [
        'the dragon was found under the bed .'.split(),
        'the dragon is found under bed .'.split(),
        'Dragon is here .'.split(),
        'Dragon is dangerous .'.split()
    ]

    references = [
        '<t> cat was found . </t> <t> It is now under the bed . </t>',
        '<t> the dog is not found . </t> <t> It is still missing . </t>',
        '<t> We found the dog . </t> <t> It is in the kitchen . </t>',
        '<t> dog was found . </t> <t> It is quite dangerous . </t>'
    ]

    summaries_1 = [split_sentences(words) for words in summaries_1]
    summaries_2 = [split_sentences(words) for words in summaries_2]
    summaries_3 = [split_sentences(words) for words in summaries_3]
    list_of_summaries = [summaries_1, summaries_2, summaries_3]
    references = [split_tagged_sentences(ref) for ref in references]

    print('ROUGE with Summaries 1 (n_process = 2)')
    print(corpus_rouge(summaries_1, references, n_process=2))
    print('\nROUGE with Summaries 2')
    print(corpus_rouge(summaries_2, references))
    print('\nROUGE with Summaries 3')
    print(corpus_rouge(summaries_3, references))

    print('\nOracle ROUGE')
    print(oracle_rouge(list_of_summaries, references))

    print('\nAverage ROUGE')
    print(avg_rouge(list_of_summaries, references))

    print('\nSelf ROUGE')
    print(self_rouge(list_of_summaries, n_process=2))
