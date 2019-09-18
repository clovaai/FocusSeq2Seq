"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import pandas as pd
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
from functools import partial
import pickle
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import re


def split_tagged_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' % (sentence_start_tag, sentence_end_tag), article)
    return bare_sents


def read(path):
    line_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            line_list.append(clean_line)
    return line_list


def vocab_read(path, max_vocab_size=50000):
    word2id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    # word2id.update({'<t>': 4, '</t>': 5})
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            id = len(word2id)
            word2id[word] = id
            if len(word2id) == max_vocab_size:
                break
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def load_data(data_dir, split='train'):
    data_dir = Path(data_dir)
    if split == 'train':
        src = read(data_dir.joinpath('train.txt.src'))
        tgt = read(data_dir.joinpath('train.txt.tgt.tagged'))

    elif split == 'val':
        src = read(data_dir.joinpath('val.txt.src'))
        tgt = read(data_dir.joinpath('val.txt.tgt.tagged'))

    elif split == 'test':
        src = read(data_dir.joinpath('test.txt.src'))
        tgt = read(data_dir.joinpath('test.txt.tgt.tagged'))

    return src, tgt


def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])


def make_focus_target(src_split, tgt):
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(src_split):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, src_split)
        if searchstring in tgt and endix < len(src_split) - 1:
            endix += 1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:  # -1:
                matches.extend([False] * (endix - startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix - 1, src_split)
                if matchstrings[full_string] >= 1:
                    matches.extend([False] * (endix - startix))
                else:
                    matches.extend([True] * (endix - startix))
                    matchstrings[full_string] += 1
            startix = endix
    return matches


def preprocess_data(data_dir, split='train', n_process=4, max_len=None):
    print(f'Preprocessing {split} dataset...')
    _src_data, _tgt_data = load_data(data_dir, split)

    assert len(_src_data) == len(_tgt_data)

    total_len = len(_src_data)
    print('# Data:', total_len)

    src_data = []
    tgt_data = []
    for src, tgt in zip(_src_data, _tgt_data):
        if len(src.split()) < 2 or len(tgt.split()) < 2:
            continue
        src_data.append(src)
        tgt_data.append(tgt)

    valid_total_len = len(src_data)
    print('# Valid Data:', valid_total_len)

    if max_len is not None:
        if valid_total_len > max_len:
            print(f'Use only {max_len} samples!')
            src_data = src_data[:max_len]
            tgt_data = tgt_data[:max_len]
            valid_total_len = max_len

    with Pool(n_process) as pool:
        data = list(tqdm(pool.imap(partial(preprocess_single_example,
                                           split=split),
                                   zip(src_data, tgt_data)),
                         total=valid_total_len))

    df = pd.DataFrame(data)
    print(f'Done! size: {len(df)}')
    return df


def preprocess_single_example(src_tgt, split='train'):
    src, tgt = src_tgt

    src_split = src.split()
    # tgt_split = tgt.split()

    source = src
    source_WORD = []
    source_WORD_encoding = []
    source_WORD_encoding_extended = []
    focus_mask = []
    focus_WORD = []
    focus_input = []
    oovs = []

    # target = tgt = split_tagged_sentences(tgt)[0]
    src_focus_annotated = make_focus_target(src_split, tgt)

    assert len(src_split) == len(src_focus_annotated)

    for i, (is_copied, word) in enumerate(zip(src_focus_annotated, src_split)):

        source_WORD.append(word)
        if word in word2id:
            source_WORD_encoding.append(word2id[word])
            source_WORD_encoding_extended.append(word2id[word])
        else:
            source_WORD_encoding.append(word2id['<unk>'])
            if word not in oovs:
                oovs.append(word)
            oov_num = oovs.index(word)
            source_WORD_encoding_extended.append(len(word2id) + oov_num)

        focus_mask.append(int(is_copied))
        if is_copied:
            focus_WORD.append(word)
            focus_input.append(word2id[word] if word in word2id else word2id['<unk>'])

        # if split in ['train', 'val'] and (i + 1) == 400:
        if (i + 1) == 400:
            source = " ".join(source_WORD)
            break

    target = " ".join(tgt.replace('<t>', '').replace('</t>', '').split())
    # target = tgt
    target_WORD = target.split()
    if split in ['train', 'val']:
        target_WORD = target_WORD[:100]
        target = " ".join(target_WORD)

    target_WORD_encoding = []
    for word in target_WORD:
        if word in word2id:
            target_WORD_encoding.append(word2id[word])
        else:
            # can be copied
            if word in oovs:
                target_WORD_encoding.append(len(word2id) + oovs.index(word))
            else:
                target_WORD_encoding.append(word2id['<unk>'])

    example = {
        'source': source,
        'source_WORD': source_WORD,
        'source_WORD_encoding': source_WORD_encoding,
        'source_len': len(source_WORD),
        'source_WORD_encoding_extended': source_WORD_encoding_extended,
        'target': target,
        'target_WORD': target_WORD,
        'target_WORD_encoding': target_WORD_encoding,
        'target_len': len(target_WORD),
        'target_tagged': tgt,
        'target_multiref': split_tagged_sentences(tgt),
        'oovs': oovs,
        'focus_WORD': focus_WORD,
        'focus_mask': focus_mask,
        'focus_input': focus_input
    }
    return example


class CNNDMDataset(Dataset):
    def __init__(self, df, split='train', n_data=None):
        # print(f'Loading {split} dataset...')

        self.df = df
        # self.df = pd.read_pickle(df_path)

        if type(n_data) == int:
            self.df = self.df[:n_data]

        # self.df = df.sort_values('source_len', ascending=False).reset_index()

        # Add <EOS> at the end of target target
        self.df.target_WORD_encoding = self.df.target_WORD_encoding.apply(
            lambda x: x + [3])  # 3: word2id['<eos>']

        # self.df.target_WORD_encoding = self.df.target_WORD_encoding.apply(torch.LongTensor)
        #
        # self.df.source_WORD_encoding = self.df.source_WORD_encoding.apply(torch.LongTensor)
        #
        # self.df.source_WORD_encoding_extended = self.df.source_WORD_encoding_extended.apply(
        #     torch.LongTensor)

        # self.df.focus_mask = self.df.focus_mask.apply(torch.LongTensor)

        # Add unused 0 padding to avoid empty batch
        self.df.focus_input = self.df.focus_input.apply(
            lambda x: [0] + x)
        # self.df.focus_input = self.df.focus_input.apply(torch.LongTensor)

        print(f'Done! Size: {len(self.df)}')

    def __getitem__(self, idx):
        return self.df.ix[idx]

    def __len__(self):
        return len(self.df)


def get_SM_loader(df, n_data_epoch=None, **kwargs):
    dataset = CNNDMDataset(df)

    sampler = None
    if n_data_epoch is not None:
        from torch.utils.data.sampler import RandomSampler
        sampler = RandomSampler(dataset, replacement=True, num_samples=n_data_epoch)
        print(f'Sample {n_data_epoch} examples at every epoch')

    def sm_collate_fn(batch):

        batch = pd.DataFrame(batch).reset_index(drop=True)

        # Add <EOS> at the end of target
        batch.target_WORD_encoding = batch.target_WORD_encoding.apply(
            lambda x: x + [3])  # 3: word2id['<eos>']

        target_WORD_encoding = batch.target_WORD_encoding.apply(torch.LongTensor)
        target_WORD_encoding = pad_sequence(
            target_WORD_encoding, batch_first=True, padding_value=0)

        source_WORD_encoding = batch.source_WORD_encoding.apply(torch.LongTensor)
        source_WORD_encoding = pad_sequence(
            source_WORD_encoding, batch_first=True, padding_value=0)

        source_WORD_encoding_extended = batch.source_WORD_encoding_extended.apply(torch.LongTensor)
        source_WORD_encoding_extended = pad_sequence(
            source_WORD_encoding_extended, batch_first=True, padding_value=0)

        focus_mask = batch.focus_mask.apply(torch.LongTensor)
        focus_mask = pad_sequence(
            focus_mask, batch_first=True, padding_value=0)

        # Add unused 0 padding to avoid empty batch
        focus_input = batch.focus_input.apply(
            lambda x: [0] + x)
        focus_input = batch.focus_input.apply(torch.LongTensor)
        focus_input = pad_sequence(
            focus_input, batch_first=True, padding_value=0)

        # Raw words
        source_WORD = batch.source_WORD.tolist()
        target_WORD = batch.target_WORD.tolist()
        focus_WORD = batch.focus_WORD.tolist()

        source_len = batch.source_len.tolist()
        target_len = batch.target_len.tolist()

        oovs = batch.oovs.tolist()

        return source_WORD_encoding, source_len, \
               target_WORD_encoding, target_len, \
               source_WORD, target_WORD, \
               focus_WORD, focus_mask, \
               focus_input, \
               source_WORD_encoding_extended, oovs

    if sampler is not None:
        return DataLoader(dataset, collate_fn=sm_collate_fn, sampler=sampler, **kwargs)
    else:
        return DataLoader(dataset, collate_fn=sm_collate_fn, **kwargs)


def load_word_vector(vector_path, word2id, dim=300):
    """
    Read pretrained vectors
    Make lookup table with vocabulary
    Load vector at lookup table
    """
    import numpy as np

    vocab_size = len(word2id)
    lookup_table = np.random.normal(size=[vocab_size, dim])

    if 'glove' in str(vector_path):
        n_total_vector = 400000

    n_covered = 0
    with open(vector_path, 'r') as f:
        for line in tqdm(f, total=n_total_vector):
            word, *vector = line.split()
            assert len(vector) == dim
            if word in word2id:
                word_id = word2id[word]
                vector = np.array([float(x) for x in vector])
                lookup_table[word_id] = vector
                n_covered += 1

    print(f'Vocab_size: {vocab_size}')
    print(f'Covered with pretrained vector: {n_covered}')
    print(f'Not covered: {vocab_size - n_covered}')

    return lookup_table


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent

    data_dir = current_dir.joinpath('cnndm/')
    out_dir = current_dir.joinpath('cnndm_out/')
    out_dir.mkdir()

    word2id, id2word = vocab_read(data_dir.joinpath('vocab'))
    with open(out_dir.joinpath('vocab.pkl'), 'wb') as f:
        pickle.dump((word2id, id2word), f)

    train_df = preprocess_data(data_dir, 'train')
    val_df = preprocess_data(data_dir, 'val')
    test_df = preprocess_data(data_dir, 'test')

    # train_loader = get_SM_loader(
    #     train_df,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=1)
    #
    # val_loader = get_SM_loader(
    #     val_df,
    #     batch_size=10,
    #     shuffle=False,
    #     num_workers=1,
    # )

    # test_loader = get_SM_loader(
    #     test_df,
    #     batch_size=10,
    #     shuffle=False,
    #     num_workers=1,
    # )

    train_df.to_pickle(out_dir.joinpath('train_df.pkl'))
    val_df.to_pickle(out_dir.joinpath('val_df.pkl'))
    test_df.to_pickle(out_dir.joinpath('test_df.pkl'))

    # word_vector = load_word_vector(glove_dir.joinpath('glove.6B.100d.txt'), word2id, dim=100)
    # with open(out_dir.joinpath('word_vector.pkl'), 'wb') as f:
    #     pickle.dump(word_vector, f)

    # with open(out_dir.joinpath('vocab.pkl'), 'wb') as f:
    #     pickle.dump((word2id, id2word), f)

    print('Preprocess Done!')
