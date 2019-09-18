"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Load NLTK Porter Stemmer / stopwords
# Download if uninstalled
try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    stopword_set = set(stopwords.words('english'))
    stopword_set.add(',')
    porter_stemmer = PorterStemmer()
except LookupError:
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    stopword_set = set(stopwords.words('english'))
    stopword_set.add(',')
    porter_stemmer = PorterStemmer()

# Linguistic Features (for NQG++)
# POS / NER / Word Case / Answer Position
pos2id = {'#': 41,
          '$': 35,
          "''": 20,
          ',': 6,
          '-LRB-': 25,
          '-RRB-': 28,
          '.': 12,
          ':': 29,
          'CC': 13,
          'CD': 11,
          'DT': 2,
          'EX': 36,
          'FW': 38,
          'IN': 4,
          'JJ': 18,
          'JJR': 30,
          'JJS': 23,
          'LS': 44,
          'MD': 33,
          'NN': 3,
          'NNP': 5,
          'NNPS': 19,
          'NNS': 15,
          'PDT': 42,
          'POS': 21,
          'PRP': 0,
          'PRP$': 26,
          'RB': 8,
          'RBR': 37,
          'RBS': 31,
          'RP': 34,
          'SYM': 43,
          'TO': 10,
          'UH': 40,
          'VB': 22,
          'VBD': 9,
          'VBG': 14,
          'VBN': 16,
          'VBP': 24,
          'VBZ': 1,
          'WDT': 27,
          'WP': 32,
          'WP$': 39,
          'WRB': 7,
          '``': 17,
          '<pad>': 45}
id2pos = {v: k for k, v in pos2id.items()}
ne2id = {'DATE': 3,
         'DURATION': 9,
         'LOCATION': 1,
         'MISC': 4,
         'MONEY': 10,
         'NUMBER': 6,
         'O': 0,
         'ORDINAL': 7,
         'ORGANIZATION': 5,
         'PERCENT': 8,
         'PERSON': 2,
         'TIME': 11,
         '<pad>': 12}
id2ne = {v: k for k, v in ne2id.items()}
case_dict = {'UP': 1, 'LOW': 0, '<pad>': 2}
bio_dict = {'B': 1, 'I': 2, 'O': 0, '<pad>': 3}


def read(path):
    line_list = []
    with open(path, encoding='utf-8', mode="r") as f:
        for line in f:
            line_list.append(line.strip())
    return line_list


def vocab_read(path, max_vocab_size=20000):
    word2id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}

    with open(path, encoding='utf-8', mode="r") as f:
        for i, line in enumerate(f):
            if i >= 4:
                line = line.strip().split(' ')
                word = line[0]
                id = len(word2id)
                word2id[word] = id
                if len(word2id) == max_vocab_size:
                    break
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def load_data(data_dir, split='train'):
    print(f'Load data from {data_dir}')
    if split == 'train':
        src = read(data_dir.joinpath('train.txt.source.txt'))
        ans_bio = read(data_dir.joinpath('train.txt.bio'))
        case = read(data_dir.joinpath('train.txt.case'))
        pos = read(data_dir.joinpath('train.txt.pos'))
        ner = read(data_dir.joinpath('train.txt.ner'))
        tgt = read(data_dir.joinpath('train.txt.target.txt'))

    elif split == 'dev':
        src = read(data_dir.joinpath('dev.txt.shuffle.dev.source.txt'))
        ans_bio = read(data_dir.joinpath('dev.txt.shuffle.dev.bio'))
        case = read(data_dir.joinpath('dev.txt.shuffle.dev.case'))
        pos = read(data_dir.joinpath('dev.txt.shuffle.dev.pos'))
        ner = read(data_dir.joinpath('dev.txt.shuffle.dev.ner'))
        tgt = read(data_dir.joinpath('dev.txt.shuffle.dev.target.txt'))

    elif split == 'test':
        src = read(data_dir.joinpath('dev.txt.shuffle.test.source.txt'))
        ans_bio = read(data_dir.joinpath('dev.txt.shuffle.test.bio'))
        case = read(data_dir.joinpath('dev.txt.shuffle.test.case'))
        pos = read(data_dir.joinpath('dev.txt.shuffle.test.pos'))
        ner = read(data_dir.joinpath('dev.txt.shuffle.test.ner'))
        tgt = read(data_dir.joinpath('dev.txt.shuffle.test.target.txt'))

    return src, ans_bio, case, pos, ner, tgt


def stem(word):
    return porter_stemmer.stem(word)


def load_word_vector(vector_path, word2id, dim=300):
    """
    Read pretrained vectors
    Make lookup table with vocabulary
    Load vector at lookup table
    """
    vocab_size = len(word2id)
    lookup_table = np.random.normal(size=[vocab_size, dim])

    if 'glove' in str(vector_path):
        n_total_vector = 400000

    n_covered = 0
    with open(vector_path, encoding='utf-8', mode="r") as f:
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


def preprocess_data(data_dir, split='train', n_process=8, vocab_path=None, PG=False):
    # word2id, id2word = vocab_read()

    print(f'Preprocessing {split} dataset...')
    src_data, ans_bio_data, case_data, pos_data, ner_data, tgt_data = load_data(data_dir, split)
    PG = [PG] * len(src_data)

    with Pool(n_process) as pool:
        data = list(tqdm(pool.imap(preprocess_single_example,
                                   zip(src_data, ans_bio_data, case_data, pos_data, ner_data, tgt_data, PG)),
                         total=len(src_data)))

    df = pd.DataFrame(data)
    print(f'Done! size: {len(df)}')
    return df


def preprocess_single_example(single_example):
    src, ans_bio, case, pos, ner, tgt, PG = single_example

    source_WORD = []
    source_WORD_encoding = []
    source_WORD_encoding_extended = []
    oovs = []
    answer_WORD = []
    answer_WORD_encoding = []
    answer_position_BIO = []
    answer_position_BIO_encoding = []
    answer_start = False
    for word, bio in zip(src.split(' '), ans_bio.split(' ')):

        source_WORD.append(word)
        answer_position_BIO.append(bio)
        answer_position_BIO_encoding.append(bio_dict[bio])

        if bio in ['B', 'I']:
            answer_WORD.append(word)
            if word in word2id:
                answer_WORD_encoding.append(word2id[word])
            else:
                answer_WORD_encoding.append(word2id['<unk>'])
            if not answer_start:
                answer_start = True
        else:
            assert bio == 'O'

        if word in word2id:
            source_WORD_encoding.append(word2id[word])
            source_WORD_encoding_extended.append(word2id[word])
        else:
            source_WORD_encoding.append(word2id['<unk>'])
            if word not in oovs:
                oovs.append(word)
            oov_num = oovs.index(word)
            source_WORD_encoding_extended.append(len(word2id) + oov_num)

    target_WORD = tgt.split(' ')
    target_WORD_encoding = []
    for word in target_WORD:
        if word in word2id:
            target_WORD_encoding.append(word2id[word])
        # can be copied
        else:
            if not PG:
                if word in source_WORD:
                    target_WORD_encoding.append(len(word2id) + source_WORD.index(word))
                else:
                    target_WORD_encoding.append(word2id['<unk>'])
            else:
                if word in oovs:
                    target_WORD_encoding.append(len(word2id) + oovs.index(word))
                else:
                    target_WORD_encoding.append(word2id['<unk>'])

    ner_split = ner.split(' ')
    ner_encoding = [ne2id[ne] for ne in ner_split]

    pos_split = pos.split(' ')
    pos_encoding = [pos2id[pos] for pos in pos_split]

    case_split = case.split(' ')
    case_encoding = [case_dict[case] for case in case_split]

    source_WORD_stem = [stem(word) for word in source_WORD]
    target_stem_set = set([stem(word) for word in target_WORD])

    focus_WORD = []
    focus_mask = []
    focus_input = []

    for i, (word, word_stem, word_pos) in enumerate(zip(source_WORD, source_WORD_stem, pos_split)):
        if word_stem in target_stem_set:
            if word not in stopword_set:
                if answer_position_BIO[i] == 'O':
                    focus_WORD.append(word)
                    focus_mask.append(1)
                    focus_input.append(word2id[word] if word in word2id else word2id['<unk>'])
                else:
                    assert answer_position_BIO[i] in ['B', 'I'], answer_position_BIO[i]
                    focus_mask.append(0)
            else:
                focus_mask.append(0)
        else:
            focus_mask.append(0)

    assert len(focus_mask) == len(source_WORD)

    example = {
        'answer_WORD': answer_WORD,
        'answer_WORD_encoding': answer_WORD_encoding,
        'source': src,
        'source_WORD': source_WORD,
        'source_WORD_encoding': source_WORD_encoding,
        'source_WORD_encoding_extended': source_WORD_encoding_extended,
        'source_len': len(source_WORD),
        'target': tgt,
        'target_WORD': target_WORD,
        'target_WORD_encoding': target_WORD_encoding,
        'target_len': len(target_WORD),
        'answer_position_BIO_encoding': answer_position_BIO_encoding,
        'ner': ner_split,
        'ner_encoding': ner_encoding,
        'pos': pos_split,
        'pos_encoding': pos_encoding,
        'case': case_split,
        'case_encoding': case_encoding,
        'focus_WORD': focus_WORD,
        'focus_mask': focus_mask,
        'focus_input': focus_input,
        'oovs': oovs,
    }
    return example


class SQuADDataset(Dataset):
    def __init__(self, df, split='train'):
        print('# Total size:', len(df))

        if split == 'train':
            idx = (df.source_len <= 100) \
                  & (df.target_len <= 100)
            self.df = df[idx].reset_index()
            print('# ignored (source or target longer then 100):', len(df) - len(self.df))
        else:
            self.df = df
        if split != 'test':
            self.df = self.df.sort_values('source_len', ascending=False).reset_index()

        print(f'Done! Size: {len(self.df)}')

    def __getitem__(self, idx):
        return self.df.loc[idx]

    def __len__(self):
        return len(self.df)


def get_QG_loader(df, mode='train', **kwargs):
    dataset = SQuADDataset(df, mode)

    def qg_collate_fn(batch):
        batch = pd.DataFrame(batch).reset_index(drop=True)

        # Add <EOS> at the end of target target
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

        answer_WORD_encoding = batch.answer_WORD_encoding.apply(torch.LongTensor)
        answer_WORD_encoding = pad_sequence(
            answer_WORD_encoding, batch_first=True, padding_value=0)

        answer_position_BIO_encoding = batch.answer_position_BIO_encoding.apply(torch.LongTensor)
        answer_position_BIO_encoding = pad_sequence(
            answer_position_BIO_encoding, batch_first=True, padding_value=3)

        ner_encoding = batch.ner_encoding.apply(torch.LongTensor)
        ner_encoding = pad_sequence(
            ner_encoding, batch_first=True, padding_value=12)

        pos_encoding = batch.pos_encoding.apply(torch.LongTensor)
        pos_encoding = pad_sequence(
            pos_encoding, batch_first=True, padding_value=45)

        case_encoding = batch.case_encoding.apply(torch.LongTensor)
        case_encoding = pad_sequence(
            case_encoding, batch_first=True, padding_value=2)

        focus_mask = batch.focus_mask.apply(torch.LongTensor)
        focus_mask = pad_sequence(
            focus_mask, batch_first=True, padding_value=2)

        # Add unused 0 padding to avoid empty batch
        focus_input = batch.focus_input.apply(
            lambda x: [0] + x)
        focus_input = focus_input.apply(torch.LongTensor)
        focus_input = pad_sequence(
            focus_input, batch_first=True, padding_value=0)

        # Raw words
        source_WORD = batch.source_WORD.tolist()
        target_WORD = batch.target_WORD.tolist()
        answer_WORD = batch.answer_WORD.tolist()
        ner = batch.ner.tolist()
        pos = batch.pos.tolist()
        case = batch.case.tolist()
        focus_WORD = batch.focus_WORD.tolist()

        source_len = batch.source_len.tolist()
        target_len = batch.target_len.tolist()

        oovs = batch.oovs.tolist()

        return source_WORD_encoding, source_len, \
               target_WORD_encoding, target_len, \
               source_WORD, target_WORD, \
               answer_position_BIO_encoding, answer_WORD, \
               ner, ner_encoding, \
               pos, pos_encoding, \
               case, case_encoding, \
               focus_WORD, focus_mask, \
               focus_input, answer_WORD_encoding, \
               source_WORD_encoding_extended, oovs

    return DataLoader(dataset, collate_fn=qg_collate_fn, **kwargs)


if __name__ == '__main__':
    # current_dir = Path(__file__).resolve().parent
    data_dir = Path('./squad')
    glove_dir = Path('./glove')
    out_dir = Path('./squad_out')
    out_dir.mkdir()

    word2id, id2word = vocab_read(data_dir.joinpath('vocab.txt'))

    with open(out_dir.joinpath('vocab.pkl'), 'wb') as f:
        pickle.dump((word2id, id2word), f)

    train_df = preprocess_data(data_dir, 'train')
    val_df = preprocess_data(data_dir, 'dev')
    test_df = preprocess_data(data_dir, 'test')

    word_vector = load_word_vector(glove_dir.joinpath('glove.6B.300d.txt'), word2id, dim=300)
    with open(out_dir.joinpath('word_vector.pkl'), 'wb') as f:
        pickle.dump(word_vector, f)

    train_df.to_pickle(out_dir.joinpath('train_df.pkl'))
    val_df.to_pickle(out_dir.joinpath('val_df.pkl'))
    test_df.to_pickle(out_dir.joinpath('test_df.pkl'))

    train_loader = get_QG_loader(
        train_df,
        mode='train',
        batch_size=10,
        shuffle=True,
        num_workers=1)

    val_loader = get_QG_loader(
        val_df,
        mode='val',
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )

    test_loader = get_QG_loader(
        test_df,
        mode='test',
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )
