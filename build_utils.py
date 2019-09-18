"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import pickle
from pathlib import Path
import pandas as pd

from models import Model, Seq2Seq, FocusSelector

current_dir = Path(__file__).resolve().parent


def get_loader(config, data_dir):
    if config.task == 'QG':
        assert config.data == 'squad'
        import QG_data_loader

        data_dir = current_dir.joinpath('squad_out/')

        with open(data_dir.joinpath('vocab.pkl'), 'rb') as f:
            word2id, id2word = pickle.load(f)

        train_df = pd.read_pickle(data_dir.joinpath('train_df.pkl'))

        if not config.eval_only:
            train_df = pd.read_pickle(data_dir.joinpath('train_df.pkl'))
            val_df = pd.read_pickle(data_dir.joinpath('val_df.pkl'))

            train_loader = QG_data_loader.get_QG_loader(
                train_df,
                mode='train',
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=1
            )
            val_loader = QG_data_loader.get_QG_loader(
                val_df,
                mode='val',
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=1
            )
        else:
            train_loader = None
            val_loader = None

        test_df = pd.read_pickle(data_dir.joinpath('test_df.pkl'))
        test_loader = QG_data_loader.get_QG_loader(
            test_df,
            mode='test',
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=1,
        )

    elif config.task == 'SM':
        assert config.data == 'cnndm'
        import CNNDM_data_loader

        data_dir = current_dir.joinpath('cnndm_out/')
        with open(data_dir.joinpath('vocab.pkl'), 'rb') as f:
            word2id, id2word = pickle.load(f)

        if not config.eval_only:
            train_df = pd.read_pickle(data_dir.joinpath('train_df.pkl'))
            val_df = pd.read_pickle(data_dir.joinpath('val_df.pkl'))

            train_loader = CNNDM_data_loader.get_SM_loader(
                train_df,
                batch_size=config.batch_size,
                shuffle=True,
            )
            val_loader = CNNDM_data_loader.get_SM_loader(
                val_df,
                batch_size=max(4, config.batch_size // config.n_mixture),
                # batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=1,
            )
        else:
            train_loader = None
            val_loader = None

        test_df = pd.read_pickle(data_dir.joinpath('test_df.pkl'))
        test_loader = CNNDM_data_loader.get_SM_loader(
            test_df,
            # batch_size=max(4, config.batch_size // config.n_mixture),
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=1,
        )

    return train_loader, val_loader, test_loader, word2id, id2word


def build_model(config, word2id, id2word):
    # ------ Model ------#
    seq2seq = Seq2Seq(
        vocab_size=len(word2id),
        word_embed_size=config.embed_size,
        num_layers=config.num_layers,
        dropout_p=config.dropout,
        enc_hidden_size=config.enc_hidden_size,
        dec_hidden_size=config.dec_hidden_size,
        use_focus=config.use_focus,
        tie=config.weight_tie,
        task=config.task,
        rnn=config.rnn,
        model=config.model,
        feature_rich=config.feature_rich,
        n_mixture=config.n_mixture if config.mixture_decoder else None
    )
    print('Created Seq2Seq!')

    if config.use_focus and not config.eval_focus_oracle:
        selector = FocusSelector(
            word_embed_size=config.embed_size,
            num_layers=config.num_layers,
            enc_hidden_size=config.enc_hidden_size,
            dec_hidden_size=config.dec_hidden_size,
            n_mixture=config.n_mixture,
            dropout_p=0.2,
            rnn=config.rnn,
            seq2seq_model=config.model,
            task=config.task,
            threshold=config.threshold,
            feature_rich=config.feature_rich
        )

        # Selector share all embeddings with Seq2Seq model
        if config.feature_rich:
            selector.add_embedding(
                seq2seq.word_embed,
                seq2seq.answer_position_embed,
                seq2seq.ner_embed,
                seq2seq.pos_embed,
                seq2seq.case_embed)
        else:
            selector.add_embedding(
                seq2seq.word_embed)
        print('Created Focus Selector!')
    else:
        selector = None

    model = Model(seq2seq, selector)

    model.word2id = word2id
    model.id2word = id2word

    print(model)
    return model


def get_ckpt_name(config):
    if config.eval_focus_oracle:
        filename = "Oracle"
    elif config.use_focus:
        filename = "Focus"
        filename += f"{config.n_mixture}"
    elif config.mixture_decoder:
        filename = "MixtureDecoder"
        assert config.n_mixture > 1
        filename += f"{config.n_mixture}"
    elif config.decoding == 'diverse_beam':
        filename = "DiverseBeam"
        filename += f"{config.beam_k}"
        if config.decode_k > 1:
            filename += f"_decode{config.decode_k}"
    else:
        filename = "Baseline"

    if config.decoding == "greedy":
        filename += "_Greedy"
    elif config.decoding == 'beam':
        filename += f"_Beam{config.beam_k}"
        if config.decode_k > 1:
            filename += f"_decode{config.decode_k}"
    elif config.decoding == 'topk_sampling':
        filename += f"_Sampling{config.temperature}"
        if config.n_mixture > 1:
            filename += f"_decode{config.n_mixture}"
    return filename
