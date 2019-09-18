"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def read_config(path):
    return Config.load(path)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_ckpt', type=int, default=9)

    # Task / Model / Data
    parser.add_argument('--task', type=str, default='QG',
                        choices=['QG', 'SM'],
                        help='QG: Question Generation / SM: Summarization')
    parser.add_argument('--model', type=str, default='NQG',
                        choices=['NQG', 'PG'],
                        help='NQG: NQG++ (Zhou et al. 2017) / PG: Pointer Generator (See et al. 2017)')
    parser.add_argument('--data', type=str, default='squad',
                        choices=['squad', 'cnndm'])

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clip norm')
    parser.add_argument('--no_clip', action='store_true',
                        help="Not to use gradient clipping")
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'amsgrad', 'adagrad'],
                        help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--dry', action='store_true',
                        help='Run training script without actually running training steps. Debugging only')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')

    # Evaluation
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='batch size during evaluation')
    parser.add_argument('--val_data_size', type=int, default=1000,
                        help='number of examples for validation / Use (for debugging) when evaluation for summarization takes so long')

    # Seq2Seq Model
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--enc_hidden_size', type=int, default=512)
    parser.add_argument('--dec_hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--rnn', type=str, default='GRU')
    parser.add_argument('--weight_tie', type=str2bool, default=True,
                        help='output layer tied with embedding')
    parser.add_argument('--embedding_freeze', type=str2bool, default=False,
                        help='Freeze word embedding during training')
    parser.add_argument('--load_glove', type=str2bool, default=True,
                        help='Initialize word embedding from glove (NQG++ only)')
    parser.add_argument('--feature_rich', action='store_true',
                        help='Use linguistic features (POS/NER/Word Case/Answer position; NQG++ only)')
    parser.add_argument('--coverage_lambda', type=float, default=1.0,
                        help='hyperparameter for coverage (Pointer Generator only)')

    # Seq2Seq Decoding
    parser.add_argument('--decoding', type=str, default='beam',
                        choices=['greedy', 'beam', 'diverse_beam', 'topk_sampling'])
    parser.add_argument('--beam_k', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--diversity_lambda', type=float, default=0.5)
    parser.add_argument('--decode_k', type=int, default=1)
    parser.add_argument('--mixture_decoder', action='store_true',
                        help='Hard Uniform Mixture Decoder (Shen et al. 2018)')

    # Focus
    parser.add_argument('--use_focus', type=str2bool, default=True,
                        help='whether to use focus or not')
    parser.add_argument('--eval_focus_oracle', action='store_true',
                        help='Feed focus guide even during test time')

    # Selector
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='focus binarization threshold')

    # Mixture
    parser.add_argument('--n_mixture', type=int, default=1,
                        help='Number of mixtures for Selector (Ours) or Mixture Decoder (Shen et al. 2018)')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()

    # Save
    config.save('config.txt')

    # Load
    loaded_config = read_config('config.txt')

    assert config.__dict__ == loaded_config.__dict__

    import os

    os.remove('config.txt')
