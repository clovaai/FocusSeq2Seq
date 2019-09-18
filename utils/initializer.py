import torch.nn as nn


def init_wt_normal(wt):
    wt.data.normal_(std=1e-4)


def init_wt_uniform(wt):
    wt.data.uniform_(-0.02, 0.02)


def init_linear_wt(linear):
    print(f'Initalizing {linear}')
    init_wt_normal(linear.weight)
    if linear.bias is not None:
        init_wt_normal(linear.bias)


def init_lstm_wt(lstm):
    print(f'Initializing {lstm}')
    for name, p in lstm.named_parameters():
        if 'weight_' in name:
            weight = p
            init_wt_uniform(weight)
        elif 'bias_' in name:
            # set forget bias to 1
            bias = p
            n = bias.size(0)
            start, end = n // 4, n * 2 // 4
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)


def init_gru_wt(gru):
    print(f'Initializing {gru}')
    for name, p in gru.named_parameters():
        if 'weight_' in name:
            weight = p
            init_wt_uniform(weight)
        elif 'bias_' in name:
            # set forget bias to 1
            bias = p
            n = bias.size(0)
            start, end = n // 3, n * 2 // 3
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)


def init_rnn_wt(rnn):
    if isinstance(rnn, (nn.LSTMCell, nn.LSTM)):
        init_lstm_wt(rnn)
    elif isinstance(rnn, (nn.GRUCell, nn.GRU)):
        init_gru_wt(rnn)


def init_weight(module):
    """Recursively apply weight initialization"""
    if isinstance(module, nn.Linear):
        init_linear_wt(module)
    elif isinstance(module, (nn.LSTMCell, nn.LSTM, nn.GRUCell, nn.GRU)):
        init_rnn_wt(module)
    elif isinstance(module, nn.Embedding):
        init_wt_normal(module.weight)
