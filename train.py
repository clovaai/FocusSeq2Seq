"""FocusSeq2Seq
Copyright (c) 2019-present NAVER Corp.
MIT license
"""

import math
import time
import os
import getpass
import multiprocessing
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs import get_config
from utils.tensor_utils import repeat
from build_utils import get_loader, build_model, get_ckpt_name
from evaluate import evaluate

if __name__ == "__main__":

    config = get_config()
    print(config)

    time_now = time.strftime("%Y-%m-%d_%H:%M:%S")
    print(time_now)

    if config.task == 'QG':
        DATASET_PATH = './squad_out'
    elif config.task == 'SM':
        DATASET_PATH = './cnndm_out'
    data_dir = Path(DATASET_PATH).resolve()

    print('Current directory:', os.getcwd())
    print('Current user:', getpass.getuser())
    print('Dataset directory:', data_dir)

    print('PyTorch Version:', torch.__version__)
    print('# CPUs:', multiprocessing.cpu_count())
    print('# GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name(0))

    print('Seed:', config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    train_loader, val_loader, test_loader, word2id, id2word = get_loader(config, data_dir)

    PAD_ID, UNK_ID = word2id['<pad>'], word2id['<unk>']
    vocab_size = len(word2id)
    print('Loaded data loaders & Vocab!')
    print('Vocab Size:', vocab_size)

    model = build_model(config, word2id, id2word)

    print('#===== Parameters =====#')
    for name, p in model.named_parameters():
        print(name, '\t', list(p.size()))

    print('#==== Weight Initialization ====#')
    if config.model == 'NQG':
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
            else:
                nn.init.xavier_normal_(p, math.sqrt(3))
    elif config.model == 'PG':
        from utils.initializer import init_linear_wt, init_rnn_wt, init_wt_normal

        init_wt_normal(model.seq2seq.word_embed.weight)
        init_rnn_wt(model.seq2seq.encoder.rnn)
        for name, m in model.seq2seq.bridge.named_modules():
            if 'linear' in name:
                init_linear_wt(m)
        init_rnn_wt(model.seq2seq.decoder.rnncell)
        if not config.weight_tie:
            init_linear_wt(model.seq2seq.decoder.readout.mlp[-1])


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print('# Total Parameters:', count_parameters(model))

    # ------ Word Embedding ------#
    if config.load_glove:
        assert config.task == 'QG'
        # from QG_data_loader import load_word_vector
        # word_embedding = load_word_vector(
        #     data_dir.joinpath('glove.6B.300d.txt'), word2id, dim=300)
        with open(data_dir.joinpath('word_vector.pkl'), 'rb') as f:
            word_embedding = pickle.load(f)
        model.seq2seq.word_embed.from_pretrained(
            embeddings=torch.from_numpy(word_embedding),
            freeze=config.embedding_freeze)
        print('Loaded word embedding!')

    # ------ Use GPUs ------#
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'Loaded model on {torch.cuda.device_count()} GPUs!')

    # ------ Optimizer ------#
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=config.lr, amsgrad=False)
    elif config.optim.lower() == 'amsgrad':
        optimizer = optim.Adam(params, lr=config.lr, amsgrad=True)
    elif config.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(params, lr=config.lr, initial_accumulator_value=0.1)

    print('Setup Loss and Optimizer!')

    # Dry-Evalutation (To check evaluation function)
    if config.dry:
        print('Dry-Evaluation')
        evaluate(val_loader, model, 0, config)

    print('Training starts!')

    best_eval_metric = 0

    best_epoch = 0
    n_epoch_no_improvement = 0

    n_iter = len(train_loader)

    for epoch in range(config.epochs):
        epoch_start = time.time()
        temp_time_start = time.time()
        print('Epoch start!')
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
        model.train()

        epoch_loss = []

        epoch_nll_loss = []
        epoch_focus_loss = []
        epoch_cov_loss = []

        temp_nll_losses = []
        temp_focus_losses = []
        temp_cov_losses = []

        for batch_i, batch in enumerate(train_loader):
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
                answer_position_BIO_encoding = ner_encoding = pos_encoding = case_encoding = None
                answer_WORD_encoding = None

            # ============#
            # Focus Loss #
            # ============#
            if config.use_focus and not config.eval_focus_oracle:
                # ================================#
                # Hard-EM
                # 1) Select a minimum-loss SELECTOR expert (E-Step)
                # 2) Train with the selected SELECTOR expert (M-Step)
                # ================================#

                if config.n_mixture == 1:
                    B = source_WORD_encoding.size(0)
                    mixture_id = torch.zeros(B,
                                             dtype=torch.long, device=device)
                else:
                    # 1) Select a minimum-loss SELECTOR expert (E-Step)
                    model.selector.eval()
                    with torch.no_grad():

                        # [B * n_mixture, L]
                        focus_logit = model.selector(
                            source_WORD_encoding,
                            answer_position_BIO_encoding=answer_position_BIO_encoding,
                            ner_encoding=ner_encoding,
                            pos_encoding=pos_encoding,
                            case_encoding=case_encoding,
                            mixture_id=None,
                            focus_input=focus_input,
                            train=True)

                        B, L = source_WORD_encoding.size()

                        # [B * n_mixture, L]
                        repeated_target = repeat(focus_mask.float(), config.n_mixture)

                        # [B * n_mixture, L]
                        focus_loss = F.binary_cross_entropy_with_logits(
                            focus_logit,
                            repeated_target,
                            reduction='none').view(B, config.n_mixture, L)
                        pad_mask = (source_WORD_encoding == PAD_ID).view(
                            B, 1, L)
                        mixture_id = focus_loss.masked_fill(
                            pad_mask, 0).sum(dim=2).argmin(dim=1)

                # 2) Train with the selected SELECTOR expert (M-Step)
                model.selector.train()

                # [B, L]
                focus_logit = model.selector(
                    source_WORD_encoding,
                    answer_position_BIO_encoding=answer_position_BIO_encoding,
                    ner_encoding=ner_encoding,
                    pos_encoding=pos_encoding,
                    case_encoding=case_encoding,
                    mixture_id=mixture_id,
                    focus_input=focus_input,
                    train=True)

                # [B, L]
                focus_loss = F.binary_cross_entropy_with_logits(
                    focus_logit,
                    focus_mask.float(),
                    reduction='none')

                pad_mask = source_WORD_encoding == PAD_ID
                valid_mask = ~pad_mask
                focus_len = valid_mask.float().sum(dim=1)

                focus_loss.masked_fill_(pad_mask, 0)
                focus_loss = focus_loss.sum(dim=1) / focus_len
                focus_loss = focus_loss.mean()

            else:
                # No need to train Selector
                focus_loss = torch.zeros(1).squeeze().to(device)

            # ===============#
            # Seq2Seq Loss #
            # ===============#

            if config.mixture_decoder:
                # ================================#
                # Hard-EM
                # 1) Select a minimum-loss decoder expert (E-Step)
                # 2) Train with the selected decoder expert (M-Step)
                # ================================#

                # 1) Select a minimum-loss decoder expert (E-Step)
                model.seq2seq.eval()
                with torch.no_grad():
                    output = model.seq2seq(
                        source_WORD_encoding,
                        answer_WORD_encoding=answer_WORD_encoding,
                        answer_position_BIO_encoding=answer_position_BIO_encoding,
                        ner_encoding=ner_encoding,
                        pos_encoding=pos_encoding,
                        case_encoding=case_encoding,
                        focus_mask=focus_mask,
                        mixture_id=None,
                        target_WORD_encoding=target_WORD_encoding,
                        source_WORD_encoding_extended=source_WORD_encoding_extended,
                        train=True)

                    if config.model == 'PG':
                        log_p, cov_loss = output
                    else:
                        log_p = output

                    B, L = source_WORD_encoding.size()
                    B, max_dec_len = target_WORD_encoding.size()

                    # [B*n_mixture, max_dec_len]
                    repeated_target = repeat(target_WORD_encoding, config.n_mixture)
                    # [B, 1, max_dec_len]
                    dec_pad_mask = (target_WORD_encoding == PAD_ID).unsqueeze(1)
                    dec_valid_mask = ~dec_pad_mask
                    # [B, 1]
                    dec_len = dec_valid_mask.float().sum(dim=2)

                    # NLL Loss (Mean Negative log-likelihood)
                    # [B*n_mixture, max_dec_len]
                    nll_loss = -log_p.gather(2, repeated_target.unsqueeze(2)).squeeze(2)
                    # [B, n_mixture, max_dec_len]
                    nll_loss = nll_loss.view(B, config.n_mixture, max_dec_len)
                    nll_loss.masked_fill_(dec_pad_mask, 0)
                    # [B, n_mixture]
                    nll_loss = nll_loss.sum(dim=2) / dec_len

                    if config.model == 'PG':
                        # [B, n_mixture, max_dec_len]
                        cov_loss = cov_loss.view(B, config.n_mixture, max_dec_len)
                        cov_loss.masked_fill_(dec_pad_mask, 0)

                        # [B, n_mixture]
                        cov_loss = cov_loss.sum(dim=2) / dec_len

                        loss = nll_loss + cov_loss

                    else:
                        loss = nll_loss

                    # [B]
                    mixture_id = loss.argmin(dim=1)

                # 2) Train with the selected decoder expert (M-Step)
                model.seq2seq.train()
                output = model.seq2seq(
                    source_WORD_encoding,
                    answer_WORD_encoding=answer_WORD_encoding,
                    answer_position_BIO_encoding=answer_position_BIO_encoding,
                    ner_encoding=ner_encoding,
                    pos_encoding=pos_encoding,
                    case_encoding=case_encoding,
                    focus_mask=focus_mask,
                    mixture_id=mixture_id,
                    target_WORD_encoding=target_WORD_encoding,
                    source_WORD_encoding_extended=source_WORD_encoding_extended,
                    train=True)

            else:
                output = model.seq2seq(
                    source_WORD_encoding,
                    answer_WORD_encoding=answer_WORD_encoding,
                    answer_position_BIO_encoding=answer_position_BIO_encoding,
                    ner_encoding=ner_encoding,
                    pos_encoding=pos_encoding,
                    case_encoding=case_encoding,
                    focus_mask=focus_mask,
                    target_WORD_encoding=target_WORD_encoding,
                    source_WORD_encoding_extended=source_WORD_encoding_extended,
                    train=True)

            if config.model == 'PG':
                log_p, cov_loss = output
            else:
                log_p = output

            # max_target_len: Maximum target sequence length in current batch + 1 (EOS)
            B, max_target_len, ext_vocab_size = log_p.size()

            # [B, max_dec_len]
            dec_pad_mask = (target_WORD_encoding == PAD_ID)
            dec_valid_mask = ~dec_pad_mask
            # [B]
            dec_len = dec_valid_mask.float().sum(dim=1)

            # NLL Loss (Mean Negative log-likelihood)
            # [B, max_dec_len]
            nll_loss = -log_p.gather(2, target_WORD_encoding.unsqueeze(2)).squeeze(2)
            nll_loss.masked_fill_(dec_pad_mask, 0)
            nll_loss = nll_loss.sum(dim=1) / dec_len
            nll_loss = nll_loss.mean()

            if config.model == 'PG':
                # [B, max_dec_len]
                cov_loss.masked_fill_(dec_pad_mask, 0)
                cov_loss = cov_loss.sum(dim=1) / dec_len
                # cov_loss = cov_loss / dec_len
                cov_loss = cov_loss.mean()

            else:
                cov_loss = torch.zeros(1).squeeze().to(device)

            loss = nll_loss + focus_loss + cov_loss * config.coverage_lambda
            optimizer.zero_grad()
            loss.backward()
            if not config.no_clip:
                nn.utils.clip_grad_norm_(params, config.clip)
            optimizer.step()
            if config.model == 'NQG':
                for p in params:
                    p.data.clamp_(-15, 15)

            loss = loss.item()
            nll_loss = nll_loss.item()
            focus_loss = focus_loss.item()
            cov_loss = cov_loss.item()

            temp_nll_losses.append(nll_loss)
            temp_focus_losses.append(focus_loss)
            temp_cov_losses.append(cov_loss)

            epoch_nll_loss.append(nll_loss)
            epoch_focus_loss.append(focus_loss)
            epoch_cov_loss.append(cov_loss)
            epoch_loss.append(loss)

            if epoch == 0 and batch_i == 0:
                running_avg_nll_loss = nll_loss
                running_avg_cov_loss = cov_loss
            decay = 0.99
            running_avg_nll_loss = running_avg_nll_loss * decay + nll_loss * (1 - decay)
            running_avg_cov_loss = running_avg_cov_loss * decay + cov_loss * (1 - decay)

            if batch_i % 100 == 0 or (batch_i + 1) == n_iter:

                log_str = f'Epoch [{epoch}/{config.epochs}]'
                log_str += f' | Iteration [{batch_i}/{n_iter}]'
                log_str += f' | NLL Loss : {np.mean(temp_nll_losses):.3f}'
                log_str += f' | NLL Loss (running avg) : {running_avg_nll_loss:.3f}'
                if config.use_focus:
                    log_str += f' | Focus Loss : {np.mean(temp_focus_losses):.3f}'
                if config.model == 'PG':
                    log_str += f' | Coverage Loss : {np.mean(temp_cov_losses):.3f}'
                    log_str += f' | Coverage Loss (running avg) : {running_avg_cov_loss:.3f}'
                time_taken = time.time() - temp_time_start
                log_str += f' | Time taken: : {time_taken:.2f}'
                print(log_str)

                temp_nll_losses = []
                temp_focus_losses = []
                temp_cov_losses = []
                temp_time_start = time.time()

        epoch_time_taken = time.time() - epoch_start
        print(f'Epoch Done! It took {epoch_time_taken:.2f}s')

        if epoch >= 0:
            metric_result, hypotheses, best_hypothesis, hyp_focus, hyp_attention = evaluate(
                val_loader, model, epoch, config)

            if config.eval_focus_oracle or max(config.n_mixture, config.decode_k) == 1:
                if config.task == 'QG':
                    metric_name = 'BLEU-4'
                elif config.task == 'SM':
                    metric_name = 'ROUGE-2'
            else:
                if config.task == 'QG':
                    metric_name = 'Oracle_BLEU-4'
                elif config.task == 'SM':
                    metric_name = 'Oracle_ROUGE-2'
            eval_metric = metric_result[metric_name]

            ckpt_dir = Path(f"./ckpt/{config.model}/").resolve()
            ckpt_dir.mkdir(exist_ok=True)
            filename = get_ckpt_name(config)
            filename += f"_epoch{epoch}.pkl"
            ckpt_path = ckpt_dir.joinpath(filename)
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'hypotheses': hypotheses,
                'metric': metric_result,
            }
            torch.save(checkpoint, ckpt_path)
            print("Model saved!")

            if eval_metric > best_eval_metric:
                best_epoch = epoch
                best_eval_metric = eval_metric

                checkpoint = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'hypotheses': hypotheses,
                    'metric': metric_result,
                }
                ckpt_dir = Path(f"./ckpt/{config.model}/").resolve()
                ckpt_dir.mkdir(exist_ok=True)
                filename = get_ckpt_name(config)
                filename += f"_epoch{epoch}.pkl"
                ckpt_path = ckpt_dir.joinpath(filename)
                # torch.save(checkpoint, ckpt_path)
                # print("Model saved!")
                best_ckpt_path = ckpt_path

                n_epoch_no_improvement = 0
            else:
                n_epoch_no_improvement += 1

            print(f'Best epoch: {best_epoch}')
            print(f'Best {metric_name}: {best_eval_metric:.3f}')

            # if config.task == 'QG':
            if n_epoch_no_improvement >= 3 and optimizer.param_groups[0]['lr'] >= 1e-6:
                optimizer.param_groups[0]['lr'] *= 0.5
                print('Halving Learning Rate => ', optimizer.param_groups[0]['lr'])

    print('Test Set Evaluation')
    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint['model'])
    print('Loaded checkpoint from best epoch:', best_epoch)

    metric_result, hypotheses, best_hypothesis, hyp_focus, hyp_attention = evaluate(
        test_loader, model, best_epoch, config, test=True)
    print('Best validation epoch:', best_epoch)
    print(f'Best validation {metric_name}: {best_eval_metric:.3f}')
