#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import NEAR_INF
from .custom_LSTM import \
    Basic_LSTM, BN_LSTM, Adaptive_LSTM, Context_Adaptive_LSTM, \
    Topic_Adaptive_LSTM, Context_Topic_Adaptive_LSTM, \
    Context_SVD_Adaptive_LSTM
from .utils import FeedForward


def _transpose_hidden_state(hidden_state):
    """
    Transpose the hidden state so that batch is the first dimension.

    RNN modules produce (num_layers x batchsize x dim) hidden state, but
    DataParallel expects batch size to be first. This helper is used to
    ensure that we're always outputting batch-first, in case DataParallel
    tries to stitch things back together.
    """
    if isinstance(hidden_state, tuple):
        return tuple(map(_transpose_hidden_state, hidden_state))
    elif torch.is_tensor(hidden_state):
        return hidden_state.transpose(0, 1)
    else:
        raise ValueError("Don't know how to transpose {}".format(hidden_state))


def opt_to_kwargs(opt):
    """Get kwargs for seq2seq from opt."""
    kwargs = {}
    for k in ['numlayers', 'dropout', 'bidirectional', 'rnn_class',
              'lookuptable', 'decoder', 'numsoftmax',
              'attention', 'attention_length', 'attention_time',
              'input_dropout']:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs


class AdaND(TorchGeneratorModel):
    """Adaptive Sequence to sequence parent module."""

    RNN_OPTS = {
        'bn_lstm': BN_LSTM,
        'basic_lstm': Basic_LSTM,
        'adaptive_lstm': Adaptive_LSTM,
        'topic_adaptive_lstm': Topic_Adaptive_LSTM,
        'context_adaptive_lstm': Context_Adaptive_LSTM,
        'context_svd_adaptive_lstm': Context_SVD_Adaptive_LSTM,
        'context_topic_adaptive_lstm': Context_Topic_Adaptive_LSTM,
    }

    def __init__(
            self, num_features, embeddingsize, hiddensize, numlayers=2, dropout=0,
            bidirectional=False, rnn_class='basic_lstm', lookuptable='unique',
            decoder='same', numsoftmax=1, attention='none', attention_length=48,
            attention_time='post', padding_idx=0, start_idx=1, end_idx=2, unknown_idx=3,
            input_dropout=0, longest_label=1, adaptive_input_size=16, adaptive_hidden_size=128,
            num_topics=3, topic_dict=None, global_dict=None, latent_size=128,
            bow_hiddensizes=(512, 128, 64), ensemble_factors=128, use_cuda=True,
            use_outside_lda_theta=False, outside_lda_num_topics=5,
    ):
        """Initialize adaptive seq2seq model.

        See cmdline args in AdaNDAgent for description of arguments.
        """
        self.use_outside_lda_theta = use_outside_lda_theta
        self.outside_lda_num_topics = outside_lda_num_topics
        super().__init__(
            padding_idx=padding_idx, start_idx=start_idx,
            end_idx=end_idx, unknown_idx=unknown_idx,
            input_dropout=input_dropout, longest_label=longest_label,
        )
        self.attn_type = attention
        self.rnn_class = rnn_class
        rnn_class = AdaND.RNN_OPTS[rnn_class]
        self.decoder = self._create_decoder(
            num_features, embeddingsize, hiddensize,
            padding_idx=padding_idx, rnn_class=rnn_class,
            numlayers=numlayers, dropout=dropout,
            attn_type=attention, attn_length=attention_length,
            attn_time=attention_time,
            bidir_input=bidirectional,
            adaptive_input_size=adaptive_input_size,
            adaptive_hidden_size=adaptive_hidden_size,
            num_topics=num_topics if not use_outside_lda_theta else outside_lda_num_topics,
            ensemble_factors=ensemble_factors,
        )

        shared_lt = (self.decoder.lt  # share embeddings between rnns
                     if lookuptable in ('enc_dec', 'all') else None)
        shared_rnn = self.decoder.rnn if decoder == 'shared' else None
        self.encoder = RNNEncoder(
            num_features, embeddingsize, hiddensize,
            padding_idx=padding_idx, rnn_class=rnn_class,
            numlayers=numlayers, dropout=dropout,
            bidirectional=bidirectional,
            shared_lt=shared_lt, shared_rnn=shared_rnn,
            unknown_idx=unknown_idx, input_dropout=input_dropout,
            adaptive_input_size=adaptive_input_size,
            adaptive_hidden_size=adaptive_hidden_size,
            num_topics=num_topics if not use_outside_lda_theta else outside_lda_num_topics,
            ensemble_factors=ensemble_factors,
        )

        shared_weight = (self.decoder.lt  # use embeddings for projection
                         if lookuptable in ('dec_out', 'all') else None)
        # self.output = OutputLayer(
        #     num_features, embeddingsize, hiddensize, dropout=dropout,
        #     numsoftmax=numsoftmax, shared_weight=shared_weight,
        #     padding_idx=padding_idx)
        self.output = self._create_output_layer(num_features, embeddingsize, hiddensize,
                                                dropout, numsoftmax, shared_weight,
                                                padding_idx)

        special_tokens = [self.NULL_IDX, start_idx, self.END_IDX, unknown_idx]
        self.topic_indicator = TopicIndicator(
            num_topics, topic_dict, global_dict,
            embeddingsize, dropout=dropout,
            latent_size=latent_size, bow_hiddensizes=bow_hiddensizes,
            special_tokens=special_tokens, use_cuda=use_cuda
        )
        self.context_encoder = ContextEncoder(
            num_features, embeddingsize, hiddensize, padding_idx,
            numlayers, dropout, bidirectional, unknown_idx,
            input_dropout, adaptive_input_size, adaptive_hidden_size,
            num_topics, ensemble_factors,
        )
        # self.bow_project = torch.nn.Sequential(
        #     torch.nn.Linear(hiddensize + latent_size, 512),
        #     torch.nn.Tanh(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(512, num_features)
        # )

    def _create_output_layer(self, num_features, embeddingsize, hiddensize, dropout=0,
                             numsoftmax=1, shared_weight=None, padding_idx=-1):
        output_layer = OutputLayer(
            num_features, embeddingsize, hiddensize, dropout=dropout,
            numsoftmax=numsoftmax, shared_weight=shared_weight,
            padding_idx=padding_idx
        )
        return output_layer

    def _create_decoder(self, num_features, embeddingsize, hiddensize,
                        padding_idx=0, rnn_class=Basic_LSTM, numlayers=2, dropout=0.1,
                        bidir_input=False, attn_type='none', attn_time='pre',
                        attn_length=-1, sparse=False, adaptive_input_size=16,
                        adaptive_hidden_size=128, num_topics=3, ensemble_factors=128):
        return RNNDecoder(
            num_features, embeddingsize, hiddensize,
            padding_idx=padding_idx, rnn_class=rnn_class,
            numlayers=numlayers, dropout=dropout,
            attn_type=attn_type, attn_length=attn_length,
            attn_time=attn_time,
            bidir_input=bidir_input,
            adaptive_input_size=adaptive_input_size,
            adaptive_hidden_size=adaptive_hidden_size,
            num_topics=num_topics,
            ensemble_factors=ensemble_factors,
        )

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def reorder_encoder_states(self, encoder_states, indices):
        """Reorder encoder states according to a new set of indices."""
        enc_out, hidden, attn_mask = encoder_states

        # make sure we swap the hidden state around, apropos multigpu settings
        hidden = _transpose_hidden_state(hidden)

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        if self.attn_type != 'none':
            enc_out = enc_out.index_select(0, indices)
            attn_mask = attn_mask.index_select(0, indices)

        # and bring it back to multigpu friendliness
        hidden = _transpose_hidden_state(hidden)

        return enc_out, hidden, attn_mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 0, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )

    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None,
                maxlen=None, bsz=None):
        """Get output predictions from the model.

        :param xs:
            input to the encoder.

        :type xs:
            Tuple: model inputs

        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.

        :type ys:
            LongTensor[bsz, outlen]

        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.

        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.

        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :param ada_context_type:
            which context to use for adaptation input, choices: 'context' for
            input context, 'exemplar' for exemplar response, and 'exemplar_context'
            for the context of exemplar response.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if ys is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        xs, lda_theta = xs
        ada_context = xs

        theta = None
        if self.rnn_class == 'context_adaptive_lstm' or self.rnn_class == 'context_svd_adaptive_lstm':
            context = self.context_encoder(ada_context),
        elif self.rnn_class == 'topic_adaptive_lstm':
            if not self.use_outside_lda_theta:
                with torch.no_grad():
                    theta, *_ = self.topic_indicator(xs)
            else:
                theta = lda_theta
            context = theta,
        elif self.rnn_class == 'context_topic_adaptive_lstm':
            context_repr = self.context_encoder(ada_context)
            if not self.use_outside_lda_theta:
                with torch.no_grad():
                    theta, *_ = self.topic_indicator(xs)
            else:
                theta = lda_theta
            context = context_repr, theta
        else:
            context = None,

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs, *context)

        if ys is not None:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, ys, *context)
        else:
            if bsz is None:
                bsz = xs.size(0)
            scores, preds = self.decode_greedy(
                encoder_states, bsz, maxlen or self.longest_label, *context
            )

        return scores, preds, encoder_states, theta

    def decode_forced(self, encoder_states, ys, *context):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, None, *context)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def decode_greedy(self, encoder_states, bsz, maxlen, *context):
        """Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, incr_state, *context)
            scores = scores[:, -1:, :]
            scores = self.output(scores)
            _, preds = scores.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs


class UnknownDropout(nn.Module):
    """With set frequency, replaces tokens with unknown token.

    This layer can be used right before an embedding layer to make the model
    more robust to unknown words at test time.
    """

    def __init__(self, unknown_idx, probability):
        """Initialize layer.

        :param unknown_idx: index of unknown token, replace tokens with this
        :param probability: during training, replaces tokens with unknown token
                            at this rate.
        """
        super().__init__()
        self.unknown_idx = unknown_idx
        self.prob = probability

    def forward(self, input_):
        """If training and dropout rate > 0, masks input with unknown token."""
        if self.training and self.prob > 0:
            mask = input_.new(input_.size()).float().uniform_(0, 1) < self.prob
            input_ = input_.masked_fill(mask, self.unknown_idx)
        return input_


class RNNEncoder(nn.Module):
    """RNN Encoder."""

    def __init__(self, num_features, embeddingsize, hiddensize, padding_idx=0,
                 rnn_class=Basic_LSTM, numlayers=2, dropout=0.1,
                 bidirectional=False, shared_lt=None, shared_rnn=None,
                 input_dropout=0, unknown_idx=None, sparse=False,
                 adaptive_input_size=16, adaptive_hidden_size=128,
                 num_topics=3, ensemble_factors=128):
        """Initialize recurrent encoder."""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hiddensize
        self.adaptive_hsz = adaptive_hidden_size
        self.rnn_class = rnn_class

        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.input_dropout = UnknownDropout(unknown_idx, input_dropout)

        if shared_lt is None:
            # self.lt = nn.Embedding(num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse)
            # TODO: (BUG) padding_idx will lead to zero embedding values for NULL_IDX,
            #  making the reverse adaptive LSTM layer explosion.
            self.lt = nn.Embedding(num_features, embeddingsize, sparse=sparse)
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            if rnn_class == Adaptive_LSTM or rnn_class == Context_Adaptive_LSTM or \
                    rnn_class == Context_SVD_Adaptive_LSTM:
                # noinspection PyArgumentList
                self.rnn = rnn_class(
                    embeddingsize, hiddensize, numlayers,
                    dropout=dropout if numlayers > 1 else 0,
                    batch_first=True, bidirectional=bidirectional,
                    adaptive_input_size=adaptive_input_size,
                    adaptive_hidden_size=adaptive_hidden_size
                )
            elif rnn_class == Context_Topic_Adaptive_LSTM:
                self.rnn = Context_Topic_Adaptive_LSTM(
                    embeddingsize, hiddensize, numlayers,
                    dropout=dropout if numlayers > 1 else 0,
                    batch_first=True, bidirectional=bidirectional,
                    adaptive_input_size=adaptive_input_size,
                    adaptive_hidden_size=adaptive_hidden_size,
                    num_topics=num_topics,
                    ensemble_factors=ensemble_factors,
                )
            elif rnn_class == Topic_Adaptive_LSTM:
                self.rnn = Topic_Adaptive_LSTM(
                    embeddingsize, hiddensize, numlayers,
                    dropout=dropout if numlayers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional,
                    num_topics=num_topics,
                    ensemble_factors=ensemble_factors
                )
            else:
                self.rnn = rnn_class(embeddingsize, hiddensize, numlayers,
                                     dropout=dropout if numlayers > 1 else 0,
                                     batch_first=True, bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs, *context):
        """Encode sequence.
        :returns: encoder outputs, hidden state, attention mask
            encoder outputs are the output state at each step of the encoding.
            the hidden state is the final hidden state of the encoder.
            the attention mask is a mask of which input values are nonzero.
        """
        bsz = len(xs)

        # embed input tokens
        xs = self.input_dropout(xs)
        xes = self.dropout(self.lt(xs))
        attn_mask = xs.ne(0)

        # TODO: implement the packed data handling in custom LSTM Cell
        # try:
        #     x_lens = torch.sum(attn_mask.int(), dim=1)
        #     xes = pack_padded_sequence(xes, x_lens, batch_first=True)
        #     packed = True
        # except ValueError:
        #     # packing failed, don't pack then
        #     packed = False
        packed = False

        encoder_output, hidden = self.rnn(xes, None, *context)
        if packed:
            # total_length to make sure we give the proper length in the case
            # of multigpu settings.
            # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
            encoder_output, _ = pad_packed_sequence(
                encoder_output, batch_first=True, total_length=xs.size(1)
            )

        hsz = self.hsz + self.adaptive_hsz if issubclass(self.rnn_class, Adaptive_LSTM) else self.hsz
        hidden = self.taking_sum_hidden(hidden, bsz, hsz)

        return encoder_output, _transpose_hidden_state(hidden), attn_mask

    def taking_sum_hidden(self, hidden, bsz, hsz):
        if self.dirs > 1:
            # project to decoder dimension by taking sum of forward and back
            if type(hidden) is tuple:
                hidden = (hidden[0].view(-1, self.dirs, bsz, hsz).sum(1),
                          hidden[1].view(-1, self.dirs, bsz, hsz).sum(1))
            else:
                hidden = hidden.view(-1, self.dirs, bsz, hsz).sum(1)
        return hidden


class RNNDecoder(nn.Module):
    """Recurrent decoder module.

    Can be used as a standalone language model or paired with an encoder.
    """

    def __init__(self, num_features, embeddingsize, hiddensize, padding_idx=0,
                 rnn_class=Basic_LSTM, numlayers=2, dropout=0.1,
                 bidir_input=False, attn_type='none', attn_time='pre',
                 attn_length=-1, sparse=False, adaptive_input_size=16,
                 adaptive_hidden_size=128, num_topics=3,
                 ensemble_factors=128):
        """Initialize recurrent decoder."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize
        self.adaptive_hsz = adaptive_hidden_size
        self.rnn_class = rnn_class

        # self.lt = nn.Embedding(num_features, embeddingsize,
        #                        padding_idx=padding_idx, sparse=sparse)
        # TODO: (BUG) padding_idx will lead to zero embedding values for NULL_IDX,
        #  making the reverse adaptive LSTM layer explosion.
        self.lt = nn.Embedding(num_features, embeddingsize, sparse=sparse)
        if rnn_class == Adaptive_LSTM or rnn_class == Context_Adaptive_LSTM or \
                rnn_class == Context_SVD_Adaptive_LSTM:
            # noinspection PyArgumentList
            self.rnn = rnn_class(
                embeddingsize, hiddensize, numlayers,
                dropout=dropout if numlayers > 1 else 0,
                batch_first=True,
                adaptive_input_size=adaptive_input_size,
                adaptive_hidden_size=adaptive_hidden_size
            )
        elif rnn_class == Context_Topic_Adaptive_LSTM:
            self.rnn = Context_Topic_Adaptive_LSTM(
                embeddingsize, hiddensize, numlayers,
                dropout=dropout if numlayers > 1 else 0,
                batch_first=True,
                adaptive_input_size=adaptive_input_size,
                adaptive_hidden_size=adaptive_hidden_size,
                num_topics=num_topics,
                ensemble_factors=ensemble_factors,
            )
        elif rnn_class == Topic_Adaptive_LSTM:
            self.rnn = Topic_Adaptive_LSTM(
                embeddingsize, hiddensize, numlayers,
                dropout=dropout if numlayers > 1 else 0,
                batch_first=True, num_topics=num_topics,
                ensemble_factors=ensemble_factors
            )
        else:
            self.rnn = rnn_class(embeddingsize, hiddensize, numlayers,
                                 dropout=dropout if numlayers > 1 else 0,
                                 batch_first=True)

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type,
                                        hiddensize=hiddensize,
                                        embeddingsize=embeddingsize,
                                        bidirectional=bidir_input,
                                        attn_length=attn_length,
                                        attn_time=attn_time)

    def forward(self, xs, encoder_output, incremental_state=None, *context):
        """Decode from input tokens.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from RNNEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask) tuple.
        :param incremental_state: most recent hidden state to the decoder.
            If None, the hidden state of the encoder is used as initial state,
            and the full sequence is computed. If not None, computes only the
            next forward in the sequence.

        :returns: (output, hidden_state) pair from the RNN.

            - output is a bsz x time x latentdim matrix. If incremental_state is
                given, the time dimension will be 1. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        enc_state, enc_hidden, attn_mask = encoder_output
        # in case of multi gpu, we need to transpose back out the hidden state
        attn_params = (enc_state, attn_mask)

        if incremental_state is not None:
            # we're doing it piece by piece, so we have a more important hidden
            # seed, and we only need to compute for the final timestep
            hidden = _transpose_hidden_state(incremental_state)
            # only need the last timestep then
            xs = xs[:, -1:]
        else:
            # starting fresh, or generating from scratch. Use the encoder hidden
            # state as our start state
            hidden = _transpose_hidden_state(enc_hidden)

        adaptive_hidden = False
        if isinstance(hidden, tuple):
            hidden = tuple(x.contiguous() for x in hidden)
            if hidden[0].size(-1) == self.hsz + self.adaptive_hsz:
                adaptive_hidden = True
        else:
            hidden = hidden.contiguous()
            if hidden.size(-1) == self.hsz + self.adaptive_hsz:
                adaptive_hidden = True

        # sequence indices => sequence embeddings
        seqlen = xs.size(1)
        xes = self.dropout(self.lt(xs))

        if self.attn_time == 'pre':
            # modify input vectors with attention
            # attention module requires we do this one step at a time
            new_xes = []
            # remove the adaptive states from the hidden
            if adaptive_hidden:
                hidden4attn = _remove_adaptive_states(hidden, self.hsz)
            else:
                hidden4attn = hidden
            for i in range(seqlen):
                nx, _, _ = self.attention(xes[:, i:i + 1], hidden4attn, attn_params)
                new_xes.append(nx)
            xes = torch.cat(new_xes, 1).to(xes.device)

        if self.attn_time != 'post':
            # no attn, we can just trust the rnn to run through
            output, new_hidden = self.rnn(xes, hidden, *context)
        else:
            # uh oh, post attn, we need run through one at a time, and do the
            # attention modifications
            new_hidden = hidden
            output = []
            for i in range(seqlen):
                o, new_hidden = self.rnn(xes[:, i, :].unsqueeze(1), new_hidden, *context)
                # remove the adaptive states from the new_hidden
                if issubclass(self.rnn_class, Adaptive_LSTM):
                    new_hidden4attn = _remove_adaptive_states(new_hidden, self.hsz)
                else:
                    new_hidden4attn = new_hidden
                o, _, _ = self.attention(o, new_hidden4attn, attn_params)
                output.append(o)
            output = torch.cat(output, dim=1).to(xes.device)

        return output, _transpose_hidden_state(new_hidden)


def _remove_adaptive_states(states, hsz):
    if isinstance(states, tuple):
        return tuple(map(_remove_adaptive_states, states, [hsz, hsz]))
    elif torch.is_tensor(states):
        return states[:, :, 0:hsz]
    else:
        raise ValueError("Don't know how to remove adaptive states from {}".format(states))


class Identity(nn.Module):
    def forward(self, x):
        return x


class OutputLayer(nn.Module):
    """Takes in final states and returns distribution over candidates."""

    def __init__(self, num_features, embeddingsize, hiddensize, dropout=0,
                 numsoftmax=1, shared_weight=None, padding_idx=-1):
        """Initialize output layer.

        :param num_features:  number of candidates to rank
        :param hiddensize:    (last) dimension of the input vectors
        :param embeddingsize: (last) dimension of the candidate vectors
        :param numsoftmax:   (default 1) number of softmaxes to calculate.
                              see arxiv.org/abs/1711.03953 for more info.
                              increasing this slows down computation but can
                              add more expressivity to the embeddings.
        :param shared_weight: (num_features x esz) vector of weights to use as
                              the final linear layer's weight matrix. default
                              None starts with a new linear layer.
        :param padding_idx:   model should output a large negative number for
                              score at this index. if set to -1 (default),
                              this is disabled. if >= 0, subtracts one from
                              num_features and always outputs -1e20 at this
                              index. only used when shared_weight is not None.
                              setting this param helps protect gradient from
                              entering shared embedding matrices.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.padding_idx = padding_idx
        self.e2s = nn.Linear(embeddingsize, num_features)
        self.shared = False

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.esz = embeddingsize
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hiddensize, numsoftmax, bias=False)
            self.latent = nn.Linear(hiddensize, numsoftmax * embeddingsize)
            self.activation = nn.Tanh()
        else:
            # rnn output to embedding
            if hiddensize != embeddingsize:
                # learn projection to correct dimensions
                self.o2e = nn.Linear(hiddensize, embeddingsize, bias=True)
            else:
                # no need for any transformation here
                self.o2e = Identity()

    def forward(self, input):
        """Compute scores from inputs.

        :param input: (bsz x seq_len x num_directions * hiddensize) tensor of
                       states, e.g. the output states of an RNN

        :returns: (bsz x seqlen x num_cands) scores for each candidate
        """
        # next compute scores over dictionary
        if self.numsoftmax > 1:
            bsz = input.size(0)
            seqlen = input.size(1) if input.dim() > 1 else 1

            # first compute different softmax scores based on input vec
            # hsz => numsoftmax * esz
            latent = self.latent(input)
            active = self.dropout(self.activation(latent))
            # esz => num_features
            logit = self.e2s(active.view(-1, self.esz))

            # calculate priors: distribution over which softmax scores to use
            # hsz => numsoftmax
            prior_logit = self.prior(input).view(-1, self.numsoftmax)
            # softmax over numsoftmax's
            prior = self.softmax(prior_logit)

            # now combine priors with logits
            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            # hsz => esz, good time for dropout
            e = self.dropout(self.o2e(input))
            # esz => num_features
            scores = self.e2s(e)

        if self.padding_idx >= 0:
            scores[:, :, self.padding_idx] = -NEAR_INF

        return scores


class AttentionLayer(nn.Module):
    """Computes attention between hidden and encoder states.

    See arxiv.org/abs/1508.04025 for more info on each attention type.
    """

    def __init__(self, attn_type, hiddensize, embeddingsize,
                 bidirectional=False, attn_length=-1, attn_time='pre'):
        """Initialize attention layer."""
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hiddensize
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = embeddingsize
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')

            # linear layer for combining applied attention weights with input
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, attn_params):
        """Compute attention over attn_params given input and hidden states.

        :param xes:         input state. will be combined with applied
                            attention.
        :param hidden:      hidden state from model. will be used to select
                            states to attend to in from the attn_params.
        :param attn_params: tuple of encoder output states and a mask showing
                            which input indices are nonzero.

        :returns: output, attn_weights
                  output is a new state of same size as input state `xes`.
                  attn_weights are the weights given to each state in the
                  encoder outputs.
        """
        global attn_w_premask
        if self.attention == 'none':
            # do nothing, no attention
            return xes, None

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        enc_out, attn_mask = attn_params
        bsz, seqlen, hszXnumdir = enc_out.size()
        numlayersXnumdir = last_hidden.size(1)

        if self.attention == 'local':
            # local attention weights aren't based on encoder states
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)

            # adjust state sizes to the fixed window size
            if seqlen > self.max_length:
                offset = seqlen - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
                seqlen = self.max_length
            if attn_weights.size(1) > seqlen:
                attn_weights = attn_weights.narrow(1, 0, seqlen)
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                # concat hidden state and encoder outputs
                hid = hid.expand(bsz, seqlen, numlayersXnumdir)
                h_merged = torch.cat((enc_out, hid), 2)
                # then do linear combination of them with activation
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                # dot product between hidden and encoder outputs
                if numlayersXnumdir != hszXnumdir:
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)
            elif self.attention == 'general':
                # before doing dot product, transform hidden state with linear
                # same as dot if linear is identity
                hid = self.attn(hid)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)

            # calculate activation scores, apply mask if needed
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask.masked_fill_(~attn_mask, -NEAR_INF)
            attn_weights = F.softmax(attn_w_premask, dim=1)

        # apply the attention weights to the encoder states
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        # concatenate the input and encoder states
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        # combine them with a linear layer and tanh activation
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))

        return output, attn_weights, attn_applied


class LatentVariation(nn.Module):
    def __init__(self, input_size, latent_size, hidden_sizes=(512,), use_cuda=False, dropout=0.1):
        super(LatentVariation, self).__init__()
        self.use_cuda = use_cuda

        self.input_transformer = FeedForward(input_size, input_size, hidden_sizes=hidden_sizes,
                                             dropout=dropout)
        self.latent_size = latent_size
        self.context_to_mean = nn.Linear(input_size, latent_size)
        self.context_to_logvar = nn.Linear(input_size, latent_size)

    def forward(self, input_):
        bsz = input_.size(0)

        context = self.input_transformer(input_)
        mean = self.context_to_mean(context)
        logvar = self.context_to_logvar(context)
        std = torch.exp(0.5 * logvar)

        z = torch.randn([bsz, self.latent_size])
        z = z.cuda() if self.use_cuda and mean.is_cuda else z
        z = z * std + mean

        return z, mean, logvar


def texts_to_bow(texts, vocab_size, special_token_idxs=None):
    bows = []
    if type(texts) is torch.Tensor:
        texts = texts.tolist()
    for sentence in texts:
        bow = Counter(sentence)
        # Remove special tokens
        if special_token_idxs is not None:
            for idx in special_token_idxs:
                bow[idx] = 0

        x = np.zeros(vocab_size, dtype=np.int64)
        x[list(bow.keys())] = list(bow.values())
        bows.append(torch.FloatTensor(x).unsqueeze(dim=0))
    bows = torch.cat(bows, dim=0)
    return bows


def transform_texts(texts, old_dict, new_dict):
    # convert texts into new_texts using new_dict
    new_texts = []
    for sentence in texts.tolist():
        new_sentence = []
        for tok_id in sentence:
            new_tok_id = new_dict[old_dict[int(tok_id)]]
            new_sentence.append(new_tok_id)
        new_texts.append(new_sentence)
    return new_texts


class TopicIndicator(nn.Module):
    def __init__(self, num_topics, topic_dict, global_dict, embedding_size, dropout=0.1,
                 latent_size=128, bow_hiddensizes=(512, 128, 64), special_tokens=(0, 1, 2, 3),
                 use_cuda=True):
        super(TopicIndicator, self).__init__()
        self.num_topics = num_topics
        self.topic_dict = topic_dict
        self.global_dict = global_dict
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.special_tokens = special_tokens
        self.use_cuda = use_cuda
        self.gx = FeedForward(latent_size, self.num_topics, dropout=dropout)
        self.prior_latent_variation = LatentVariation(len(topic_dict), self.latent_size,
                                                      hidden_sizes=bow_hiddensizes,
                                                      use_cuda=use_cuda, dropout=dropout)
        self.post_latent_variation = LatentVariation(len(topic_dict), self.latent_size,
                                                     hidden_sizes=bow_hiddensizes,
                                                     use_cuda=use_cuda, dropout=dropout)
        self.Domain_Matrix = nn.Parameter(torch.Tensor(self.num_topics, embedding_size))
        self.Word_Matrix = nn.Parameter(torch.Tensor(len(self.topic_dict), embedding_size))

        small_val = self.to_cuda(torch.zeros(self.num_topics))
        init.constant_(small_val, 1e-5)
        self.small_diag = torch.diag(small_val)
        self.reset_parameters()

    def to_bow(self, texts):
        topic_texts = transform_texts(texts, self.global_dict, self.topic_dict)
        bows = texts_to_bow(topic_texts, len(self.topic_dict), self.special_tokens)
        if self.use_cuda:
            bows = bows.cuda()
        return bows

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        self.Domain_Matrix.data.uniform_(-stdv, stdv)
        self.Word_Matrix.data.uniform_(-stdv, stdv)

    def normal_kl_div(self, mean1, logvar1, mean2=None,
                      logvar2=None):
        if mean2 is None:
            mean2 = torch.FloatTensor([0.0]).unsqueeze(dim=1).expand(mean1.size(0), mean1.size(1))
            if self.use_cuda:
                mean2 = mean2.cuda()
        if logvar2 is None:
            logvar2 = torch.FloatTensor([0.0]).unsqueeze(dim=1).expand(logvar1.size(0), logvar1.size(1))
            if self.use_cuda:
                logvar2 = logvar2.cuda()
        kl_div = 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mean1 - mean2).pow(2)) / torch.exp(logvar2) - 1.0,
            dim=1
        ).mean().squeeze()
        return kl_div

    def to_cuda(self, tensor):
        if self.use_cuda:
            return tensor.cuda()
        else:
            return tensor

    def topic_diversity_regularisation(self):
        prod = torch.mm(self.Domain_Matrix, self.Domain_Matrix.t())
        diag_sqrt = torch.sqrt(torch.diag(prod)).unsqueeze(dim=1)
        norm_dot = torch.mm(diag_sqrt, diag_sqrt.t())
        # arccos = torch.acos(torch.tanh(-3.5 + ((torch.abs(prod) / norm_dot + 1) / 2.0) * 7.5))
        arccos = torch.acos(torch.abs(prod) / norm_dot - self.small_diag)
        mean = torch.sum(arccos) / (self.num_topics * self.num_topics)
        var = torch.sum(torch.pow(arccos - mean, 2)) / (self.num_topics * self.num_topics)
        # As loss, mean is encouraged to be larger, and the var is suppressed to be smaller so that
        # all of the topics will be pushed away from each other in the topic semantic space
        return var - mean

    def forward(self, input_xs, input_ys=None):
        xs_bow = self.to_bow(input_xs)
        prior_latent_z, prior_mean, prior_logvar = self.prior_latent_variation(xs_bow)

        if input_ys is not None:
            dx = torch.cat([input_xs, input_ys], dim=-1)
            dx_bow = self.to_bow(dx)
            post_latent_z, post_mean, post_logvar = self.post_latent_variation(dx_bow)
            kl_loss = self.normal_kl_div(post_mean, post_logvar, prior_mean, prior_logvar)
            latent_z = post_latent_z
        else:
            kl_loss = None
            latent_z = prior_latent_z
        gx_out = self.gx(latent_z)
        gx_out = F.layer_norm(gx_out, (self.num_topics,))
        theta = F.softmax(gx_out, dim=1)
        theta_x_beta = torch.mm(
            F.layer_norm(torch.mm(theta, self.Domain_Matrix), (self.embedding_size,)),
            F.layer_norm(self.Word_Matrix, (self.embedding_size,)).t()
        )
        return theta, kl_loss, theta_x_beta, latent_z


class ContextEncoder(nn.Module):
    """Context Encoder."""

    def __init__(self, num_features, embeddingsize, hiddensize,
                 padding_idx=0, numlayers=2, dropout=0.1,
                 bidirectional=False, unknown_idx=None,
                 input_dropout=0, adaptive_input_size=16,
                 adaptive_hidden_size=128, num_topics=3,
                 ensemble_factors=128):
        super().__init__()
        self.numlayers = numlayers
        self.dirs = 2 if bidirectional else 1
        self.hiddensize = hiddensize
        self.encoder = RNNEncoder(
            num_features, embeddingsize, hiddensize,
            padding_idx=padding_idx, rnn_class=BN_LSTM,
            numlayers=numlayers, dropout=dropout,
            bidirectional=bidirectional, unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            adaptive_input_size=adaptive_input_size,
            adaptive_hidden_size=adaptive_hidden_size,
            num_topics=num_topics,
            ensemble_factors=ensemble_factors
        )
        # self.attention = AttentionLayer(attn_type='general',
        #                                 hiddensize=hiddensize,
        #                                 embeddingsize=embeddingsize,
        #                                 bidirectional=bidirectional,
        #                                 attn_length=48,
        #                                 attn_time='post')

    def forward(self, input_):
        """
        Transform the input_ into its hidden representation.
        :param input_: (bsz, seq_len)
        :return: input_hidden: (bsz, hiddensize)
        """
        bsz = input_.size(0)

        # with torch.no_grad():
        #     theta, _, _ = self.topic_indicator(input_)
        enc_output, enc_hidden, _ = self.encoder(input_)
        hidden = _transpose_hidden_state(enc_hidden)[0]  # (num_layers * dirs) * bsz * hiddensize
        hidden = hidden.view(self.numlayers, bsz, self.hiddensize)
        input_hidden = hidden[-1]
        # return self.encoder(input_)
        return input_hidden
