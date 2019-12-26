#!/usr/bin/env python3

import copy
import datetime
import json
import math
import os
import random
from collections import namedtuple, Counter
from operator import attrgetter

import torch
import torch.nn.functional as F
from nltk.util import bigrams, trigrams

from parlai.agents.AdaND.criterions import LabelSmoothing, CrossEntropyLabelSmoothing
from parlai.agents.AdaND.embedding_metrics import sentence_average_score, sentence_greedy_score, sentence_extrema_score
from parlai.agents.AdaND.modules import AdaND, opt_to_kwargs
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import (
    AttrDict, argsort, padded_tensor, warn_once, round_sigfigs, neginf
)


class Beam(object):
    """Generic beam class. It keeps information about beam_size hypothesis."""

    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1,
                 eos_token=2, min_n_best=3, cuda='cpu', block_ngram=0):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param min_length:
            minimum length of the predicted sequence
        :param padding_token:
            Set to 0 as usual in ParlAI
        :param bos_token:
            Set to 1 as usual in ParlAI
        :param eos_token:
            Set to 2 as usual in ParlAI
        :param min_n_best:
            Beam will not be done unless this amount of finished hypothesis
            (with EOS) is done
        :param cuda:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        # recent score for each hypo in the beam
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(
            self.device)
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [torch.Tensor(self.beam_size).long()
                            .fill_(self.bos).to(self.device)]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.HypothesisTail = namedtuple(
            'HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.block_ngram = block_ngram
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    @staticmethod
    def find_ngrams(input_list, n):
        """Get list of ngrams with context length n-1"""
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_output_from_current_step(self):
        """Get the outputput at the current step."""
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """Get the backtrack at the current step."""
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        """Advance the beam one step."""
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = neginf(softmax_probs.dtype)
        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            # we need to sum up hypo scores and curr softmax scores before topk
            # [beam_size, voc_size]
            beam_scores = (softmax_probs +
                           self.scores.unsqueeze(1).expand_as(softmax_probs))
            for i in range(self.outputs[-1].size(0)):
                if self.block_ngram > 0:
                    current_hypo = self.partial_hyps[i][1:]
                    current_ngrams = []
                    for ng in range(self.block_ngram):
                        ngrams = Beam.find_ngrams(current_hypo, ng)
                        if len(ngrams) > 0:
                            current_ngrams.extend(ngrams)
                    counted_ngrams = Counter(current_ngrams)
                    if any(v > 1 for k, v in counted_ngrams.items()):
                        # block this hypothesis hard
                        beam_scores[i] = neginf(softmax_probs.dtype)

                #  if previous output hypo token had eos
                # we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    # beam_scores[i] is voc_size array for i-th hypo
                    beam_scores[i] = neginf(softmax_probs.dtype)

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(
                flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] +
                             [tok_ids[i].item()] for i in range(self.beam_size)]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                              hypid=hypid,
                                              score=self.scores[hypid],
                                              tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        """Return whether beam search is complete."""
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (self.get_hyp_from_finished(top_hypothesis_tail),
                top_hypothesis_tail.score)

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs)-1

        :param hyp_id:
            id with range up to beam_size-1

        :return:
            hypothesis sequence
        """
        assert (self.outputs[hypothesis_tail.timestep]
                [hypothesis_tail.hypid] == self.eos)
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(
                timestep=i, hypid=endback, score=self.all_scores[i][endback],
                tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        """Return prettier version of the hypotheses."""
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses in rescored order.

        :param n_best:
            how many n best hypothesis to return

        :return:
            list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(self.HypothesisTail(
                timestep=finished_item.timestep, hypid=finished_item.hypid,
                score=finished_item.score / length_penalty,
                tokenid=finished_item.tokenid))

        srted = sorted(rescored_finished, key=attrgetter('score'),
                       reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        Check if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # to pass assert in L102, it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                          hypid=0,
                                          score=self.all_scores[-1][0],
                                          tokenid=self.outputs[-1][0])

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Create pydot graph representation of the beam.

        :param outputs:
            self.outputs from the beam

        :param dictionary:
            tok 2 word dict to save words in the tree nodes

        :returns:
            pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue',
                         'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(
                hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid,
                                                score=all_scores[tstep][hypid],
                                                tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                        "<{}".format(dictionary.vec2txt([token])
                                     if dictionary is not None else token) +
                        " : " +
                        "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3))

                graph.add_node(pydot.Node(
                    node_tail.__repr__(), label=label, fillcolor=color,
                    style='filled',
                    xlabel='{}'.format(rank) if rank is not None else ''))

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep, hypid=prev_id,
                        score=all_scores[revtstep][prev_id],
                        tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep + 1, hypid=i,
                        score=all_scores[revtstep + 1][i],
                        tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph


class Batch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order.
    """

    def __init__(self, text_vec=None, text_lengths=None,
                 label_vec=None, label_lengths=None, labels=None,
                 valid_indices=None,
                 candidates=None, candidate_vecs=None,
                 image=None, observations=None,
                 lda_theta=None,
                 **kwargs):
        super().__init__(
            text_vec=text_vec, text_lengths=text_lengths,
            label_vec=label_vec, label_lengths=label_lengths, labels=labels,
            valid_indices=valid_indices,
            candidates=candidates, candidate_vecs=candidate_vecs,
            image=image, observations=observations,
            lda_theta=lda_theta,
            **kwargs)


def anneal_weight(step):
    return (math.tanh((step - 3500) / 1000) + 1) / 2


class MyDictionaryAgent(DictionaryAgent):
    def remove_tokens(self, tokens):
        for token in tokens:
            if token in self.tok2ind:
                del self.freq[token]
                idx = self.tok2ind.pop(token)
                del self.ind2tok[idx]


class AdaNDAgent(TorchGeneratorAgent):

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return MyDictionaryAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Adaptive Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot',
                                    'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')
        agent.add_argument('--weight_decay', type=float, default=1e-7)

        # ---------------------- For adaptive learning ------------------------#
        agent.add_argument('-rnn', '--rnn-class', default='basic_lstm',
                           choices=AdaND.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('--adaptive_input_size', type=int, default=32)
        agent.add_argument('--adaptive_hidden_size', type=int, default=128)

        # ---------------------- For logging ----------------------------------#
        agent.add_argument('--report_freq', type=float, default=0.0)

        # ---------------------- For topic adaptive ---------------------------#
        agent.add_argument('--num_topics', type=int, default=3)
        agent.add_argument('--topic_dict', type=str, default=None)
        agent.add_argument('--latent_size', type=int, default=128)
        agent.add_argument('--bow_hiddensizes', type=str, default='512, 128, 64',
                           help='MLP hidden sizes of layers used in the latent topic indicator.')
        agent.add_argument('--stopwords', type=str, default=None)
        agent.add_argument('--topic_maxtokens', type=int, default=5000)
        agent.add_argument('--ensemble_factors', type=int, default=128)

        agent.add_argument('--eval_embedding_type', type=str, default='glove',
                           help='Embedding type (or embedding file) for response evaluation.')

        # ---------------------- For label smoothing --------------------------#
        agent.add_argument('--label_smoothing', type=float, default=0.0)

        # ---------------------- For topic indicator --------------------------#
        agent.add_argument('--kl_loss_w', type=float, default=1)
        agent.add_argument('--diver_reg_w', type=float, default=0.1)

        # ---------------------- For outside lda topic distributions ----------#
        agent.add_argument('--use_outside_lda_theta', type='bool', default=False)
        agent.add_argument('--outside_lda_num_topics', type=int, default=5)

        agent.add_argument('--interactive_mode', type='bool', default=False)

        super(AdaNDAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions.
        Version 1 split from version 0 on Aug 29, 2018.
        Version 2 split from version 1 on Nov 13, 2018
        To use version 0, use --model legacy:seq2seq:0
        To use version 1, use --model legacy:seq2seq:1
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def _init_eval_embedding(self, embedding_type=None):
        if embedding_type is None:
            embedding_type = 'glove'
        print('[ Init {} embeddings for evaluation ]'.format(embedding_type))
        embs, _ = self._get_embtype(embedding_type)
        self.eval_embs = embs

    def share(self):
        shared = super().share()
        shared['eval_embs'] = self.eval_embs

        return shared

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'AdaND'
        if shared:
            self.eval_embs = shared['eval_embs']
        else:
            # Add the metrics for distinct evaluations
            self.add_metric('total_unigram_cnt', 0)
            self.add_metric('total_bigram_cnt', 0)
            self.add_metric('total_trigram_cnt', 0)
            self.add_metric('dist_unigram_tokens', set())
            self.add_metric('dist_bigram_tokens', set())
            self.add_metric('dist_trigram_tokens', set())

            self.add_metric('rec_loss', 0.0)
            self.add_metric('rec_loss_cnt', 0)
            self.add_metric('kl_loss', 0.0)
            self.add_metric('kl_loss_cnt', 0)
            self.add_metric('diver_reg_loss_cnt', 0)
            self.add_metric('diver_reg_loss', 0.0)

            self.add_metric('embed_avg_cnt', 0)
            self.add_metric('embed_avg', 0.0)
            self.add_metric('embed_greedy_cnt', 0)
            self.add_metric('embed_greedy', 0.0)
            self.add_metric('embed_extrema_cnt', 0)
            self.add_metric('embed_extrema', 0.0)
            self.add_metric('process_time', 0.0)
            self.add_metric('process_tokens', 0)

            self._init_eval_embedding(embedding_type=opt.get('eval_embedding_type'))

        self.reset()

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['total_unigram_cnt'] = 0
        self.metrics['total_bigram_cnt'] = 0
        self.metrics['total_trigram_cnt'] = 0
        self.metrics['dist_unigram_tokens'] = set()
        self.metrics['dist_bigram_tokens'] = set()
        self.metrics['dist_trigram_tokens'] = set()
        self.metrics['rec_loss'] = 0.0
        self.metrics['rec_loss_cnt'] = 0
        self.metrics['kl_loss'] = 0.0
        self.metrics['kl_loss_cnt'] = 0
        self.metrics['diver_reg_loss_cnt'] = 0
        self.metrics['diver_reg_loss'] = 0.0
        self.metrics['embed_avg_cnt'] = 0
        self.metrics['embed_avg'] = 0.0
        self.metrics['embed_greedy_cnt'] = 0
        self.metrics['embed_greedy'] = 0.0
        self.metrics['embed_extrema_cnt'] = 0
        self.metrics['embed_extrema'] = 0.0
        self.metrics['process_time'] = 0.0
        self.metrics['process_tokens'] = 0

    def report(self):
        base = super().report()
        m = dict()

        if self.metrics['total_unigram_cnt'] > 0:
            m['dist_1_cnt'] = len(self.metrics['dist_unigram_tokens'])
            m['dist_1_ratio'] = m['dist_1_cnt'] / self.metrics['total_unigram_cnt']

        if self.metrics['total_bigram_cnt'] > 0:
            m['dist_2_cnt'] = len(self.metrics['dist_bigram_tokens'])
            m['dist_2_ratio'] = m['dist_2_cnt'] / self.metrics['total_bigram_cnt']

        if self.metrics['total_trigram_cnt'] > 0:
            m['dist_3_cnt'] = len(self.metrics['dist_trigram_tokens'])
            m['dist_3_ratio'] = m['dist_3_cnt'] / self.metrics['total_trigram_cnt']

        if self.metrics['rec_loss_cnt'] > 0:
            m['rec_loss'] = self.metrics['rec_loss'] / self.metrics['rec_loss_cnt']

        if self.metrics['kl_loss_cnt'] > 0:
            m['kl_loss'] = self.metrics['kl_loss'] / self.metrics['kl_loss_cnt']
        if self.metrics['diver_reg_loss_cnt'] > 0:
            m['diver_reg_loss'] = self.metrics['diver_reg_loss'] / self.metrics['diver_reg_loss_cnt']

        if self.metrics['embed_avg_cnt'] > 0:
            m['embed_avg'] = self.metrics['embed_avg'] / self.metrics['embed_avg_cnt']
        if self.metrics['embed_extrema_cnt'] > 0:
            m['embed_extrema'] = self.metrics['embed_extrema'] / self.metrics['embed_extrema_cnt']
        if self.metrics['embed_greedy_cnt'] > 0:
            m['embed_greedy'] = self.metrics['embed_greedy'] / self.metrics['embed_greedy_cnt']

        m['process_time'] = self.metrics['process_time']
        m['process_tokens'] = self.metrics['process_tokens']
        if self.metrics['process_time'] > 0:
            m['tokens_per_ms'] = self.metrics['process_tokens'] / self.metrics['process_time']

        if not self.model.training:
            m['total_metric'] = \
                -base.get('ppl', 0) * 0.25 + \
                (m.get('dist_1_ratio', 0) + m.get('dist_2_ratio', 0) + m.get('dist_3_ratio', 0)) * 100 + \
                (m.get('embed_avg', 0) + m.get('embed_greedy', 0) + m.get('embed_greedy', 0)) * 10
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def build_criterion(self):
        # set up criteria
        smoothing = self.opt.get('label_smoothing', 0.0)
        assert 0.0 <= smoothing < 1.0, '[ label smoothing value must lie in [0, 1) ! ]'
        if self.opt.get('numsoftmax', 1) > 1:
            # self.criterion = nn.NLLLoss(
            #     ignore_index=self.NULL_IDX, reduction='sum')
            criterion = LabelSmoothing(len(self.dict), self.NULL_IDX, smoothing)
        else:
            # self.criterion = nn.CrossEntropyLoss(
            #     ignore_index=self.NULL_IDX, reduction='sum')
            criterion = CrossEntropyLabelSmoothing(len(self.dict), self.NULL_IDX, smoothing)

        return criterion

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr, 'weight_decay': opt['weight_decay']}
        if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = opt['momentum']
            if opt['optimizer'] == 'sgd' and opt.get('nesterov', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = opt.get('nesterov', True)
            elif opt['optimizer'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = opt.get('nus', (0.7,))[0]
        elif opt['optimizer'] == 'adam':
            # turn on amsgrad for adam
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif opt['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus', (0.7, 1.0))
        if opt['optimizer'] in ['adam', 'sparseadam', 'adamax', 'qhadam']:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas', (0.9, 0.999))

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)
        if optim_states:
            if saved_optim_type != opt['optimizer']:
                print('WARNING: not loading optim state since optim class '
                      'changed.')
            else:
                try:
                    self.optimizer.load_state_dict(optim_states)
                except ValueError:
                    print('WARNING: not loading optim state since model '
                          'params changed.')
                if self.use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

    def add_metric(self, metric_name: str, default_value):
        assert self.metrics is not None, 'The metrics is not initialized!'
        assert type(metric_name) == str, 'metric_name should be a string!'
        self.metrics[metric_name] = default_value

    def load_stopwords(self):
        if self.opt['stopwords'] is None:
            from nltk.corpus import stopwords
            sw = set(stopwords.words('english'))
        else:
            sw_file = self.opt['stopwords']
            assert os.path.isfile(sw_file), '{} does not exist!'.format(sw_file)
            with open(sw_file) as f:
                sw = set([line.strip() for line in f.readlines()])
        return sw

    def build_topic_dict(self):
        topic_dict_file = self.opt['topic_dict']
        stopwords = self.load_stopwords()
        if topic_dict_file is not None:
            assert os.path.isfile(topic_dict_file), '{} does not exist!'.format(topic_dict_file)
            topic_dict = self.dictionary_class()({'dict_file': topic_dict_file})
            topic_dict.remove_tokens(stopwords)
            topic_dict.sort(trim=False)
        else:
            topic_dict = copy.deepcopy(self.dict)
            topic_dict.maxtokens = self.opt['topic_maxtokens']
            # remove the stopwords from topic_dict
            topic_dict.remove_tokens(stopwords)
            topic_dict.sort(trim=True)
        return topic_dict

    def build_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        bow_hiddensizes = tuple([int(item.strip()) for item in opt['bow_hiddensizes'].split(',')])
        model = AdaND(
            len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            end_idx=self.END_IDX, unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            adaptive_input_size=opt['adaptive_input_size'],
            adaptive_hidden_size=opt['adaptive_hidden_size'],
            num_topics=opt['num_topics'],
            topic_dict=self.build_topic_dict(),
            global_dict=self.dict,
            latent_size=opt['latent_size'],
            bow_hiddensizes=bow_hiddensizes,
            ensemble_factors=opt['ensemble_factors'],
            use_cuda=self.use_cuda,
            use_outside_lda_theta=opt.get('use_outside_lda_theta', False),
            outside_lda_num_topics=opt.get('outside_lda_num_topics', 5),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._init_topic_words_embeddings(model)
            self._copy_embeddings(model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)

        if states:
            # set loaded states if applicable
            model.load_state_dict(states['model'], strict=False)
            if 'longest_label' in states:
                model.longest_label = states['longest_label']

        if opt['embedding_type'].endswith('fixed'):
            print('AdaND: fixing embedding weights.')
            # self.model.topic_indicator.Word_Matrix.requires_grad = False
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                model.decoder.e2s.weight.requires_grad = False

        return model

    def _init_topic_words_embeddings(self, model):
        embs, name = self._get_embtype(self.opt['embedding_type'])
        cnt = 0
        for w, i in model.topic_indicator.topic_dict.tok2ind.items():
            if w in embs.stoi:
                vec = self._project_vec(embs.vectors[embs.stoi[w]],
                                        model.topic_indicator.Word_Matrix.size(1))
                model.topic_indicator.Word_Matrix.data[i] = vec
                cnt += 1

        print('Initialized topic words embeddings for {} tokens ({}%) from {}.'
              ''.format(cnt, round(cnt * 100 / len(model.topic_indicator.topic_dict), 1), name))

    def _get_embtype(self, emb_type):
        # set up preinitialized embeddings
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        pretrained_dim = 300
        if emb_type.startswith('glove'):
            if 'twitter' in emb_type:
                init = 'glove-twitter'
                name = 'twitter.27B'
                pretrained_dim = 200
            else:
                init = 'glove'
                name = '840B'
            embs = vocab.GloVe(
                name=name, dim=pretrained_dim,
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:glove_vectors'))
        elif emb_type.startswith('fasttext_cc'):
            init = 'fasttext_cc'
            embs = vocab.FastText(
                language='en',
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_cc_vectors'))
        elif emb_type.startswith('fasttext'):
            init = 'fasttext'
            embs = vocab.FastText(
                language='en',
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_vectors'))
        else:
            # emb_type does not matching any type embeddings list above,
            # so we think it is a file_path to the embedding file,
            # if not, raise error
            assert os.path.isfile(emb_type), \
                'emb_type does not matching any type embeddings list above, ' \
                'so we think it is a file_path to the embedding file!'
            init = os.path.basename(emb_type)
            cache = '.vector_cache'
            if not os.path.exists(cache):
                os.makedirs(cache)
            embs = vocab.Vectors(emb_type, cache=cache)
        return embs, init

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def compute_topics_with_top_words(self, k):
        topic_words_matrix = torch.mm(self.model.topic_indicator.Domain_Matrix,
                                      self.model.topic_indicator.Word_Matrix.t())
        topic_with_topk_words = torch.topk(topic_words_matrix, k=k, dim=1)[1]
        topic_words = {}
        for i in range(len(topic_with_topk_words)):
            _vec = topic_with_topk_words[i]
            if hasattr(_vec, 'cpu'):
                _vec = _vec.cpu()
            topic_words['topic_{}'.format(i)] = self.model.topic_indicator.topic_dict.vec2txt(_vec)
        return topic_words

    def save_topical_word_embeddings(self, path):
        tok2vec = {}
        topical_embeddings = self.model.topic_indicator.Word_Matrix.data
        if hasattr(topical_embeddings, 'cpu'):
            topical_embeddings = topical_embeddings.cpu()
        topical_embeddings = topical_embeddings.tolist()
        for i in range(len(topical_embeddings)):
            tok = self.model.topic_indicator.topic_dict[i]
            vec = topical_embeddings[i]
            tok2vec[tok] = vec
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        with open(path, 'w') as f:
            for tok, vec in tok2vec.items():
                f.write('{}\t{}\n'.format(tok, ' '.join([str(item) for item in vec])))

    def compute_indicator_loss(self, batch: Batch):
        if batch.label_vec is None:
            raise ValueError('Cannot compute indicator loss without a label.')
        _, kl_loss, theta_x_beta, latent_z = self.model.topic_indicator(batch.text_vec, input_ys=batch.label_vec)
        # compute bow_loss
        # _, hidden, *_ = self.model.encoder(batch.text_vec)
        # xs_repr = hidden[0][:, -1, :]  # bsz, hiddensize
        # bow_logits = self.model.bow_project(torch.cat([xs_repr, latent_z], dim=1))  # bsz, num_features
        # target_bow = texts_to_bow(batch.label_vec, len(self.dict), self.model.topic_indicator.special_tokens)

        dx = torch.cat([batch.text_vec, batch.label_vec], dim=-1)
        target_bow = self.model.topic_indicator.to_bow(dx)
        reconstruct_loss = -F.log_softmax(theta_x_beta, dim=1) * target_bow
        reconstruct_loss = torch.sum(reconstruct_loss) / batch.label_vec.size(0)
        diver_reg_loss = self.model.topic_indicator.topic_diversity_regularisation()
        self.metrics['rec_loss'] += reconstruct_loss.item()
        self.metrics['rec_loss_cnt'] += 1
        self.metrics['kl_loss'] += kl_loss.item()
        self.metrics['kl_loss_cnt'] += 1
        self.metrics['diver_reg_loss_cnt'] += 1
        self.metrics['diver_reg_loss'] += diver_reg_loss.item()
        return reconstruct_loss, kl_loss, diver_reg_loss

    def observe(self, observation):
        # Override this method to add additional field into the observation
        # add additional fields here
        obs = observation
        if 'text' in obs:
            original_text = obs['text']
            item_arr = [item.strip() for item in original_text.split('|')]
            obs.force_set('text', item_arr[0])
            if self.opt.get('use_outside_lda_theta', False):
                obs['lda_theta'] = torch.FloatTensor([float(i) for i in item_arr[1].split()])

        return super().observe(obs)

    # noinspection PyTypeChecker
    def batchify(self, obs_batch, sort=False):
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        if any('text_vec' in ex for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = padded_tensor(_xs, self.NULL_IDX, self.use_cuda)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = (labels_avail or
                             any('eval_labels_vec' in ex for ex in exs))

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = padded_tensor(label_vecs, self.NULL_IDX, self.use_cuda)
            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens,
                    descending=True
                )

        # LDA THETA
        lda_theta = None
        if any('lda_theta' in ex for ex in exs):
            lda_theta = [ex.get('lda_theta', torch.FloatTensor(torch.ones(self.opt['outside_lda_num_topics']))).unsqueeze(dim=0) for ex in exs]
            lda_theta = torch.cat(lda_theta, dim=0)
            if self.use_cuda:
                lda_theta = lda_theta.cuda()

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(text_vec=xs, text_lengths=x_lens, label_vec=ys,
                     label_lengths=y_lens, labels=labels,
                     valid_indices=valid_inds, candidates=cands,
                     candidate_vecs=cand_vecs, image=imgs,
                     observations=exs, lda_theta=lda_theta)

    def _model_input(self, batch: Batch):
        """
        Creates the input (x) value for the model. Must return a tuple.
        This will be passed directly into the model via *args, i.e.,

        # >>> model(*_model_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs.
        """
        return batch.text_vec, batch.lda_theta

    def _dummy_batch(self, batchsize, maxlen):
        """
        Creates a dummy batch. This is used to preinitialize the cuda buffer,
        or otherwise force a null backward pass after an OOM.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, 2).long().cuda(),
            lda_theta=torch.ones(batchsize, self.opt['outside_lda_num_topics']).float().cuda()
        )

    def inference(self, input_sents):
        assert type(input_sents) is list and len(input_sents) > 0
        if self.opt['dict_lower']:
            batch_obs = [{'text': text.lower()} for text in input_sents]
        else:
            batch_obs = [{'text': text} for text in input_sents]
        batch_observed_obs = [self.observe(obs) for obs in batch_obs]
        for obs in batch_observed_obs:
            obs['text_vec'] = self._vectorize_text(obs['text'])
        batch = self.batchify(batch_observed_obs)
        eval_output = self.eval_step(batch)
        output = eval_output[0]
        return output

    def compute_loss(self, batch: Batch, return_output=False):
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens
        loss /= target_tokens  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def train_step(self, batch: Batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        # with autograd.detect_anomaly():
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            # with autograd.detect_anomaly():
            loss = self.compute_loss(batch)
            self.metrics['loss'] += loss.item()
            loss.backward()
            self.update_params()

            # if self.opt['rnn_class'] == 'topic_adaptive_lstm' or \
            #         self.opt['rnn_class'] == 'context_topic_adaptive_lstm':
            # training the topic indicator
            self.zero_grad()
            reconstruct_loss, kl_loss, diver_reg_loss = self.compute_indicator_loss(batch)
            indicator_loss = \
                reconstruct_loss + \
                anneal_weight(self._number_training_updates) * kl_loss * self.opt['kl_loss_w'] + \
                diver_reg_loss * self.opt['diver_reg_w']
            indicator_loss.backward()
            self.update_params()

        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def eval_step(self, batch: Batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()
            # calculate indicator loss
            self.compute_indicator_loss(batch)

        preds, theta = None, None
        if self.skip_generation:
            # noinspection PyTypeChecker
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        elif self.beam_size == 1:
            # greedy decode
            timer_begin = datetime.datetime.now()
            _, preds, *_, theta = self.model(*self._model_input(batch), bsz=bsz)
            timer_end = datetime.datetime.now()
            m_secs = (timer_end - timer_begin).total_seconds() * 1000
            self.metrics['process_time'] += m_secs
        elif self.beam_size > 1:
            timer_begin = datetime.datetime.now()
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=3,
            )
            timer_end = datetime.datetime.now()
            m_secs = (timer_end - timer_begin).total_seconds() * 1000
            self.metrics['process_time'] += m_secs
            beam_preds_scores, _, beams, theta = out
            preds, scores = zip(*beam_preds_scores)

            # if self.beam_dot_log is True:
            #     self._write_beam_dots(batch.text_vec, beams)

        if batch.label_vec is not None and not self.skip_generation:
            label_text = [self.remove_null_token(self._v2t(y)) for y in batch.label_vec]
            # we are in the validation mode, print some generated responses for debugging
            for i in range(len(preds)):
                if random.random() > (1 - self.opt['report_freq']):
                    context_text = self._v2t(batch.text_vec[i])
                    context_text = self.remove_null_token(context_text)
                    print('TEXT: ', context_text)
                    print('TARGET: ', self._v2t(batch.label_vec[i]))
                    print('PREDICTION: ', self._v2t(preds[i]), '\n~')
        else:
            label_text = None
        cand_choices = None
        # We do not need the candidates in this agent

        if self.skip_generation:
            return None
        text = [self._v2t(p) for p in preds] if preds is not None else None
        if theta is not None and self.opt['interactive_mode']:
            for i in range(len(text)):
                text[i] = text[i] + '\t' + self._pretty_theta(theta[i])
        return Output(text, cand_choices), label_text

    @staticmethod
    def _pretty_theta(theta):
        if hasattr(theta, 'cpu'):
            theta = theta.cpu()
        return '\t'.join([str(round_sigfigs(item, 4)) for item in theta.tolist()])

    def remove_null_token(self, text):
        return ' '.join(list(filter(lambda x: x != self.dict[self.model.NULL_IDX],
                                    text.split())))

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        batch_size = len(observations)
        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        # check if there are any labels available, if so we will train on them
        is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)

        if is_training:
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back graidients.
                eval_output = self.eval_step(batch)
                if eval_output is not None:
                    output = eval_output[0]
                    self.metrics['process_tokens'] += sum([len(sent.split()) for sent in output.text])
                    label_text = eval_output[1]
                    if label_text is not None:
                        # noinspection PyTypeChecker
                        self._eval_embedding_metrics(output, label_text)
                        self._eval_distinct_metrics(output)
                else:
                    output = None
        if output is None:
            self.replies['batch_reply'] = None
            return batch_reply

        self.match_batch(batch_reply, batch.valid_indices, output)
        self.replies['batch_reply'] = batch_reply
        # self._save_history(observations, batch_reply)  # save model predictions

        return batch_reply

    def _eval_embedding_metrics(self, output, label_text):
        # Evaluation of embedding distance between predictions and labels
        text = output.text
        mean_emb_avg = 0
        mean_emb_greedy = 0
        mean_emb_extrema = 0
        for i in range(len(text)):
            pred_sent = text[i].split()
            target_sent = label_text[i].split()
            emb_avg = sentence_average_score(target_sent, pred_sent, self.eval_embs)  # maybe None

            emb_greedy1 = sentence_greedy_score(target_sent, pred_sent, self.eval_embs)
            emb_greedy2 = sentence_greedy_score(pred_sent, target_sent, self.eval_embs)
            emb_greedy = (emb_greedy1 + emb_greedy2) / 2.0

            emb_extrema = sentence_extrema_score(target_sent, pred_sent, self.eval_embs)  # maybe None
            if emb_avg is not None:
                mean_emb_avg += emb_avg
                self.metrics['embed_avg_cnt'] += 1
                self.metrics['embed_avg'] += emb_avg

            mean_emb_greedy += emb_greedy
            self.metrics['embed_greedy_cnt'] += 1
            self.metrics['embed_greedy'] += emb_greedy

            if emb_extrema is not None:
                mean_emb_extrema += emb_extrema
                self.metrics['embed_extrema_cnt'] += 1
                self.metrics['embed_extrema'] += emb_extrema

    def _eval_distinct_metrics(self, output: Output):
        text = output.text

        for i in range(len(text)):
            pred_sent = text[i]
            unigram_tokens = pred_sent.split()
            bigram_tokens = list(bigrams(unigram_tokens))
            trigram_tokens = list(trigrams(unigram_tokens))

            self.metrics['total_unigram_cnt'] += len(unigram_tokens)
            self.metrics['total_bigram_cnt'] += len(bigram_tokens)
            self.metrics['total_trigram_cnt'] += len(trigram_tokens)
            self.metrics['dist_unigram_tokens'] = set.union(
                self.metrics['dist_unigram_tokens'], set(unigram_tokens)
            )
            self.metrics['dist_bigram_tokens'] = set.union(
                self.metrics['dist_bigram_tokens'], set(bigram_tokens)
            )
            self.metrics['dist_trigram_tokens'] = set.union(
                self.metrics['dist_trigram_tokens'], set(trigram_tokens)
            )

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            if hasattr(self.model, 'module'):
                model['model'] = self.model.module.state_dict()
                model['longest_label'] = self.model.module.longest_label
            else:
                model['model'] = self.model.state_dict()
                model['longest_label'] = self.model.longest_label
            if hasattr(self, 'optimizer'):
                model['optimizer'] = self.optimizer.state_dict()
                model['optimizer_type'] = self.opt['optimizer']

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def beam_search(self, model, batch, beam_size, start=1, end=2,
                    pad=0, min_length=3, min_n_best=5, max_ts=40, block_ngram=0):
        """
        Beam search given the model and Batch

        This function expects to be given a TorchGeneratorModel. Please refer to
        that interface for information.

        :param TorchGeneratorModel model:
            Implements the above interface
        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int start:
            start of sequence token
        :param int end:
            end of sequence token
        :param int pad:
            padding token
        :param int min_length:
            minimum length of the decoded sequence
        :param int min_n_best:
            minimum number of completed hypothesis generated from each beam
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """

        ada_context = batch.text_vec

        theta = None
        if self.opt['rnn_class'] == 'context_adaptive_lstm' or \
                self.opt['rnn_class'] == 'context_svd_adaptive_lstm':
            context = self.model.context_encoder(ada_context),
        elif self.opt['rnn_class'] == 'topic_adaptive_lstm':
            if not self.opt.get('use_outside_lda_theta'):
                with torch.no_grad():
                    theta, *_ = self.model.topic_indicator(batch.text_vec)
            else:
                theta = batch.lda_theta
            context = theta,
        elif self.opt['rnn_class'] == 'context_topic_adaptive_lstm':
            context_repr = self.model.context_encoder(ada_context)
            if not self.opt.get('use_outside_lda_theta', None):
                with torch.no_grad():
                    theta, *_ = self.model.topic_indicator(batch.text_vec)
            else:
                theta = batch.lda_theta
            context = context_repr, theta
        else:
            context = None,

        encoder_states = model.encoder(self._model_input(batch)[0], *context)
        dev = batch.text_vec.device

        bsz = len(batch.text_lengths)
        beams = [
            Beam(beam_size, min_length=min_length, padding_token=pad,
                 bos_token=start, eos_token=end, min_n_best=min_n_best,
                 cuda=dev, block_ngram=block_ngram)
            for _ in range(bsz)
        ]

        # repeat encoder outputs and decoder inputs
        decoder_input = torch.LongTensor([start]).expand(bsz * beam_size, 1).to(dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        context = list(context)
        context = [item.index_select(0, inds) if item is not None else item for item in context]
        # context = _extend_adaptive_input(context, inds)

        for ts in range(max_ts):
            # exit early if needed
            if all((b.done() for b in beams)):
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state, *context)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [beam_size * i +
                 b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        for b in beams:
            b.check_finished()

        beam_preds_scores = [list(b.get_top_hyp()) for b in beams]
        for pair in beam_preds_scores:
            pair[0] = Beam.get_pretty_hypothesis(pair[0])

        n_best_beams = [b.get_rescored_finished(n_best=min_n_best) for b in beams]
        n_best_beam_preds_scores = []
        for i, beamhyp in enumerate(n_best_beams):
            this_beam = []
            for hyp in beamhyp:
                pred = beams[i].get_pretty_hypothesis(
                    beams[i].get_hyp_from_finished(hyp))
                score = hyp.score
                this_beam.append((pred, score))
            n_best_beam_preds_scores.append(this_beam)

        return beam_preds_scores, n_best_beam_preds_scores, beams, theta


def _extend_adaptive_input(adaptive_input, inds):
    if isinstance(adaptive_input, list):
        return list(map(_extend_adaptive_input, adaptive_input))
    elif isinstance(adaptive_input, tuple):
        return tuple(map(_extend_adaptive_input, adaptive_input))
    elif torch.is_tensor(adaptive_input):
        return adaptive_input.index_select(0, inds)
    elif adaptive_input is None:
        return None
    else:
        raise ValueError("Don't know how to extend {}".format(adaptive_input))
