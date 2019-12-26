"""Implementation of customized LSTM."""
import copy
import numbers
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

from parlai.agents.AdaND.utils import reverse, FeedForward


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @staticmethod
    def compute_layernorm_stats(input_):
        mu = input_.mean(-1, keepdim=True)
        sigma = input_.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    def forward(self, input_):
        # noinspection PyTypeChecker
        mu, sigma = self.compute_layernorm_stats(input_)
        return (input_ - mu) / sigma * self.weight + self.bias


class AdaptiveNorm(nn.Module):

    def __init__(self, adaptive_input_size, adaptive_hidden_size, hidden_size):
        super(AdaptiveNorm, self).__init__()
        self.adaptive_input_size = adaptive_input_size
        self.zw_linear = nn.Linear(adaptive_hidden_size, adaptive_input_size)
        self.alpha_linear = nn.Linear(adaptive_input_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.zw_linear.weight.data, 0.00)
        init.constant_(self.zw_linear.bias.data, 1.0)
        # init.constant_(self.zw_linear.bias.data, 0.00)
        init.constant_(self.alpha_linear.weight.data, 0.1 / self.adaptive_input_size)
        # init.constant_(self.alpha_linear.weight.data, 0.01 / self.adaptive_input_size)

    def forward(self, input_, adaptive_output):
        zw = self.zw_linear(adaptive_output)
        alpha = self.alpha_linear(zw)
        result = input_ * alpha
        return result


class AdaptiveBias(nn.Module):
    def __init__(self, adaptive_input_size, adaptive_hidden_size, hidden_size):
        super(AdaptiveBias, self).__init__()
        self.zb_linear = nn.Linear(adaptive_hidden_size, adaptive_input_size, bias=False)
        self.beta_linear = nn.Linear(adaptive_input_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.zb_linear.weight.data, 0, 0.01)
        init.constant_(self.beta_linear.weight.data, 0.00)

    def forward(self, input_, adaptive_output):
        zb = self.zb_linear(adaptive_output)
        beta = self.beta_linear(zb)
        result = input_ + beta
        return result


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.FloatTensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.FloatTensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.FloatTensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(4, 1)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        init.constant_(self.bias_ih.data, val=0)
        init.constant_(self.bias_hh.data, val=0)

    def forward(self, input_, state, *context):
        # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input_, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0, decompose_layernorm=False):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_ih.data)
        # init.orthogonal_(self.weight_hh.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(4, 1)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

    def forward(self, input_, state, *context):
        # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input_, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = self.dropout_layer(torch.tanh(cellgate))
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class TopicAdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_topics=3, dropout: float = 0,
                 ensemble_factors=128, decompose_layernorm=False, ):
        super(TopicAdaptiveLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_topics = num_topics
        self.ensemble_factors = ensemble_factors
        self.weight_ih_a = Parameter(torch.randn(4 * hidden_size, self.ensemble_factors))
        self.weight_ih_b = Parameter(torch.randn(self.ensemble_factors, num_topics))
        self.weight_ih_c = Parameter(torch.randn(self.ensemble_factors, input_size))
        self.weight_hh_a = Parameter(torch.randn(4 * hidden_size, self.ensemble_factors))
        self.weight_hh_b = Parameter(torch.randn(ensemble_factors, num_topics))
        self.weight_hh_c = Parameter(torch.randn(self.ensemble_factors, hidden_size))

        self.theta_trans_ih = nn.Linear(self.num_topics, self.num_topics)
        self.theta_trans_hh = nn.Linear(self.num_topics, self.num_topics)

        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_ih_a.data)
        # init.orthogonal_(self.weight_ih_b.data)
        init.constant_(self.weight_ih_b.data, 0.1 / self.num_topics)
        init.orthogonal_(self.weight_ih_c.data)

        init.normal_(self.weight_hh_a.data, 0, 0.1)
        # init.orthogonal_(self.weight_hh_a.data)
        # init.normal_(self.weight_hh_b.data, 0, 0.1)
        init.constant_(self.weight_hh_b.data, 0.1 / self.num_topics)
        init.normal_(self.weight_hh_c.data, 0, 0.1)
        # init.orthogonal_(self.weight_hh_c.data)

        init.constant_(self.theta_trans_ih.weight.data, 0.00)
        init.constant_(self.theta_trans_ih.bias.data, 1.0)

        init.constant_(self.theta_trans_hh.weight.data, 0.00)
        init.constant_(self.theta_trans_hh.bias.data, 1.0)

        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(self.num_topics * 4, 1)
        # with torch.no_grad():
        #     self.weight_hh.set_(weight_hh_data)

    def ensemble_operate(self, input_, hx, topic_w):
        # batch_w_ih = torch.mm(topic_w, self.weight_ih.view(self.num_topics, -1)).view(
        #     -1, 4 * self.hidden_size, self.input_size
        # )  # bsz, 4h, input_size
        # igates = torch.bmm(batch_w_ih, input_.view(-1, self.input_size, 1)).view(-1, 4 * self.hidden_size)

        input_x_ih_c_t = torch.mm(input_, self.weight_ih_c.t())
        igates = torch.mm(
            input_x_ih_c_t * torch.mm(self.weight_ih_b, self.theta_trans_ih(topic_w).t()).t(), self.weight_ih_a.t()
        )
        # directly use the topic_w
        # igates = torch.mm(input_x_ih_c_t * topic_w, self.weight_ih_a.t())

        # igates = self.layernorm_i(igates)

        # batch_w_hh = torch.mm(topic_w, self.weight_hh.view(self.num_topics, -1)).view(
        #     -1, 4 * self.hidden_size, self.hidden_size
        # )  # bsz, 4h, h
        # hgates = torch.bmm(batch_w_hh, hx.view(-1, self.hidden_size, 1)).view(-1, 4 * self.hidden_size)

        hx_x_hh_c_t = torch.mm(hx, self.weight_hh_c.t())
        hgates = torch.mm(
            hx_x_hh_c_t * torch.mm(self.weight_hh_b, self.theta_trans_hh(topic_w).t()).t(), self.weight_hh_a.t()
        )
        # hgates = torch.mm(hx_x_hh_c_t * topic_w, self.weight_hh_a.t())
        # hgates = self.layernorm_h(hgates)

        return igates, hgates

    def forward(self, input_, state, *context):
        topic_w = context[0]
        hx, cx = state

        igates, hgates = self.ensemble_operate(input_, hx, topic_w)
        igates = self.layernorm_i(igates)
        hgates = self.layernorm_h(hgates)

        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = self.dropout_layer(torch.tanh(cellgate))
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class AdaptiveLSTMCell2(nn.Module):
    def __init__(self, input_size, hidden_size, adaptive_input_size=16,
                 adaptive_hidden_size=128, dropout: float = 0, decompose_layernorm=False):
        super(AdaptiveLSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adaptive_input_size = adaptive_input_size
        self.adaptive_hidden_size = adaptive_hidden_size
        self.dropout = dropout
        # self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        # self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.adaptive_cell = self.create_adaptive_cell()
        self.adaptive_w_ih1 = nn.Linear(self.adaptive_hidden_size, self.adaptive_input_size)
        self.adaptive_w_ih2 = nn.Linear(self.adaptive_input_size, input_size * 4 * hidden_size)
        self.adaptive_w_hh1 = nn.Linear(self.adaptive_hidden_size, self.adaptive_input_size)
        self.adaptive_w_hh2 = nn.Linear(self.adaptive_input_size, hidden_size * 4 * hidden_size)
        self.reset_parameters()

    def create_adaptive_cell(self):
        return LayerNormLSTMCell(self.input_size + self.hidden_size,
                                 self.adaptive_hidden_size, dropout=self.dropout)

    def reset_parameters(self):
        # init.orthogonal_(self.weight_ih.data)
        # # init.orthogonal_(self.weight_hh.data)
        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(4, 1)
        # with torch.no_grad():
        #     self.weight_hh.set_(weight_hh_data)
        pass

    def adaptive_operate(self, input_, hx, cx, adaptive_output):
        bsz = input_.size(0)
        adaptive_hidden_ih = self.adaptive_w_ih1(adaptive_output)
        m_w_ih = self.adaptive_w_ih2(adaptive_hidden_ih)  # bsz, input_size * 4 * hidden_size
        adaptive_hidden_hh = self.adaptive_w_hh1(adaptive_output)
        m_w_hh = self.adaptive_w_hh2(adaptive_hidden_hh)  # bsz, hidden_size * 4 * hidden_size
        igates = torch.bmm(input_.view(bsz, 1, -1),
                           m_w_ih.view(bsz, -1, 4 * self.hidden_size)).squeeze(dim=1)
        hgates = torch.bmm(hx.view(bsz, 1, -1),
                           m_w_hh.view(bsz, -1, 4 * self.hidden_size)).squeeze(dim=1)

        igates = self.layernorm_i(igates)
        hgates = self.layernorm_h(hgates)
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

    def forward(self, input_, state, *context):
        # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([input_, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class AdaptiveLSTMCell(nn.Module):
    """
    AdaptiveLSTM, with Layer Norm and Recurrent Dropout without Memory Loss.
    https://arxiv.org/abs/1609.09106
    """

    def __init__(self, input_size, hidden_size, adaptive_input_size=16,
                 adaptive_hidden_size=128, dropout=0.1,
                 decompose_layernorm=False,
                 ada_norm=AdaptiveNorm):
        super(AdaptiveLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adaptive_input_size = adaptive_input_size
        self.adaptive_hidden_size = adaptive_hidden_size
        self.dropout = dropout

        self.weight_ih = Parameter(torch.randn(4 * self.hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * self.hidden_size, self.hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * self.hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * self.hidden_size))
        self.adaptive_norm_scopes = ['ix', 'jx', 'fx', 'ox', 'ih', 'jh', 'fh', 'oh']
        self.adaptive_bias_scopes = ['ixb', 'jxb', 'fxb', 'oxb', 'ihb', 'jhb', 'fhb', 'ohb']
        for adaptive_norm in self.adaptive_norm_scopes:
            setattr(self, 'adaptive_norm_' + adaptive_norm, ada_norm(adaptive_input_size, adaptive_hidden_size,
                                                                     self.hidden_size))
        for adaptive_bias in self.adaptive_bias_scopes:
            setattr(self, 'adaptive_bias_' + adaptive_bias, AdaptiveBias(adaptive_input_size, adaptive_hidden_size,
                                                                         self.hidden_size))

        # The layernorms provide learnable biases
        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * self.hidden_size)
        self.layernorm_h = ln(4 * self.hidden_size)
        self.layernorm_c = ln(self.hidden_size)

        self.dropout_layer = nn.Dropout(dropout)
        self.adaptive_cell = self.create_adaptive_cell()
        self.reset_parameters()

    def create_adaptive_cell(self):
        return LayerNormLSTMCell(self.input_size + self.hidden_size,
                                 self.adaptive_hidden_size, dropout=self.dropout)

    def reset_parameters(self):
        if hasattr(self, 'weight_ih'):
            init.orthogonal_(self.weight_ih.data)
        if hasattr(self, 'weight_hh'):
            # init.orthogonal_(self.weight_hh.data)
            weight_hh_data = torch.eye(self.hidden_size)
            weight_hh_data = weight_hh_data.repeat(4, 1)
            with torch.no_grad():
                self.weight_hh.set_(weight_hh_data)
        if hasattr(self, 'bias_ih'):
            init.constant_(self.bias_ih.data, val=0)
        if hasattr(self, 'bias_hh'):
            init.constant_(self.bias_hh.data, val=0)

        # stdv = 0.01 / math.sqrt(self.hidden_size)
        #
        # self.weight_ih.data.uniform_(-stdv, stdv)
        # self.weight_hh.data.uniform_(-stdv, stdv)

    def real_adaptive_operate(self, xh, hh, adaptive_output):
        ix, jx, fx, ox = xh.chunk(4, 1)
        ix = self.adaptive_norm_ix(ix, adaptive_output)
        jx = self.adaptive_norm_jx(jx, adaptive_output)
        fx = self.adaptive_norm_fx(fx, adaptive_output)
        ox = self.adaptive_norm_ox(ox, adaptive_output)

        ih, jh, fh, oh = hh.chunk(4, 1)
        ih = self.adaptive_norm_ih(ih, adaptive_output)
        jh = self.adaptive_norm_jh(jh, adaptive_output)
        fh = self.adaptive_norm_fh(fh, adaptive_output)
        oh = self.adaptive_norm_oh(oh, adaptive_output)

        ixb, jxb, fxb, oxb = self.bias_ih.chunk(4, 0)
        ixb = self.adaptive_bias_ixb(ixb, adaptive_output)
        jxb = self.adaptive_bias_jxb(jxb, adaptive_output)
        fxb = self.adaptive_bias_fxb(fxb, adaptive_output)
        oxb = self.adaptive_bias_oxb(oxb, adaptive_output)

        ihb, jhb, fhb, ohb = self.bias_hh.chunk(4, 0)
        ihb = self.adaptive_bias_ihb(ihb, adaptive_output)
        jhb = self.adaptive_bias_jhb(jhb, adaptive_output)
        fhb = self.adaptive_bias_fhb(fhb, adaptive_output)
        ohb = self.adaptive_bias_ohb(ohb, adaptive_output)

        input_gates_i = ix + ixb
        input_gates_j = jx + jxb
        input_gates_f = fx + fxb
        input_gates_o = ox + oxb

        hidden_gates_i = ih + ihb
        hidden_gates_j = jh + jhb
        hidden_gates_f = fh + fhb
        hidden_gates_o = oh + ohb

        input_gates = torch.cat(
            [input_gates_i, input_gates_j, input_gates_f, input_gates_o], dim=1
        )
        hidden_gates = torch.cat(
            [hidden_gates_i, hidden_gates_j, hidden_gates_f, hidden_gates_o], dim=1
        )

        return input_gates, hidden_gates

    def cell_operate(self, xh, hh, cx):
        input_gates = xh
        hidden_gates = hh
        input_gates = self.layernorm_i(input_gates)
        hidden_gates = self.layernorm_h(hidden_gates)
        gates = input_gates + hidden_gates
        i, j, f, o = gates.chunk(4, 1)

        ingate = torch.sigmoid(i)
        forgetgate = torch.sigmoid(f)
        cellgate = self.dropout_layer(torch.tanh(j))
        outgate = torch.sigmoid(o)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def adaptive_operate(self, input_, hx, cx, adaptive_output):
        xh = torch.mm(input_, self.weight_ih.t())
        hh = torch.mm(hx, self.weight_hh.t())
        input_gates, hidden_gates = self.real_adaptive_operate(xh, hh, adaptive_output)
        return self.cell_operate(input_gates, hidden_gates, cx)

    def forward(self, input_, state, *context):
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([input_, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class ContextAdaptiveLSTMCell(AdaptiveLSTMCell):
    def create_adaptive_cell(self):
        return LayerNormLSTMCell(self.hidden_size + self.hidden_size,
                                 self.adaptive_hidden_size,
                                 dropout=self.dropout)

    def forward(self, input_, state, *context):
        context = context[0]
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([context, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class ContextAdaptiveLSTMCell2(AdaptiveLSTMCell2):
    def create_adaptive_cell(self):
        return LayerNormLSTMCell(self.hidden_size + self.hidden_size, self.adaptive_hidden_size)

    def forward(self, input_, state, *context):
        context = context[0]
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([context, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class SVDAda(nn.Module):
    def __init__(self, svd_ada_factors, input_dim, ada_dim, out_dim):
        super(SVDAda, self).__init__()
        self.svd_ada_factors = svd_ada_factors
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.ada_dim = ada_dim
        self.weight_c = Parameter(torch.randn(self.svd_ada_factors, self.input_dim))
        self.weight_b = Parameter(torch.randn(self.svd_ada_factors, self.ada_dim))
        self.weight_a = Parameter(torch.randn(self.out_dim, self.svd_ada_factors))
        self.layernorm = nn.LayerNorm(self.out_dim)

    def reset_parameters(self):
        init.orthogonal_(self.weight_a.data)
        init.orthogonal_(self.weight_c.data)
        init.constant_(self.weight_b.data, 0.1 / self.ada_dim)

    def forward(self, input_, ada_tensor):
        input_X_c = torch.mm(input_, self.weight_c.t())
        singular_val = torch.mm(self.weight_b, ada_tensor.t()).t()
        svd_output = torch.mm(input_X_c * singular_val, self.weight_a.t())
        svd_output = self.layernorm(svd_output)
        return svd_output


class ContextTopicAdaptiveLSTMCell(ContextAdaptiveLSTMCell):
    def __init__(self, input_size, hidden_size, adaptive_input_size=16,
                 adaptive_hidden_size=128, num_topics=3, ensemble_factors=128,
                 dropout: float = 0, decompose_layernorm=False):
        self.num_topics = num_topics
        self.ensemble_factors = ensemble_factors
        self.adaptive_hidden_size = adaptive_hidden_size
        super().__init__(input_size, hidden_size, adaptive_input_size,
                         self.adaptive_hidden_size, dropout, decompose_layernorm)

        # ------------------------- SVD adaptation using topic_w ----------------- #
        self.weight_ih_a = Parameter(torch.randn(4 * hidden_size, self.ensemble_factors))
        self.weight_ih_b = Parameter(torch.randn(self.ensemble_factors, num_topics))
        self.weight_ih_c = Parameter(torch.randn(self.ensemble_factors, input_size))

        self.weight_hh_a = Parameter(torch.randn(4 * hidden_size, self.ensemble_factors))
        self.weight_hh_b = Parameter(torch.randn(self.ensemble_factors, num_topics))
        self.weight_hh_c = Parameter(torch.randn(self.ensemble_factors, hidden_size))
        self.theta_trans_ih = nn.Linear(self.num_topics, self.num_topics)
        self.theta_trans_hh = nn.Linear(self.num_topics, self.num_topics)
        TopicAdaptiveLSTMCell.reset_parameters(self)

        # ------------------------- SVD adaptation using context ------------------ #
        self.weight_ih_a_c = Parameter(torch.randn(4 * hidden_size, self.adaptive_hidden_size))
        self.weight_ih_c_c = Parameter(torch.randn(self.adaptive_hidden_size, input_size))

        self.weight_hh_a_c = Parameter(torch.randn(4 * hidden_size, self.adaptive_hidden_size))
        self.weight_hh_c_c = Parameter(torch.randn(self.adaptive_hidden_size, hidden_size))

        init.orthogonal_(self.weight_ih_a_c.data)
        init.orthogonal_(self.weight_ih_c_c.data)

        init.normal_(self.weight_hh_a_c.data, 0, 0.1)
        init.normal_(self.weight_hh_c_c.data, 0, 0.1)
        # init.orthogonal_(self.weight_hh_a_c.data)
        # init.orthogonal_(self.weight_hh_c_c.data)

        # For gating combination
        # self.xh_gate = nn.Linear(hidden_size, 4 * hidden_size)
        self.xh_gate_trans_input = nn.Linear(input_size, 4 * hidden_size)
        self.xh_gate_trans_topic = nn.Linear(num_topics, 4 * hidden_size)
        self.xh_gate_trans_context = nn.Linear(self.adaptive_hidden_size, 4 * hidden_size)

        # self.hh_gate = nn.Linear(hidden_size, 4 * hidden_size)
        self.hh_gate_trans_input = nn.Linear(hidden_size, 4 * hidden_size)
        self.hh_gate_trans_topic = nn.Linear(num_topics, 4 * hidden_size)
        self.hh_gate_trans_context = nn.Linear(self.adaptive_hidden_size, 4 * hidden_size)

        # init.constant_(self.xh_gate.weight.data, 0.00)
        # init.constant_(self.xh_gate.bias.data, 1.0)
        # init.constant_(self.hh_gate.weight.data, 0.00)
        # init.constant_(self.hh_gate.bias.data, 1.0)
        # for scope in ['xh', 'hh']:
        #     for name in ['input', 'topic', 'context']:
        #         init.orthogonal_(getattr(self, scope + '_gate_trans_' + name).weight.data)
        #         init.constant_(getattr(self, scope + '_gate_trans_' + name).bias.data, 0)

        self.gate_trans_layernorm = nn.LayerNorm(4 * hidden_size)
        # self.gate_trans_layernorm2 = nn.LayerNorm(4 * hidden_size)
        # self.gate_trans_layernorm3 = nn.LayerNorm(4 * hidden_size)

        # self.res_layernorm = nn.LayerNorm(4 * hidden_size)

        # adaptive gating
        self.xh_gate_trans_svd_ada_topic = SVDAda(ensemble_factors, input_size,
                                                  num_topics, 4 * hidden_size)
        self.hh_gate_trans_svd_ada_topic = SVDAda(ensemble_factors, hidden_size,
                                                  num_topics, 4 * hidden_size)
        self.xh_gate_trans_svd_ada_context = SVDAda(ensemble_factors, input_size,
                                                    adaptive_hidden_size, 4 * hidden_size)
        self.hh_gate_trans_svd_ada_context = SVDAda(ensemble_factors, hidden_size,
                                                    adaptive_hidden_size, 4 * hidden_size)

        # For attention combination
        self.xh_mlp = FeedForward(input_size + 4 * hidden_size, 1, hidden_sizes=(512, 128, 32),
                                  dropout=self.dropout)
        self.hh_mlp = FeedForward(hidden_size + 4 * hidden_size, 1, hidden_sizes=(512, 128, 32),
                                  dropout=self.dropout)

    def svd_ada_gating_fuse(self, topic_res, context_res, input_, adaptive_output, topic_w, mode='xh'):
        # gating
        gate = torch.sigmoid(
            # getattr(self, mode + '_gate')(
            getattr(self, mode + '_gate_trans_svd_ada_topic')(input_, topic_w) +
            getattr(self, mode + '_gate_trans_svd_ada_context')(input_, adaptive_output)
            # )
        )

        fused_res = gate * topic_res + (1 - gate) * context_res
        return fused_res

    def gating_fuse(self, topic_res, context_res, input_, adaptive_output, topic_w, mode='xh'):
        # gating
        gate = torch.sigmoid(
            # getattr(self, mode + '_gate')(
            self.gate_trans_layernorm(getattr(self, mode + '_gate_trans_input')(input_)) +
            self.gate_trans_layernorm(getattr(self, mode + '_gate_trans_topic')(topic_w)) +
            self.gate_trans_layernorm(getattr(self, mode + '_gate_trans_context')(adaptive_output))
            # )
        )

        fused_res = gate * topic_res + (1 - gate) * context_res

        # naive combine
        # fused_res = topic_res + context_res

        # non-linear combine
        # fused_res = topic_res * context_res + topic_res + context_res

        # return self.res_layernorm(fused_res)
        return fused_res

    def attention_fuse(self, topic_res, context_res, input_, adaptive_output, topic_w, mode='xh'):
        input_topic_sim = getattr(self, mode + '_mlp')(torch.cat([input_, topic_res], dim=1))
        input_context_sim = getattr(self, mode + '_mlp')(torch.cat([input_, context_res], dim=1))
        attn_scores = torch.cat([input_topic_sim, input_context_sim], dim=1)  # bsz, 2
        attn_weights = F.softmax(attn_scores, dim=1)  # bsz, 2
        topic_context = torch.stack([topic_res, context_res]).transpose(0, 1)  # bsz, 2, 4h
        fused_res = torch.bmm(attn_weights.view(-1, 1, 2), topic_context).squeeze(1)
        return fused_res

    def adaptive_operate(self, input_, hx, cx, adaptive_output, topic_w=None):
        # ------------------------- SVD adaptation using topic_w ----------------- #
        input_x_ih_c_t = torch.mm(input_, self.weight_ih_c.t())
        adaptive_topic_output_xh = torch.mm(self.weight_ih_b, self.theta_trans_ih(topic_w).t()).t()
        singular_value_xh = adaptive_topic_output_xh  # * adaptive_output + adaptive_topic_output_xh + adaptive_output
        # singular_value_xh = self.dropout_layer(singular_value_xh)
        topic_svd_xh = torch.mm(
            input_x_ih_c_t * singular_value_xh,
            self.weight_ih_a.t()
        )  # bsz, 4*hiddensize

        hx_x_hh_c_t = torch.mm(hx, self.weight_hh_c.t())
        adaptive_topic_output_hh = torch.mm(self.weight_hh_b, self.theta_trans_hh(topic_w).t()).t()
        singular_value_hh = adaptive_topic_output_hh  # * adaptive_output + adaptive_topic_output_hh + adaptive_output
        # singular_value_hh = self.dropout_layer(singular_value_hh)
        topic_svd_hh = torch.mm(
            hx_x_hh_c_t * singular_value_hh,
            self.weight_hh_a.t()
        )

        # ------------------------- SVD adaptation using context ------------------ #
        input_x_ih_c_c_t = torch.mm(input_, self.weight_ih_c_c.t())
        context_svd_xh = torch.mm(
            input_x_ih_c_c_t * adaptive_output, self.weight_ih_a_c.t()
        )

        hx_x_hh_c_c_t = torch.mm(hx, self.weight_hh_c_c.t())
        context_svd_hh = torch.mm(
            hx_x_hh_c_c_t * adaptive_output, self.weight_hh_a_c.t()
        )

        # ------------------------ Right adaptation using context ---------------- #
        # original_xh = torch.mm(input_, self.weight_ih.t())
        # original_hh = torch.mm(hx, self.weight_hh.t())
        # context_right_ada_xh, context_right_ada_hh = self.real_adaptive_operate(
        #     original_xh, original_hh, adaptive_output
        # )

        # gating or attention
        fuse_method = 'gating'
        xh = getattr(self, fuse_method + '_fuse')(
            topic_svd_xh, context_svd_xh, input_, adaptive_output, topic_w)
        hh = getattr(self, fuse_method + '_fuse')(
            topic_svd_hh, context_svd_hh, hx, adaptive_output, topic_w, mode='hh')
        return self.cell_operate(xh, hh, cx)

    def forward(self, input_, state, *context):
        adaptive_context = context[0]
        topic_w = context[1]
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([adaptive_context, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output, topic_w)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class ContextSVDAdaptiveLSTMCell(ContextAdaptiveLSTMCell):
    def __init__(self, input_size, hidden_size, adaptive_input_size=16,
                 adaptive_hidden_size=128, dropout: float = 0, decompose_layernorm=False):
        self.adaptive_hidden_size = adaptive_hidden_size
        super().__init__(input_size, hidden_size, adaptive_input_size,
                         self.adaptive_hidden_size, dropout, decompose_layernorm)

        # ------------------------- SVD adaptation using context ------------------ #
        self.weight_ih_a_c = Parameter(torch.randn(4 * hidden_size, self.adaptive_hidden_size))
        self.weight_ih_c_c = Parameter(torch.randn(self.adaptive_hidden_size, input_size))

        self.weight_hh_a_c = Parameter(torch.randn(4 * hidden_size, self.adaptive_hidden_size))
        self.weight_hh_c_c = Parameter(torch.randn(self.adaptive_hidden_size, hidden_size))

        init.orthogonal_(self.weight_ih_a_c.data)
        init.orthogonal_(self.weight_ih_c_c.data)

        init.normal_(self.weight_hh_a_c.data, 0, 0.1)
        init.normal_(self.weight_hh_c_c.data, 0, 0.1)

    def adaptive_operate(self, input_, hx, cx, adaptive_output):
        # ------------------------- SVD adaptation using context ------------------ #
        input_x_ih_c_c_t = torch.mm(input_, self.weight_ih_c_c.t())
        context_svd_xh = torch.mm(
            input_x_ih_c_c_t * adaptive_output, self.weight_ih_a_c.t()
        )

        hx_x_hh_c_c_t = torch.mm(hx, self.weight_hh_c_c.t())
        context_svd_hh = torch.mm(
            hx_x_hh_c_c_t * adaptive_output, self.weight_hh_a_c.t()
        )

        return self.cell_operate(context_svd_xh, context_svd_hh, cx)

    def forward(self, input_, state, *context):
        adaptive_context = context[0]
        total_h, total_c = state

        hx = total_h[:, 0:self.hidden_size]
        cx = total_c[:, 0:self.hidden_size]

        adaptive_state = (total_h[:, self.hidden_size:], total_c[:, self.hidden_size:])

        adaptive_input = torch.cat([adaptive_context, hx], dim=1)
        adaptive_output, adaptive_new_state = self.adaptive_cell(adaptive_input, adaptive_state)

        hy, cy = self.adaptive_operate(input_, hx, cx, adaptive_output)

        adaptive_h, adaptive_c = adaptive_new_state
        new_total_c = torch.cat([cy, adaptive_c], 1)
        new_total_h = torch.cat([hy, adaptive_h], 1)

        return hy, (new_total_h, new_total_c)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input_, state, *context):
        """
        @:param input_: Tensor in shape (seq_len, bsz, input_size)
        @:param state: Tuple of (Tensor:(bsz, hidden_size), Tensor:(bsz, hidden_size))
        type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        """
        inputs = input_.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, *context)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input_, state, *context):
        """
        @:param input_: Tensor in shape (seq_len, bsz, input_size)
        @:param state: Tuple of (Tensor:(bsz, hidden_size), Tensor:(bsz, hidden_size))
        type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        """
        inputs = reverse(input_.unbind(0))
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, *context)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    def forward(self, input_, states, *context):
        # (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = []
        output_states = []
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input_, state, *context)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, layer, first_layer_args,
                 other_layer_args, dropout=0.2):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_, states, *context):
        # (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = []
        output = input_
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state, *context)
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states


class StackedLSTM2(nn.Module):
    def __init__(self, num_layers, layer, first_layer_args,
                 other_layer_args, dropout=0.2):
        super(StackedLSTM2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_, states, *context):
        # (Tensor, List[List[Tuple[Tensor, Tensor]]])
        # -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]
        # List[List[LSTMState]]: The outer list is for layers, inner list is for directions.
        output_states = []
        output = input_
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state, *context)
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


class LSTMBase(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.2, bidirectional=False):
        super().__init__()
        assert bias, 'Custom LSTM does not support bias==False!'
        assert 0 <= dropout < 1
        if bidirectional:
            stack_type = StackedLSTM2
            layer_type = BidirLSTMLayer
            dirs = 2
        else:
            stack_type = StackedLSTM
            layer_type = LSTMLayer
            dirs = 1
        self.dirs = dirs
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout = dropout
        if num_layers == 1 and dropout > 0:
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")
        self.lstm = self._create_lstm(stack_type, layer_type, cell_type)

    def _create_lstm(self, stack_type, layer_type, cell_type):
        return stack_type(self.num_layers, layer_type,
                          first_layer_args=[cell_type, self.input_size, self.hidden_size],
                          other_layer_args=[cell_type, self.hidden_size * self.dirs, self.hidden_size],
                          dropout=self.dropout)

    def forward(self, input_, states=None, *context):
        is_packed = isinstance(input_, PackedSequence)
        assert not is_packed  # we do not pack the input_
        batch_size = input_.size(0) if self.batch_first else input_.size(1)
        if states is None:
            if self.dirs == 1:
                states = input_.new_zeros(self.num_layers, batch_size,
                                          self.hidden_size, requires_grad=False)
                states = (states, states)
            else:
                states = input_.new_zeros(self.num_layers, self.dirs, batch_size,
                                          self.hidden_size, requires_grad=False)
                states = (states, states)
        h, c = states
        if self.dirs == 2:
            h = h.view(self.num_layers, self.dirs, batch_size, self.hidden_size)
            c = c.view(self.num_layers, self.dirs, batch_size, self.hidden_size)
            states = [[(h[i][0], c[i][0]), (h[i][1], c[i][1])] for i in range(self.num_layers)]
        else:
            h = h.view(self.num_layers, batch_size, self.hidden_size)
            c = c.view(self.num_layers, batch_size, self.hidden_size)
            states = [(h[i], c[i]) for i in range(self.num_layers)]

        if self.batch_first:
            input_ = input_.transpose(0, 1)

        output, output_states = self.lstm(input_, states, *context)
        if self.dirs == 1:
            h_n, c_n = flatten_states(output_states)
        else:
            h_n, c_n = double_flatten_states(output_states)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_n, c_n)


class BN_LSTM(LSTMBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.2, bidirectional=False):
        super().__init__(LayerNormLSTMCell, input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional)

    def _create_lstm(self, stack_type, layer_type, cell_type):
        return stack_type(self.num_layers, layer_type,
                          first_layer_args=[cell_type, self.input_size, self.hidden_size,
                                            self.dropout],
                          other_layer_args=[cell_type, self.hidden_size * self.dirs,
                                            self.hidden_size, self.dropout],
                          dropout=self.dropout)


class Basic_LSTM(LSTMBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.2, bidirectional=False):
        super().__init__(LSTMCell, input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional)


class Adaptive_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128,
                 cell_type=AdaptiveLSTMCell):
        super().__init__()
        assert bias, 'Adaptive_LSTM does not support bias==False!'
        assert 0 <= dropout < 1
        if bidirectional:
            stack_type = StackedLSTM2
            layer_type = BidirLSTMLayer
            dirs = 2
        else:
            stack_type = StackedLSTM
            layer_type = LSTMLayer
            dirs = 1
        self.dirs = dirs
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout = dropout
        self.adaptive_input_size = adaptive_input_size
        self.adaptive_hidden_size = adaptive_hidden_size
        if num_layers == 1 and dropout > 0:
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")
        self.lstm = self._create_lstm(stack_type, layer_type, cell_type)

    def _create_lstm(self, stack_type, layer_type, cell_type):
        return stack_type(self.num_layers, layer_type,
                          first_layer_args=[cell_type, self.input_size, self.hidden_size,
                                            self.adaptive_input_size, self.adaptive_hidden_size, self.dropout],
                          other_layer_args=[cell_type, self.hidden_size * self.dirs, self.hidden_size,
                                            self.adaptive_input_size, self.adaptive_hidden_size, self.dropout],
                          dropout=self.dropout)

    def forward(self, input_, states=None, *context):
        is_packed = isinstance(input_, PackedSequence)
        assert not is_packed  # we do not pack the input_
        batch_size = input_.size(0) if self.batch_first else input_.size(1)
        if states is None:
            if self.dirs == 1:
                states = input_.new_zeros(self.num_layers, batch_size,
                                          self.hidden_size + self.adaptive_hidden_size,
                                          requires_grad=False)
                states = (states, copy.deepcopy(states))

            else:
                states = input_.new_zeros(self.num_layers, self.dirs, batch_size,
                                          self.hidden_size + self.adaptive_hidden_size,
                                          requires_grad=False)
                states = (states, copy.deepcopy(states))

        h, c = states

        if self.dirs == 2:
            if h.size(-1) == self.hidden_size:
                h = h.view(self.num_layers, self.dirs, batch_size, self.hidden_size)
                c = c.view(self.num_layers, self.dirs, batch_size, self.hidden_size)
                # padding the adaptive_init_hidden
                adaptive_h = h.new_zeros(self.num_layers, self.dirs, batch_size,
                                         self.adaptive_hidden_size, requires_grad=False)
                h = torch.cat([h, adaptive_h], -1)
                c = torch.cat([c, copy.deepcopy(adaptive_h)], -1)
            else:
                h = h.view(self.num_layers, self.dirs, batch_size,
                           self.hidden_size + self.adaptive_hidden_size)
                c = c.view(self.num_layers, self.dirs, batch_size,
                           self.hidden_size + self.adaptive_hidden_size)
            states = [[(h[i][0], c[i][0]), (h[i][1], c[i][1])] for i in range(self.num_layers)]
        else:
            if h.size(-1) == self.hidden_size:
                h = h.view(self.num_layers, batch_size, self.hidden_size)
                c = c.view(self.num_layers, batch_size, self.hidden_size)
                adaptive_h = h.new_zeros(self.num_layers, batch_size, self.adaptive_hidden_size,
                                         requires_grad=False)
                h = torch.cat([h, adaptive_h], -1)
                c = torch.cat([c, copy.deepcopy(adaptive_h)], -1)
            else:
                h = h.view(self.num_layers, batch_size, self.hidden_size + self.adaptive_hidden_size)
                c = c.view(self.num_layers, batch_size, self.hidden_size + self.adaptive_hidden_size)
            states = [(h[i], c[i]) for i in range(self.num_layers)]

        if self.batch_first:
            input_ = input_.transpose(0, 1)

        output, output_states = self.lstm(input_, states, *context)
        if self.dirs == 1:
            h_n, c_n = flatten_states(output_states)
        else:
            h_n, c_n = double_flatten_states(output_states)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_n, c_n)


class Adaptive_LSTM2(Adaptive_LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
                         adaptive_input_size, adaptive_hidden_size, cell_type=AdaptiveLSTMCell2)


class Context_Adaptive_LSTM(Adaptive_LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
                         adaptive_input_size, adaptive_hidden_size, cell_type=ContextAdaptiveLSTMCell)

    def forward(self, input_, states=None, *context):
        assert len(context) == 1, '[ There should be only one element in the context. ]'
        assert context[0] is not None, '[ context[0] should not be None in the Context_Adaptive_LSTM! ]'
        assert context[0].size(1) == self.hidden_size, '[ context[0]\'dimension must be hiddensize! ]'
        return super().forward(input_, states, *context)


class Context_SVD_Adaptive_LSTM(Adaptive_LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
                         adaptive_input_size, adaptive_hidden_size, cell_type=ContextSVDAdaptiveLSTMCell)

    def forward(self, input_, states=None, *context):
        assert len(context) == 1, '[ There should be only one element in the context. ]'
        assert context[0] is not None, '[ context[0] should not be None in the Context_SVD_Adaptive_LSTM! ]'
        assert context[0].size(1) == self.hidden_size, '[ context[0]\'dimension must be hiddensize! ]'
        return super().forward(input_, states, *context)


class Context_Adaptive_LSTM2(Adaptive_LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
                         adaptive_input_size, adaptive_hidden_size, cell_type=ContextAdaptiveLSTMCell2)

    def forward(self, input_, states=None, *context):
        assert len(context) == 1, '[ There should be only one element in the context. ]'
        assert context[0] is not None, '[ context[0] should not be None in the Context_Adaptive_LSTM ! ]'
        assert context[0].size(1) == self.hidden_size, '[ context[0]\'dimension must be hiddensize ! ]'
        return super().forward(input_, states, *context)


class Topic_Adaptive_LSTM(LSTMBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.2, bidirectional=False,
                 num_topics=3, ensemble_factors=128):
        self.num_topics = num_topics
        self.ensemble_factors = ensemble_factors
        super().__init__(TopicAdaptiveLSTMCell, input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional)

    def _create_lstm(self, stack_type, layer_type, cell_type):
        return stack_type(self.num_layers, layer_type,
                          first_layer_args=[cell_type, self.input_size, self.hidden_size,
                                            self.num_topics, self.dropout,
                                            self.ensemble_factors],
                          other_layer_args=[cell_type, self.hidden_size * self.dirs,
                                            self.hidden_size, self.num_topics,
                                            self.dropout, self.ensemble_factors],
                          dropout=self.dropout)


class Context_Topic_Adaptive_LSTM(Adaptive_LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0.2, bidirectional=False, adaptive_input_size=16, adaptive_hidden_size=128,
                 num_topics=3, ensemble_factors=128):
        self.num_topics = num_topics
        self.ensemble_factors = ensemble_factors
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout,
                         bidirectional, adaptive_input_size, adaptive_hidden_size,
                         cell_type=ContextTopicAdaptiveLSTMCell)

    def _create_lstm(self, stack_type, layer_type, cell_type):
        return stack_type(self.num_layers, layer_type,
                          first_layer_args=[cell_type, self.input_size, self.hidden_size,
                                            self.adaptive_input_size, self.adaptive_hidden_size,
                                            self.num_topics, self.ensemble_factors, self.dropout],
                          other_layer_args=[cell_type, self.hidden_size * self.dirs, self.hidden_size,
                                            self.adaptive_input_size, self.adaptive_hidden_size,
                                            self.num_topics, self.ensemble_factors, self.dropout],
                          dropout=self.dropout)

    def forward(self, input_, states=None, *context):
        assert len(context) >= 2, 'Two arguments (context_repr, theta) are expected!'
        assert context[1].size(1) == self.num_topics, \
            'theta.size(1) must be equal to {}'.format(self.num_topics)
        return super().forward(input_, states, *context)
