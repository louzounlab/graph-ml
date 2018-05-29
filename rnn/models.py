import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as functional

from pygcn.layers import GraphConvolution


class RNNGCNModel(nn.Module):
    RNN_TYPES = ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']

    def __init__(self, feat_x_n, topo_x_n, n_output, h_layers, dropout=0., rnn_type="RNN_RELU"):
        super(RNNModel, self).__init__()
        self._comb = "topo"
        n_input = {"feat": feat_x_n, "topo": topo_x_n, "comb": feat_x_n + topo_x_n}[self._comb]

        all_layers = [n_input] + h_layers + [n_output]
        self._rnn_type = rnn_type
        self._dropout = dropout
        self.drop = nn.Dropout(self._dropout)
        if self._rnn_type not in self.RNN_TYPES:
            raise ValueError("An invalid rnn_type, options are %s" % (self.RNN_TYPES,))
        self._rnn_in, self._rnn_out, self._rnn_layers = all_layers[0], all_layers[0], 1
        self.rnn = nn.RNNBase(self._rnn_type, self._rnn_in, self._rnn_out, self._rnn_layers,
                              dropout=self._dropout)

        self.gcn_layers = nn.ModuleList([GraphConvolution(first, second)
                                         for first, second in zip(all_layers[:-1], all_layers[1:])])

        self._activation_func = functional.relu

    def forward(self, feat_x, topo_x, adj, hidden):
        if "comb" == self._comb:
            x = torch.cat((topo_x, feat_x), dim=1)
        else:
            x = {"feat": feat_x, "topo": topo_x}[self._comb]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)

        for layer in self.gcn_layers:
            x = self._activation_func(layer(x, adj))
        output = functional.log_softmax(x, dim=1)
        return output, hidden

    def init_hidden(self):
        if self._rnn_type != 'LSTM':
            return torch.zeros(self._rnn_layers, self._rnn_in, self._rnn_out)
        return (torch.zeros(self._rnn_layers, self._rnn_in, self._rnn_out),
                torch.zeros(self._rnn_layers, self._rnn_in, self._rnn_out))

    def init_hidden1(self, bsz):
        weight = next(self.parameters())
        if self._rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNModel(nn.Module):
    RNN_TYPES = ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']

    def __init__(self, n_features, n_output, n_layers, batch_size, dropout=0., rnn_type="RNN_RELU"):
        super(RNNModel, self).__init__()

        n_hidden = 100
        self._rnn_type = rnn_type
        self._dropout = dropout
        self.drop = nn.Dropout(self._dropout)
        if self._rnn_type not in self.RNN_TYPES:
            raise ValueError("An invalid rnn_type, options are %s" % (self.RNN_TYPES,))
        self._rnn_in, self._hidden_size, self._rnn_layers = n_features, n_hidden, n_layers
        self.rnn = nn.RNNBase(self._rnn_type, self._rnn_in, self._hidden_size, self._rnn_layers, bias=False,
                              dropout=self._dropout, batch_first=True)
        self._linear = nn.Linear(n_hidden, n_output)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._activation_func = functional.relu
        self._batch_size = batch_size
        self._hidden = self._init_hidden()

    def detach_hidden(self):
        self._hidden = self._init_hidden()

    def detach_hidden1(self):
        if self._rnn_type != 'LSTM':
            self._hidden = self._hidden.detach()
        self._hidden = tuple([hidden.detach() for hidden in self._hidden])

    def _init_hidden(self, is_random=False):
        func = torch.randn if is_random else torch.zeros
        gen = lambda: autograd.Variable(func(self._rnn_layers, self._batch_size, self._hidden_size))
        if self._rnn_type == 'LSTM':
            return gen(), gen()
        return gen()

    def forward(self, inputs):
        # input = (batch, seq_len, features)
        output, self._hidden = self.rnn(inputs, self._hidden)
        output = self._linear(output)
        output = self.log_softmax(output)
        return output


if __name__ == "__main__":
    # model = RNNModel(200, 6, [100, 30, 10], dropout=0.2, rnn_type="RNN_TANH")
    print("Finished")


class DemoRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(DemoRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden1(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
