import torch.nn as nn

from ..layers import GraphConvolution


class RecurrentGCNLayer(GraphConvolution):
    def __init__(self, in_features, out_features, dropout, is_before=True, bias=True, rnn_type="RNN_RELU"):
        super(RecurrentGCNLayer, self).__init__(in_features, out_features, bias=bias)
        ninp = in_features if is_before else out_features
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, ninp, 1, dropout=dropout)
            nn.LSTM
        elif rnn_type in ["RNN_TANH", "RNN_RELU"]:
            self.rnn = nn.RNN(ninp, ninp, 1, nonlinearity=rnn_type[4:].lower(), dropout=dropout)
        else:
            raise ValueError(
                "An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']")

        self._before = is_before
        self.rnn_type = rnn_type

    # noinspection PyMethodOverriding
    def forward(self, x, adj, hidden):
        if self._before:
            x, hidden = self.rnn(x, hidden)
            x = super(RecurrentGCNLayer, self).forward(x, adj)
        else:
            x = super(RecurrentGCNLayer, self).forward(x, adj)
            x, hidden = self.rnn(x, hidden)
        return x, hidden
