from torch import optim

from pygcn.recurrent.data_loader import GraphLoader
from pygcn.recurrent.models import RNNModel

learning_rate = 0.01
weight_decay = 0.0005


def main():
    loader = GraphLoader()
    model = RNNModel(feat_x_n=feat_x.shape[1] if feat_x.shape else 0,
                     topo_x_n=topo_x.shape[1] if topo_x.shape else 0,
                     n_output=labels.max().data[0] + 1,
                     h_layers=config["hidden_layers"],
                     dropout=config["dropout"],
                     rnn_type="RNN_RELU",
                     )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_model(model, optimizer, train_data, test_data)
    evaluate_test(model, test_data, )

if __name__ == "__main__":
    main()
