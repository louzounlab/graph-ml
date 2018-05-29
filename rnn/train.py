from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle
import random
import time
from itertools import product
from operator import itemgetter as at

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

from feature_meta import NODE_FEATURES
from features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from features_infra.feature_calculators import FeatureMeta
from loggers import PrintLogger, FileLogger, multi_logger, EmptyLogger, CSVLogger
from pygcn.recurrent.data_loader import GraphLoader
from pygcn.recurrent.models import RNNModel

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    # help='Disables CUDA training.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='Initial learning rate.')
    # parser.add_argument('--weight_decay', type=float, default=5e-4,
    #                     help='Weight decay (L2 loss on parameters).')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--dataset', type=str, default="cora",
    #                     help='The dataset to use.')

    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not torch.cuda.is_available():
        args.cuda = None

    return args


KIPF_BASE = {"dropout": 0.5, "weight_decay": 5e-4, "lr": 0.01, "hidden": 16, "fast_mode": False}

NEIGHBOR_FEATURES = {
    "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


DATA_PATH = "/home/benami/git/data/firms/years"


class ModelRunner:
    def __init__(self, data_path, cuda, logger, data_logger=None):
        self._logger = logger
        self._cuda = cuda
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._data_path = data_path

        # feature_meta = NEIGHBOR_FEATURES
        feature_meta = NODE_FEATURES
        # feature_meta = NODE_FEATURES.copy()
        # feature_meta.update(NEIGHBOR_FEATURES)
        self.loader = GraphLoader(feature_meta, cuda_num=cuda, logger=self._logger)

    def get_gnx_paths(self):
        for path in sorted(os.listdir(self._data_path), key=int):
            yield os.path.join(self._data_path, path)

    def run(self, ordered_config):
        config = dict(ordered_config)

        first_path = os.path.join(next(self.get_gnx_paths()), "features_1")
        self.loader.split_data(config["test_p"], self.get_gnx_paths())

        # Load data
        adj_r_t, adj, feat_x, topo_x, labels, idx_train, idx_test = self.loader.load(data_type="asymmetric",
                                                                                     feature_path=first_path)

        # Model and optimizer
        model1 = RNNModel(feat_x_n=feat_x.shape[1] if feat_x.shape else 0,
                          topo_x_n=topo_x.shape[1] if topo_x.shape else 0,
                          n_output=labels.max().data[0] + 1,
                          h_layers=config["hidden_layers"],
                          dropout=config["dropout"],
                          rnn_type="RNN_RELU",
                          )
        # nbow=bow_feat.shape[1],
        # nfeat=topo_feat.shape[1],
        #
        # nclass=labels.max() + 1,
        optimizer1 = optim.Adam(model1.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        models = {
            "our": {"model": model1, "optimizer": optimizer1, },
        }

        if self._cuda is not None:
            [model["model"].cuda(self._cuda) for model in models.values()]

        # for model in models.values():
        #     model["arguments"] = list(map(Variable, model["arguments"]))
        #     model["labels"] = Variable(model["labels"])

        # Train model
        t_total = time.time()
        for epoch in range(config["epochs"]):
            for name, model_args in models.items():
                self._train(epoch, name, model_args)

        self._logger.debug("Optimization Finished!")
        self._logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        # self._train(epoch, config["train_p"], name, model_args, temp_gnx)
        return {name: self._test(name, model_args, ordered_config) for name, model_args in models.items()}

    # def _train(self, epoch, model, optimizer, model_args, idx_train, idx_val, labels):
    def _train(self, epoch, name, model_args):
        model, optimizer = model_args["model"], model_args["optimizer"]
        # arguments, labels = model_args["arguments"], model_args["labels"]

        t = time.time()
        model.train()
        optimizer.zero_grad()

        hidden = model.init_hidden()

        loss_train, acc_train = [], []
        for gnx_path in self.get_gnx_paths():
            adj_r_t, adj, feat_x, topo_x, labels, idx_train, idx_test = self.loader.load(data_type="asymmetric",
                                                                                         feature_path=gnx_path)

            output = model(feat_x, topo_x, adj, hidden)
            loss_train.append(functional.nll_loss(output[idx_train], labels[idx_train]))
            acc_train.append(accuracy(output[idx_train], labels[idx_train]))
            loss_train[-1].backward()
            optimizer.step()

            # if not KIPF_BASE["fastmode"]:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            # model.eval()
            # output = model(*arguments)
        loss_train = loss_train[-1]
        acc_train = acc_train[-1]

        # loss_val = functional.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        self._logger.debug(name + ": " +
                           'Epoch: {:04d} '.format(epoch + 1) +
                           'loss_train: {:.4f} '.format(loss_train.data[0]) +
                           'acc_train: {:.4f} '.format(acc_train.data[0]) +
                           # 'loss_val: {:.4f} '.format(loss_val.data[0]) +
                           # 'acc_val: {:.4f} '.format(acc_val.data[0]) +
                           'time: {:.4f}s'.format(time.time() - t))

    def _test(self, name, model_args, config):
        model, arguments = model_args["model"], model_args["arguments"]
        model.eval()

        loss_test = []
        acc_test = []
        hidden = model.init_hidden()
        for gnx_path in self.get_gnx_paths():
            adj_r_t, adj, feat_x, topo_x, labels, idx_train, idx_val, idx_test = self.loader.load(
                data_type="asymmetric", feature_path=gnx_path)

            output = model(feat_x, topo_x, adj, hidden)
            loss_test.append(functional.nll_loss(output[idx_test], labels[idx_test]))
            acc_test.append(accuracy(output[idx_test], labels[idx_test]))
            # loss_train = functional.nll_loss(output[idx_train], labels[idx_train])
            # acc_train = accuracy(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
        # output = model(*arguments)
        # loss_test = functional.nll_loss(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])
        loss_test = loss_test[-1]
        acc_test = acc_test[-1]
        self._logger.info(name + " " +
                          "Test set results: " +
                          "loss= {:.4f} ".format(loss_test.data[0]) +
                          "accuracy= {:.4f}".format(acc_test.data[0]))
        self._data_logger.info(name, loss_test.data[0], acc_test.data[0], *list(map(at(1), config)))
        return {"loss": loss_test.data[0], "acc": acc_test.data[0]}


def main(product_params, args):
    train_p = 50
    num_samples = 3

    config = {"hidden_layers": [70, 35], "dropout": KIPF_BASE["dropout"], "learning_rate": KIPF_BASE["lr"],
              "weight_decay": KIPF_BASE["weight_decay"], "epochs": args.epochs, "train_p": 0,
              "feat_type": "neighbors", "dataset": "firms", "seed": 12345678}

    products_path = os.path.join(PROJ_DIR, "logs", config["dataset"], time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.INFO),
        FileLogger("results_%s" % config["dataset"], path=products_path, level=logging.INFO),
        FileLogger("results_%s_all" % config["dataset"], path=products_path, level=logging.DEBUG),
    ], name=None)

    # data_logger = CSVLogger("results_%s" % config["dataset"], path=products_path)
    # all_args = set(config).union(map(at(0), product_params))
    # data_logger.info("name", "loss", "accuracy", *sorted(all_args))

    runner = ModelRunner(DATA_PATH, args.cuda, logger, None)  # data_logger)

    train_p /= 100.
    config["test_p"] = 1 - train_p
    config["train_p"] = train_p

    # for train_p in [5]:  # + list(range(5, 100, 10)):
    for pars in product(*map(at(1), product_params)):
        current_params = list(zip(map(at(0), product_params), pars))
        # cur_seed = 214899513 # random.randint(1, 1000000000)
        cur_seed = random.randint(1, 1000000000)
        current_params.append(("seed", cur_seed))
        config.update(current_params)

        if "seed" in config:
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            if args.cuda is not None:
                torch.cuda.manual_seed(config["seed"])

        config_args = sorted(config.items(), key=at(0))
        logger.info("Arguments: (train %1.2f) " + ", ".join("%s: %s" % (name, val) for name, val in current_params),
                    train_p)
        res = []
        for _ in range(num_samples):
            res.append(runner.run(config_args))

        # res = [runner.run(config) for _ in range(num_samples)]
        pickle.dump({"params": current_params, "res": res}, open(os.path.join(products_path, "quant_res.pkl"), "ab"))


def grid_search():
    args = parse_args()
    all_params = [[
        ("weight_decay", [0.0001]),
        ("dropout", [0.6]),
        ("learning_rate", [0.01]),
        ("hidden_layers", [[16]]),  # , 20], [70, 35], [16], [100, 50, 20], [300, 100, 30], [300, 200, 80, 20]] +
        # [[x + 100, x] for x in range(10, 20)] +
        # [[x] for x in range(10, 20)] +
        # [[200 + x, 100 + x, x] for x in range(10, 20)]),
        ("feat_type", ["neighbors", "features", "combined"]), ]]

    # ("weight_decay", [0.0005, 0.001, 0.005, 0.01, 0.035]),
    # ("dropout", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    # ("learning_rate", [0.01, 0.02, 0.05, 0.08, 0.1]),
    # ("hidden_layers", [[100, 20], [70, 35], [16], [100, 50, 20], [300, 100, 30], [300, 200, 80, 20]] +
    #  [[x + 100, x] for x in range(10, 20)] +
    #  [[x] for x in range(10, 20)] +
    #  [[200 + x, 100 + x, x] for x in range(10, 20)]),
    # ("feat_type", ["neighbors", "features", "combined"]), ]]
    # ], [
    # ("weight_decay", [0.0005, 0.001, 0.005, 0.01, 0.035]),
    # ("dropout", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    # ("learning_rate", [0.01, 0.02, 0.05, 0.08, 0.1]),
    # ("hidden_layers", [[x + 100, x] for x in range(10, 20)]),
    # ("feat_type", ["neighbors", "features", "combined"]),
    # ], [
    #     ("weight_decay", [0.0005, 0.001, 0.005, 0.01, 0.035]),
    #     ("dropout", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    #     ("learning_rate", [0.01, 0.02, 0.05, 0.08, 0.1]),
    #     ("hidden_layers", [[x] for x in range(10, 20)]),
    #     ("feat_type", ["neighbors", "features", "combined"]),
    # ], [  # Shouldn't get here
    #     ("weight_decay", [0.0005, 0.001, 0.005, 0.01, 0.035]),
    #     ("dropout", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    #     ("learning_rate", [0.01, 0.02, 0.05, 0.08, 0.1]),
    #     ("hidden_layers", [[200 + x, 100 + x, x] for x in range(10, 20)]),
    #     ("feat_type", ["neighbors", "features", "combined"]),
    # ]]

    # all_params = [
    #     # [
    #     # ("weight_decay", [0.0005, 0.001, 0.005,]),
    #     # ("dropout", [0.2, 0.3, 0.4, 0.5, 0.6,]),
    #     # ("learning_rate", [0.01, 0.02, 0.05]),
    #     # ("hidden_layers", [[14], [15], [16], [17]]),
    #     # ("feat_type", ["neighbors"]),
    #     # ],
    #     [
    #         # ("weight_decay", [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]),
    #         ("weight_decay", [x for i in range(-4, 0) for x in [y * (10 ** i) for y in range(1, 10, 2)]]),
    #         # ("weight_decay", [0.0005]),
    #         ("dropout", [0.5]),
    #         ("learning_rate", [0.01]),
    #         # ("hidden_layers", [[13], [14], [15], [16], [17]]),
    #         ("hidden_layers", [[16]]),
    #         ("feat_type", ["features"]),  # , "neighbors", "combined"]),
    #     ],
    # ]
    for params in all_params:
        main(params, args)


if __name__ == "__main__":
    grid_search()


def main2():
    num_samples = 5
    logger = PrintLogger("IdansLogger", level=logging.INFO)
    runner = ModelRunner(logger)

    config = {"train_p": 0, "hidden_layers": [70, 35], "dropout": args.dropout, "learning_rate": args.lr,
              "weight_decay": args.weight_decay, "epochs": args.epochs}

    for train_p in [5]:  # + list(range(5, 100, 10)):
        train_p /= 100.
        val_p = test_p = (1 - train_p) / 2.
        train_p /= val_p + train_p
        config["train_p"] = train_p

        runner.loader.split_test(test_p)
        for _ in range(num_samples):
            runner.run(train_p)
