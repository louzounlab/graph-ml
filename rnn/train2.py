from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from feature_meta import NODE_FEATURES
from features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from features_infra.feature_calculators import FeatureMeta
from loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger
from pygcn.model_meter import ModelMeter
from pygcn.recurrent.data_loader import GraphLoader
from pygcn.recurrent.models import RNNModel

PROJ_DIR = os.path.realpath(os.path.join(os.path.realpath(__file__), "..", "..", ".."))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    # help='Disables CUDA training.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10,
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
    # if not torch.cuda.is_available():
    args.cuda = None

    return args


KIPF_BASE = {"dropout": 0.5, "weight_decay": 5e-4, "lr": 0.01, "hidden": 16, "fast_mode": False}

NEIGHBOR_FEATURES = {
    "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}

DATA_PATH = "/home/benami/git/data/firms/years"


class ModelRunner:
    def __init__(self, data_path, test_p, cuda, logger, data_logger=None):
        self._logger = logger
        self._cuda = cuda
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._data_path = data_path

        # feature_meta = NEIGHBOR_FEATURES
        feature_meta = NODE_FEATURES
        # feature_meta = NODE_FEATURES.copy()
        # feature_meta.update(NEIGHBOR_FEATURES)
        self.loader = GraphLoader(self._data_path, feature_meta, test_p, cuda_num=cuda, logger=self._logger)

    def get_gnx_paths(self):
        for path in sorted(os.listdir(self._data_path), key=int):
            yield os.path.join(self._data_path, path)

    def run(self, ordered_config):
        config = dict(ordered_config)
        nbatch = config["nbatch"]

        # first_path = os.path.join(next(self.get_gnx_paths()), "features_0")
        # self.loader.split_data(config["test_p"], self.get_gnx_paths())

        # Load data
        # adj_r_t, adj, feat_x, topo_x, labels, idx_train, idx_test = self.loader.load(data_type="asymmetric",
        #                                                                              feature_path=first_path)

        # n_features, n_output, n_layers, batch_size, dropout = 0., rnn_type = "RNN_RELU"):
        n_features = self.loader.num_features
        n_labels = self.loader.num_labels
        n_layers = self.loader.num_layers

        # Model and optimizer
        model = RNNModel(n_features=n_features, n_output=n_labels, n_layers=n_layers, batch_size=nbatch,
                         dropout=config["dropout"], rnn_type="RNN_RELU")
        # n_output=labels.max().data[0] + 1,
        # h_layers=config["hidden_layers"],
        # dropout=config["dropout"],
        # rnn_type="RNN_RELU",
        # )

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        if self._cuda is not None:
            model.cuda(self._cuda)

        self._data_logger.info("epoch", "batch_num", "loss_train", "acc_train")
        self._logger.debug("Training...")
        for epoch in range(config["epochs"]):
            self._train(epoch, model, optimizer, nbatch)

        self._data_logger.space()
        self._data_logger.info("batch_num", "loss_test", "acc_test")
        # self._logger.debug("Testing...")
        # return self._test(model, nbatch)

        # for model in models.values():
        #     model["arguments"] = list(map(Variable, model["arguments"]))
        #     model["labels"] = Variable(model["labels"])

        # self._logger.debug("Optimization Finished!")
        # self._logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        # self._train(epoch, config["train_p"], name, model_args, temp_gnx)

    # def _train(self, epoch, model, optimizer, model_args, idx_train, idx_val, labels):
    def _train(self, epoch, model, optimizer, nbatch):
        criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([1/4, 1/85, 1/108]), ignore_index=2)

        model.train()

        meter = ModelMeter(self.loader.labels, logger=self._logger)
        for i, (inputs, targets) in enumerate(self.loader.load_train_data(nbatch)):
            preds = model(inputs)
            view_preds = preds.view(nbatch * self.loader.sequence_len, -1)
            optimizer.zero_grad()
            loss = criterion(view_preds, targets)
            meter.update(loss, view_preds, targets)

            loss.backward()
            optimizer.step()

            model.detach_hidden()
            self._data_logger.info(epoch, i, meter.last_loss, meter.last_acc)

            meter.log("[%03d/%02d] Train" % (i, epoch,))

        meter.log("%02d. Train" % epoch, func=np.mean, level=logging.INFO)
        print(meter._conf.value())
        print("Counter preds: %s" % (meter._c_inp,))
        print("Counter targets: %s" % (meter._c_tar,))
        return meter

    def _test(self, model, nbatch):
        loss_func = torch.nn.NLLLoss(weight=torch.FloatTensor([1 / 4, 1 / 85, 1 / 108]), ignore_index=2)

        model.eval()

        # model.init_hidden()

        meter = ModelMeter(self.loader.labels, logger=self._logger)
        for i, (inputs, targets) in enumerate(self.loader.load_test_data(nbatch)):
            preds = model(inputs)
            view_preds = preds.view(nbatch * self.loader.sequence_len, -1)
            loss = loss_func(view_preds, targets)
            meter.update(loss, view_preds, targets)

            self._data_logger.info(i, meter.last_loss, meter.last_acc)

            meter.log("[%03d] Test" % (i,))
        meter.log("Test", func=np.mean, level=logging.INFO)
        print("Counter preds: %s" % (meter._c_inp,))
        print("Counter targets: %s" % (meter._c_tar,))
        return meter


def main():
    args = parse_args()
    num_samples = 1

    # "weight_decay": 5e-4, "dropout": 0.5, "hidden_layers": [100],
    config = {"dropout": 0.0, "learning_rate": 0.01, "weight_decay": 0, "epochs": args.epochs,
              "feat_type": "features", "dataset": "firms", "seed": 12345678, "nbatch": 100}

    products_path = os.path.join(PROJ_DIR, "logs", config["dataset"], time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.DEBUG),
        # FileLogger("results_%s" % config["dataset"], path=products_path, level=logging.INFO),
        # FileLogger("results_%s_all" % config["dataset"], path=products_path, level=logging.DEBUG),
    ], name="IdanLogger")

    data_logger = CSVLogger("results_%s" % config["dataset"], path=products_path)

    runner = ModelRunner(DATA_PATH, 0.5, args.cuda, logger, data_logger)

    # config['seed'] = random.randint(1, 1000000000)

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if args.cuda is not None:
            torch.cuda.manual_seed(config["seed"])

            # config_args = sorted(config.items(), key=at(0))
    logger.info("Arguments: " + ", ".join("%s: %s" % (name, val) for name, val in config.items()))
    res = []
    for _ in range(num_samples):
        res.append(runner.run(config))
        # res = [runner.run(config) for _ in range(num_samples)]
        # pickle.dump({"params": current_params, "res": res}, open(os.path.join(products_path, "quant_res.pkl"), "ab"))


if __name__ == "__main__":
    main()


    # def _test1(self, name, model_args, config):
    #     model, arguments = model_args["model"], model_args["arguments"]
    #     model.eval()
    #
    #     loss_test = []
    #     acc_test = []
    #     hidden = model.init_hidden()
    #     for gnx_path in self.get_gnx_paths():
    #         adj_r_t, adj, feat_x, topo_x, labels, idx_train, idx_val, idx_test = self.loader.load(
    #             data_type="asymmetric", feature_path=gnx_path)
    #
    #         output = model(feat_x, topo_x, adj, hidden)
    #         loss_test.append(functional.nll_loss(output[idx_test], labels[idx_test]))
    #         acc_test.append(accuracy(output[idx_test], labels[idx_test]))
    #         # loss_train = functional.nll_loss(output[idx_train], labels[idx_train])
    #         # acc_train = accuracy(output[idx_train], labels[idx_train])
    #         # loss_train.backward()
    #         # optimizer.step()
    #     # output = model(*arguments)
    #     # loss_test = functional.nll_loss(output[idx_test], labels[idx_test])
    #     # acc_test = accuracy(output[idx_test], labels[idx_test])
    #     loss_test = loss_test[-1]
    #     acc_test = acc_test[-1]
    #     self._logger.info(name + " " +
    #                       "Test set results: " +
    #                       "loss= {:.4f} ".format(loss_test.data[0]) +
    #                       "accuracy= {:.4f}".format(acc_test.data[0]))
    #     self._data_logger.info(name, loss_test.data[0], acc_test.data[0], *list(map(at(1), config)))
    #     return {"loss": loss_test.data[0], "acc": acc_test.data[0]}
