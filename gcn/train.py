from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

from feature_meta import NODE_FEATURES
from features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from features_infra.feature_calculators import FeatureMeta
from gcn import *
from gcn.data_loader import GraphLoader
from gcn.layers import AsymmetricGCN
from gcn.models import GCNCombined, GCN
from loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='The dataset to use.')
    parser.add_argument('--prefix', type=str, default="",
                        help='The prefix of the products dir name.')

    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not torch.cuda.is_available():
        args.cuda = None
    return args


NEIGHBOR_FEATURES = {
    "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_features():
    # if config["feat_type"] == "neighbors":
    #     feature_meta = NEIGHBOR_FEATURES
    # elif config["feat_type"] == "features":
    #     feature_meta = NODE_FEATURES
    # else:
    feature_meta = NODE_FEATURES.copy()
    feature_meta.update(NEIGHBOR_FEATURES)
    return feature_meta


class ModelRunner:
    def __init__(self, dataset_path, conf, logger, data_logger=None):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._criterion = torch.nn.NLLLoss()
        self._conf = conf

        features_meta = get_features()
        self.loader = GraphLoader(dataset_path, features_meta, is_max_connected=False,  # self._conf['dataset'] == "citeseer",
                                  cuda_num=conf["cuda"], logger=self._logger)

    def _get_models(self):
        bow_feat = self.loader.bow_mx
        topo_feat = self.loader.topo_mx

        model1 = GCN(nfeat=bow_feat.shape[1],
                     hlayers=[self._conf["kipf"]["hidden"]],
                     nclass=self.loader.num_labels,
                     dropout=self._conf["kipf"]["dropout"])
        opt1 = optim.Adam(model1.parameters(), lr=self._conf["kipf"]["lr"],
                          weight_decay=self._conf["kipf"]["weight_decay"])

        model2 = GCNCombined(nbow=bow_feat.shape[1],
                             nfeat=topo_feat.shape[1],
                             hlayers=self._conf["hidden_layers"],
                             nclass=self.loader.num_labels,
                             dropout=self._conf["dropout"])
        opt2 = optim.Adam(model2.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        model3 = GCN(nfeat=topo_feat.shape[1],
                     hlayers=self._conf["multi_hidden_layers"],
                     nclass=self.loader.num_labels,
                     dropout=self._conf["dropout"],
                     layer_type=None)
        opt3 = optim.Adam(model3.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        model4 = GCN(nfeat=topo_feat.shape[1],
                     hlayers=self._conf["multi_hidden_layers"],
                     nclass=self.loader.num_labels,
                     dropout=self._conf["dropout"],
                     layer_type=AsymmetricGCN)
        opt4 = optim.Adam(model4.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        return {
            "kipf": {
                "model": model1, "optimizer": opt1,
                "arguments": [self.loader.bow_mx, self.loader.adj_mx],
                "labels": self.loader.labels,
            },
            "our_combined": {
                "model": model2, "optimizer": opt2,
                "arguments": [self.loader.bow_mx, self.loader.topo_mx, self.loader.adj_rt_mx],
                "labels": self.loader.labels,
            },
            # "our_topo_sym": {
            #     "model": model3, "optimizer": opt3,
            #     "arguments": [self.loader.topo_mx, self.loader.adj_mx],
            #     "labels": self.loader.labels,
            # },
            # "our_topo_asymm": {
            #     "model": model4, "optimizer": opt4,
            #     "arguments": [self.loader.topo_mx, self.loader.adj_rt_mx],
            #     "labels": self.loader.labels,
            # },
        }

    def run(self, train_p):
        self.loader.split_train(train_p)

        models = self._get_models()

        if self._conf["cuda"] is not None:
            [model["model"].cuda(self._conf["cuda"]) for model in models.values()]

        for model in models.values():
            model["arguments"] = list(map(Variable, model["arguments"]))
            model["labels"] = Variable(model["labels"])

        # Train model
        train_idx, val_idx = self.loader.train_idx, self.loader.val_idx
        for epoch in range(self._conf["epochs"]):
            for name, model_args in models.items():
                self._train(epoch, name, model_args, train_idx, val_idx)

        # Testing
        test_idx = self.loader.test_idx
        result = {name: self._test(name, model_args, test_idx) for name, model_args in models.items()}
        for name, val in sorted(result.items(), key=lambda x: x[0]):
            self._data_logger.info(name, val["loss"], val["acc"], (train_p / (2 - train_p)) * 100)
        return result

    def _train(self, epoch, model_name, model_args, idx_train, idx_val):
        model, optimizer = model_args["model"], model_args["optimizer"]
        arguments, labels = model_args["arguments"], model_args["labels"]

        model.train()
        optimizer.zero_grad()
        output = model(*arguments)
        loss_train = self._criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not self._conf["fastmode"]:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(*arguments)

        loss_val = self._criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        self._logger.debug(model_name + ": " +
                           'Epoch: {:04d} '.format(epoch + 1) +
                           'loss_train: {:.4f} '.format(loss_train.data[0]) +
                           'acc_train: {:.4f} '.format(acc_train.data[0]) +
                           'loss_val: {:.4f} '.format(loss_val.data[0]) +
                           'acc_val: {:.4f} '.format(acc_val.data[0]))

    def _test(self, model_name, model_args, test_idx):
        model, arguments, labels = model_args["model"], model_args["arguments"], model_args["labels"]
        model.eval()
        output = model(*arguments)
        loss_test = functional.nll_loss(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])
        self._logger.info(model_name + " Test: " +
                          "loss= {:.4f} ".format(loss_test.data[0]) +
                          "accuracy= {:.4f}".format(acc_test.data[0]))
        return {"loss": loss_test.data[0], "acc": acc_test.data[0]}


def init_seed(seed, cuda=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed(seed)


def aggregate_results(res_list):
    aggregated = {}
    for cur_res in res_list:
        for name, vals in cur_res.items():
            if name not in aggregated:
                aggregated[name] = {}
            for key, val in vals.items():
                if key not in aggregated[name]:
                    aggregated[name][key] = []
                aggregated[name][key].append(val)
    return aggregated


def execute_runner(runner, logger, train_p, num_iter=1):
    train_p /= 100
    val_p = test_p = (1 - train_p) / 2.
    train_p /= (val_p + train_p)

    runner.loader.split_test(test_p)
    res = [runner.run(train_p) for _ in range(num_iter)]
    aggregated = aggregate_results(res)
    for name, vals in aggregated.items():
        val_list = sorted(vals.items(), key=lambda x: x[0], reverse=True)
        logger.info("*"*15 + "%s mean: %s", name, ", ".join("%s=%3.4f" % (key, np.mean(val)) for key, val in val_list))
        logger.info("*"*15 + "%s std: %s", name, ", ".join("%s=%3.4f" % (key, np.std(val)) for key, val in val_list))


def main_clean():
    args = parse_args()
    dataset = "citeseer"

    seed = random.randint(1, 1000000000)
    # "feat_type": "neighbors",
    conf = {
        "kipf": {"hidden": args.hidden, "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay},
        "hidden_layers": [16], "multi_hidden_layers": [100, 35], "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
        "dataset": dataset, "epochs": args.epochs, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed}

    init_seed(conf['seed'], conf['cuda'])
    dataset_path = os.path.join(PROJ_DIR, "data", dataset)

    products_path = os.path.join(CUR_DIR, "logs", args.prefix + dataset, time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.DEBUG),
        FileLogger("results_%s" % conf["dataset"], path=products_path, level=logging.INFO),
        FileLogger("results_%s_all" % conf["dataset"], path=products_path, level=logging.DEBUG),
    ], name=None)

    data_logger = CSVLogger("results_%s" % conf["dataset"], path=products_path)
    data_logger.info("model_name", "loss", "acc", "train_p")

    runner = ModelRunner(dataset_path, conf, logger=logger, data_logger=data_logger)
    # execute_runner(runner, logger, 5, num_iter=30)

    for train_p in range(5, 90, 10):
        execute_runner(runner, logger, train_p, num_iter=10)
    logger.info("Finished")


if __name__ == "__main__":
    main_clean()
