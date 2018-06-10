from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.optim as optim

from feature_meta import NODE_FEATURES
from loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger
from rnn import *
# from rnn.data_loader import GraphLoader
from rnn.data_loader_content import GraphLoader
from rnn.model_meter import ModelMeter
from rnn.models import FFModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    # help='Disables CUDA training.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    # parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=200,
    #                     help='Number of epochs to train.')
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

DATA_PATH = "/home/benami/git/data/firms/years"


class ModelRunner:
    def __init__(self, data_path, test_p, cuda, logger, data_logger=None):
        self._logger = logger
        self._cuda = cuda
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._data_path = data_path

        all_features = [NODE_FEATURES]  # , UNDIRECTED_NEIGHBOR_FEATURES]
        feature_meta = dict(y for x in all_features for y in x.items())
        self.loader = GraphLoader(self._data_path, feature_meta, test_p, gnx_idx=6, cuda_num=cuda, logger=self._logger)

    def get_gnx_paths(self):
        for path in sorted(os.listdir(self._data_path), key=int):
            yield os.path.join(self._data_path, path)

    def run(self, ordered_config):
        config = dict(ordered_config)
        nbatch = config["nbatch"]

        n_features = self.loader.num_features
        n_labels = self.loader.num_labels
        n_layers = self.loader.num_layers

        # Model and optimizer
        model = FFModel(n_features=n_features, n_output=n_labels, n_layers=n_layers, batch_size=nbatch,
                        dropout=config["dropout"], rnn_type="RNN_RELU")
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        if self._cuda is not None:
            model.cuda(self._cuda)

        self._data_logger.info("epoch", "batch_num", "loss_train", "acc_train")
        self._logger.debug("Training...")
        for epoch in range(config["epochs"]):
            self._train(epoch, model, optimizer, nbatch)

        self._data_logger.space()
        self._data_logger.info("batch_num", "loss_test", "acc_test", new_file=True)
        self._logger.debug("Testing...")
        return self._test(model, nbatch)

        # for model in models.values():
        #     model["arguments"] = list(map(Variable, model["arguments"]))
        #     model["labels"] = Variable(model["labels"])

        # self._logger.debug("Optimization Finished!")
        # self._logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        # self._train(epoch, config["train_p"], name, model_args, temp_gnx)

    # def _train(self, epoch, model, optimizer, model_args, idx_train, idx_val, labels):
    def _train(self, epoch, model, optimizer, nbatch):
        # criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([1 / 4, 1 / 85, 1 / 108]), ignore_index=2)
        criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([1 / 5, 1 / 100, 0]), ignore_index=2)

        model.train()

        meter = ModelMeter(self.loader.labels, logger=self._logger)
        for i, (inputs, targets) in enumerate(self.loader.load_train_data(nbatch)):
            preds = model(inputs)
            view_preds = preds  # view_preds = preds.view(nbatch * self.loader.sequence_len, -1)
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
        # loss_func = torch.nn.NLLLoss(weight=torch.FloatTensor([1 / 4, 1 / 85, 1 / 108]), ignore_index=2)
        loss_func = torch.nn.NLLLoss(weight=torch.FloatTensor([1 / 5, 1 / 100, 0]), ignore_index=2)

        model.eval()

        # model.init_hidden()

        meter = ModelMeter(self.loader.labels, logger=self._logger)
        for i, (inputs, targets) in enumerate(self.loader.load_test_data(nbatch)):
            preds = model(inputs)
            view_preds = preds  # .view(nbatch * self.loader.sequence_len, -1)
            loss = loss_func(view_preds, targets)
            meter.update(loss, view_preds, targets)

            self._data_logger.info(i, meter.last_loss, meter.last_acc)

            meter.log("[%03d] Test" % (i,))
        meter.log("Test", func=np.mean, level=logging.INFO)
        print(meter._conf.value())
        print("Counter preds: %s" % (meter._c_inp,))
        print("Counter targets: %s" % (meter._c_tar,))
        return meter


def init_seed(seed, cuda=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed(seed)


def main():
    args = parse_args()
    num_samples = 1

    # "weight_decay": 5e-4, "dropout": 0.5, "hidden_layers": [100],
    seed = random.randint(1, 1000000000)
    config = {"dropout": 0.0, "learning_rate": 0.01, "weight_decay": 0, "epochs": 50,
              "feat_type": "features", "dataset": "firms", "seed": seed, "nbatch": 50, "test_p": 0.3}

    products_path = os.path.join(PROJ_DIR, "logs", config["dataset"], time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.INFO),
        # FileLogger("results_%s" % config["dataset"], path=products_path, level=logging.INFO),
        # FileLogger("results_%s_all" % config["dataset"], path=products_path, level=logging.DEBUG),
    ], name="IdanLogger")

    data_logger = CSVLogger("results_%s" % config["dataset"], path=products_path)

    runner = ModelRunner(DATA_PATH, config["test_p"], args.cuda, logger, data_logger)

    if "seed" in config:
        init_seed(config["seed"], cuda=args.cuda)

    # config_args = sorted(config.items(), key=at(0))
    logger.info("Arguments: " + ", ".join("%s: %s" % (name, val) for name, val in config.items()))
    res = []
    for _ in range(num_samples):
        res.append(runner.run(config))
        # res = [runner.run(config) for _ in range(num_samples)]
        # pickle.dump({"params": current_params, "res": res}, open(os.path.join(products_path, "quant_res.pkl"), "ab"))


if __name__ == "__main__":
    main()
