import logging
from collections import Counter

import numpy as np
import torch

from torchnet.meter import AUCMeter
from torchnet.meter import ConfusionMeter


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def confusion_matrix(output, labels, num_labels):
    conf_meter = ConfusionMeter(num_labels)
    auc_meter = AUCMeter()
    preds = output.max(1)[1].type_as(labels)
    conf_meter.add(preds.data.squeeze(), labels.type(torch.LongTensor).data)
    auc_meter.add(preds.data.squeeze(), labels.data.squeeze())
    return conf_meter, auc_meter


class ModelMeter:
    def __init__(self, labels, logger=None):
        # self._labels = labels
        self._logger = logger
        self._loss = []
        self._acc = []
        self._labels = labels
        self._conf = ConfusionMeter(len(self._labels))
        # self._auc = {label: AUCMeter() for label in range(len(labels))}
        self._auc = AUCMeter()
        self._c_inp, self._c_tar = Counter(), Counter()

    def update(self, loss, output, targets):
        self._loss.append(loss)
        self._acc.append(accuracy(output, targets))

        preds = output.max(1)[1].type_as(targets)
        p = preds.cpu().data.numpy()
        t = targets.cpu().data.numpy()
        cur_indexes = np.logical_and(p != 2, t != 2)
        self._auc.add(p[cur_indexes], t[cur_indexes])
        self._conf.add(preds.data.squeeze(), targets.data.squeeze())

        self._c_inp.update(p)
        self._c_tar.update(t)
        # preds = preds[preds != 2 and (targets != 2)]
        # for label, meter in self._auc.items():
        #     cur_pred = preds.eq(label)
        #     cur_label = targets.eq(label)
        #     meter.add(cur_pred.data.squeeze(), cur_label.data.squeeze())

    def log(self, msg, level=logging.DEBUG, func=None):
        if func is None:
            loss, acc = self._loss[-1].data[0], self._acc[-1].data[0]
        else:
            loss, acc = func(self._loss).data[0], func(self._acc).data[0]
        self._logger.log(level, "%s: loss: %3.4f, acc: %3.4f", msg, loss, acc)

    @property
    def last_acc(self):
        return self._acc[-1].data[0]

    @property
    def last_loss(self):
        return self._loss[-1].data[0]
