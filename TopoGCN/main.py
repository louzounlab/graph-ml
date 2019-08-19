import subprocess
import numpy as np
import torch
import torch_geometric
import matplotlib.pyplot as plt
import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch.optim as optim
import copy
import os
import timeit
import argparse
from topologic import create_topological_features, create_topological_edges, create_knn_neighbors
import torch_geometric.transforms as T
import nni
import time

# available data sets are: cora, CiteSeer, PubMed
DataSetName = "cora"
# is_nni must be True if running through nni platform
IsNNI = True
SpaceNeededOnGPU = 8000


class Net(nn.Module):

    def __init__(self, num_features, num_classes, h_layers=[16], bases=5, dropout=0.5, activation="relu"):
        super(Net, self).__init__()
        hidden_layers = [num_features] + h_layers + [num_classes]
        self._layers = nn.ModuleList([RGCNConv(first, second, num_relations=2, num_bases=int(bases))
                                      for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._dropout = dropout
        if activation == 'tanh':
            self._activation_func = F.tanh
        else:
            self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, edges, edges_type):

        x, edge_index = data.x, data.edge_index

        layers = list(self._layers)
        for layer in layers[:-1]:
            x = self._activation_func(layer(x, edges, edges_type))
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = layers[-1](x, edges, edges_type)

        return F.log_softmax(x, dim=1)


class Model:

    def __init__(self, parameters):
        self._params = parameters

        self._data_set = None
        self._data = None
        self._data_path = None

        self._net = None
        self._criterion = None
        self._optimizer = None

        # choosing device
        torch.cuda.empty_cache()
        gpu_available_memmory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                         '--format=csv,nounits,noheader'])
        gpu_memory = [int(x) for x in gpu_available_memmory.strip().split()]
        available_gpu = [i for i, x in enumerate(gpu_memory) if x>=SpaceNeededOnGPU]

        if self._params['is_nni']:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self._device.type == 'cuda' and torch.cuda.current_device() not in available_gpu:
                self._device = "cpu"
        else:
            self._device = torch.device("cuda:{}".format(available_gpu[0]) if torch.cuda.is_available() 
                                        and len(available_gpu) > 0 else "cpu")

        self._topo_edges = None
        self._edges = None
        self._edges_type = None

    def load_data(self):
        data_name = self._params['data_name']
        self._data_path = './DataSets/{}'.format(data_name)
        self._data_set = Planetoid(self._data_path, data_name)
        self._data_set.data.to(self._device)
        self._data = self._data_set.data
        return self._data_set

    @property
    def data(self):
        return self._data.clone()

    def set_device(self, device):
        self._device = device
        self._topo_edges.to(self._device)
        self._edges.to(self._device)
        self._edges_type.to(self._device)

    def build_architecture(self):
        if self._topo_edges is None:
            self.build_topological_edges()

        self._net = Net(self._data_set.num_features, self._data_set.num_classes, h_layers=self._params['hidden_sizes'],
                        bases=self._params['bases'], dropout=self._params['dropout_rate'],
                        activation=self._params['activation'])
        self._net.to(self._device)

        self._criterion = nn.NLLLoss()
        self._optimizer = optim.Adam(self._net.parameters(), lr=self._params['learning_rate'],
                                     weight_decay=self._params['weight_decay'])

    def build_topological_edges(self):
        nodes = list(range(self._data.num_nodes))
        edges = list(zip(self._data.edge_index[0].cpu().numpy(), self._data.edge_index[1].cpu().numpy()))
        topo_path = "./DataSets/{}/edges/topo_edges{}{}{}NN".format(self._params['data_name'],
                                                                    int(self._params['directed']),
                                                                    int(self._params['neighbors_ft']),
                                                                    self._params['knn'])
        if os.path.isfile(topo_path):
            self._topo_edges = torch.load(topo_path, map_location=self._device)
        else:
            # getting normalized features
            topo_mx = create_topological_features(nodes, edges, labels=self._data.y, directed=self._params['directed'],
                                                  train_set=self._data.train_mask, data_path=self._data_path,
                                                  neighbors=self._params['neighbors_ft'])
                                                  # directed=self._data.is_directed()

            if self._params['knn'] == 0:
                # create edges using threshold on mahalanobis distance
                self._topo_edges = create_topological_edges(topo_mx).to(self._device)
            else:
                # create edges using K-Nearest-Neighbors
                self._topo_edges = create_knn_neighbors(topo_mx, neighbors=self._params['knn'],
                                                        directed=self._params['directed']).to(self._device)

            if not os.path.isdir(os.path.dirname(topo_path)):
                os.mkdir(os.path.dirname(topo_path))
            torch.save(self._topo_edges, topo_path)

        # edges for Relational GCN
        self._edges = torch.cat((self._data.edge_index, self._topo_edges), dim=1)
        self._edges_type = torch.ones(self._edges.shape[1]).long().to(self._device)
        self._edges_type[list(range(self._data.num_edges))] = 0

        return None

    def train(self):
        self._net.train()
        labels = self._data.y

        for epoch in range(int(self._params['epochs'])):
            # start = timeit.default_timer()
            self._optimizer.zero_grad()
            outputs = self._net(self._data, self._edges, self._edges_type)
            # outputs = self._net(self._data, self._topo_edges)

            loss = self._criterion(outputs[self._data.train_mask], labels[self._data.train_mask])

            # initialize parameters for early stopping
            if epoch == 0:
                best_loss = loss
                best_model = copy.deepcopy(self._net)
                best_epoch = 0

            torch.cuda.empty_cache()
            loss.backward()
            self._optimizer.step()

            # validation
            self._net.eval()
            val_outputs = self._net(self._data, self._edges, self._edges_type)
            # val_outputs = self._net(self._data, self._topo_edges)
            val_loss = self._criterion(val_outputs[self._data.val_mask], labels[self._data.val_mask])
            if val_loss < best_loss:
                best_loss = val_loss
                # torch.save(self._net, 'myASR.pth')
                best_model = copy.deepcopy(self._net)
                best_epoch = epoch + 1
            self._net.train()

            # print results
            if self._params['verbose'] == 2:
                _, pred = outputs.max(dim=1)
                correct_train = float(pred[self._data.train_mask].eq(labels[self._data.train_mask]).sum().item())
                acc_train = correct_train / self._data.train_mask.sum().item()
                _, val_pred = val_outputs.max(dim=1)
                correct_val = float(val_pred[self._data.val_mask].eq(labels[self._data.val_mask]).sum().item())
                acc_val = correct_val / self._data.val_mask.sum().item()

                print("epoch: {:3d}, train loss: {:.3f} train acc: {:.3f},"
                      " val loss: {:.3f} val acc: {:.3f}".format(epoch + 1, loss, acc_train, val_loss, acc_val))

            # stop = timeit.default_timer()
            # print('finih epoch {}, time:{}'.format(epoch, stop - start))

        print("best model obtained after {} epochs".format(best_epoch))
        self._net = copy.deepcopy(best_model)
        self._net.eval()
        return

    def infer(self, data):
        self._net.eval()
        outputs = self._net(data, self._edges, self._edges_type)
        # outputs = self._net(data, self._topo_edges)
        _, pred = outputs.max(dim=1)
        return pred

    def test(self):
        self._net.eval()
        outputs = self._net(self._data, self._edges, self._edges_type)
        _, pred = outputs.max(dim=1)
        correct = float(pred[self._data.test_mask].eq(self._data.y[self._data.test_mask]).sum().item())
        acc = correct / self._data.test_mask.sum().item()
        print("test acc: {:.3f}".format(acc))

        return acc


def run_trial(parameters):
    model = Model(parameters)
    model.load_data()
    acc = []
    for _ in range(parameters['trials']):
        try:
            model.build_architecture()
            model.train()
            acc.append(model.test())
        except:
            model.set_device("cpu")
            # model.build_topological_edges()
            model.build_architecture()
            model.train()
            acc.append(model.test())

    avg_acc = np.mean(acc)
    print("average test acc: {:.3f}".format(avg_acc))
    # output for nni - auto ml
    if parameters['is_nni']:
        nni.report_final_result(avg_acc)

    return


if __name__ == '__main__':
    if IsNNI:
        params = nni.get_next_parameter()
        params.update({"is_nni": True})
    else:
        params = {"neighbors_ft": True,
                  "activation": "relu",
                  "dropout_rate": 0.5,
                  "bases": 5,
                  "epochs": 200,
                  "hidden_sizes": [16],
                  "learning_rate": 0.01,
                  "weight_decay": 5e-4,
                  "directed": True,
                  "knn": 4,
                  "is_nni": False
                  }
    params.update({"data_name": DataSetName,
                   "verbose": 2,
                   "trials": 2})

    run_trial(params)
    print("finish")

