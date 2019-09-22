import subprocess
import numpy as np
import torch
import torch_geometric
import matplotlib.pyplot as plt
import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor, PPI
from torch_geometric.data import DataLoader, Data
import torch.optim as optim
import copy
import os
import timeit
import argparse

from gcn.layers import AsymmetricGCN
from gcn.data_loader import handle_matrix_symmetric, sparse_mx_to_torch_sparse_tensor, handle_matrix_concat
from topologic import create_topological_features, create_topological_edges, create_knn_neighbors
from graph_measures.feature_meta import NODE_FEATURES, NEIGHBOR_FEATURES
import torch_geometric.transforms as T
import nni
import time
from scipy import stats
from sklearn.model_selection import train_test_split
from gcn.models import GCN, GCNCombined
import networkx as nx
import pickle
from graph_measures.loggers import PrintLogger


DataSetName = "Cora"              # "Cora", "CiteSeer", "PubMed", "CoraFull", "CS", "Physics"
Net = "siam_gat"                # "gcn", "siam", "gat", "siam_gat", "combined", "symmetric", "asymmetric"
Trials = 100
CUDA_Device = 1
Logger = PrintLogger()
Split = "standard"               # "standard", "20_30" , "percent", None

# if Split is None, specify the exact number of train, validation, and test
Train = None
Val = None
Test = None
# Train = None
# Val = 500
# Test = 500
# Train = 140
# Val = 500
# Test = 1000
# Train = 0.05
# Val = None
# Test = None

IsNNI = False
MinValForNNI = 0.77


# original GCN
class GCNNet(nn.Module):

    def __init__(self, num_features, num_classes, h_layers=[16], dropout=0.5, activation="relu"):
        super(GCNNet, self).__init__()

        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(h_layers[0], num_classes)

        self._drop_out = dropout
        self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, topo_edges=None):
        x, edge_index = data.x, data.edge_index

        x = self._conv1(x, edge_index)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._drop_out, training=self.training)
        x = self._conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Combined GCN
class CombinedGCN(nn.Module):

    def __init__(self, num_features, num_classes, h_layers=[100, 35], dropout=0.5, activation="relu", num_topology=0):
        super(CombinedGCN, self).__init__()

        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(h_layers[0] + num_topology, h_layers[1])
        self._conv3 = GCNConv(h_layers[1], num_classes)

        self._drop_out = dropout
        self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, topo_mx):
        x, edge_index = data.x, data.edge_index

        x = self._conv1(x, edge_index)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._drop_out, training=self.training)

        x = torch.cat([x, topo_mx], dim=1)
        x = self._conv2(x, edge_index)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._drop_out, training=self.training)

        x = self._conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

# T-GCN
# siam net do separately GCN on original edges, and topological edges using the same input.
# then concat both outputs and perform another GCN using original edges
class SiamNet(nn.Module):

    def __init__(self, num_features, num_classes, h_layers=[16, 16], dropout=0.5, activation="relu"):
        super(SiamNet, self).__init__()

        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(num_features, h_layers[1])
        if len(h_layers) == 3:
            self._conv3 = GCNConv(h_layers[0] + h_layers[1], h_layers[2])
            self._conv4 = GCNConv(h_layers[2], num_classes)
        else:
            self._conv3 = None
            self._conv4 = GCNConv(h_layers[0] + h_layers[1], num_classes)

        self._dropout = dropout
        if activation == 'tanh':
            self._activation_func = F.tanh
        else:
            self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, topo_edges):

        x, edge_index = data.x, data.edge_index

        x1 = x.clone()
        x1 = self._activation_func(self._conv1(x1, edge_index))
        x1 = F.dropout(x1, p=self._dropout, training=self.training)

        x2 = x.clone()
        x2 = self._activation_func(self._conv2(x2, topo_edges))
        x2 = F.dropout(x2, p=self._dropout, training=self.training)

        x = torch.cat((x1, x2), dim=1)
        if self._conv3 is not None:
            x = self._activation_func(self._conv3(x, edge_index))
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


# original GAT
class GatNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, h_layers=[8], dropout=0.6, activation="elu", heads=[8,1]):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(num_features, h_layers[0], heads=heads[0], dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(h_layers[0] * heads[0], num_classes, heads=heads[1], concat=False, dropout=dropout)

        self._dropout = dropout
        if activation == 'tanh':
            self._activation_func = F.tanh
        elif activation == 'elu':
            self._activation_func = F.elu
        else:
            self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, topo_edges=None):
        x = F.dropout(data.x, p=self._dropout, training=self.training)
        x = self._activation_func(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

# T-GAT
# siam gat of original and topological edges
class SiamGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, h_layers=[8, 8], dropout=0.6, activation="elu", heads=[8,8,1]):
        super(SiamGAT, self).__init__()
        self._conv1 = GATConv(num_features, h_layers[0], heads=heads[0], dropout=dropout)
        self._conv2 = GATConv(num_features, h_layers[1], heads=heads[1], dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self._conv3 = GATConv(h_layers[0] * heads[0] + h_layers[1] * heads[1], num_classes, heads=heads[2], concat=False, dropout=dropout)

        self._dropout = dropout
        if activation == 'tanh':
            self._activation_func = F.tanh
        elif activation == 'elu':
            self._activation_func = F.elu
        else:
            self._activation_func = F.relu

    def forward(self, data: torch_geometric.data, topo_edges):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(data.x, p=self._dropout, training=self.training)

        x1 = x.clone()
        x1 = self._activation_func(self._conv1(x1, edge_index))
        x1 = F.dropout(x1, p=self._dropout, training=self.training)

        x2 = x.clone()
        x2 = self._activation_func(self._conv2(x2, topo_edges))
        x2 = F.dropout(x2, p=self._dropout, training=self.training)

        x = torch.cat((x1, x2), dim=1)
        x = self._conv3(x, edge_index)

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

        # choosing GPU device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device != "cpu":
            torch.cuda.empty_cache()
            if not self._params['is_nni']:
                self._device = torch.device("cuda:{}".format(CUDA_Device))

        self._topo_edges = None
        self._external_edges = None

        self._num_features = None
        self._num_classes = None

        self._topo_mx = None
        self._adjacency = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None

    def load_data(self):
        data_name = self._params['data_name']
        if self._params['net'] in {'combined', 'symmetric', 'asymmetric', 'combined_gcn'}:
            self._data_path = './data/{}'.format(data_name)
            gnx = nx.read_gpickle("./data/{}/gnx.pkl".format(data_name))
            bow = pickle.load(open("./data/{}/content.pkl".format(data_name), "rb"))
            nodes = sorted(gnx.nodes)
            dict = {x: i for i, x in enumerate(nodes)}
            x = torch.Tensor(np.vstack([bow[node] for node in nodes])).to(self._device)
            y = torch.LongTensor([gnx.nodes[node]['label'] for node in nodes]).to(self._device)
            edges = torch.LongTensor(np.vstack([[dict[x[0]] for x in gnx.edges],
                                               [dict[x[1]] for x in gnx.edges]])).to(self._device)
            self._data = Data(x=x, edge_index=edges, y=y)
            self._num_features = x.shape[1]
            self._num_classes = len(gnx.graph['node_labels'])

            # Adjacency matrices
            adj = nx.adjacency_matrix(gnx, nodelist=nodes).astype(np.float32)
            if self._params['net'] == 'symmetric':
                self._adj = handle_matrix_symmetric(adj)
                self._adj = sparse_mx_to_torch_sparse_tensor(self._adj).to_dense().to(self._device)
            else:
                self._adj = handle_matrix_concat(adj, should_normalize=True)
                self._adj = sparse_mx_to_torch_sparse_tensor(self._adj).to_dense().to(self._device)

            return self._data

        data_transform = T.NormalizeFeatures() if self._params['norm'] == True else None
        self._data_path = './DataSets/{}'.format(data_name)
        if data_name == "CoraFull":
            self._data_set = CoraFull(self._data_path)
        elif data_name in {"CS", "Physics"}:
            self._data_set = Coauthor(self._data_path, data_name)
        else:
            self._data_set = Planetoid(self._data_path, data_name, data_transform)
        self._data_set.data.to(self._device)
        self._data = self._data_set[0]
        # self._data = self._data_set.data

        self._num_features = self._data_set.num_features
        self._num_classes = self._data_set.num_classes

        return self._data

    # exact amount of train, val, test is necessary
    def split_train_val_test(self, train_s=None, val_s=None, test_s=None):
        if Split == 'standard':
            self._train_mask = self._data.train_mask
            self._val_mask = self._data.val_mask
            self._test_mask = self._data.test_mask
        else:
            num_nodes = self._data.num_nodes
            if Split == "20_30":
                num_classes = self._num_classes
                train_s = 20 * num_classes
                val_s = 30 * num_classes
                test_s = num_nodes - val_s - train_s
            elif Split == "percent":
                train_s = int(np.ceil(train_s * num_nodes))
                val_s = int(np.floor((num_nodes - train_s) / 2))
                test_s = num_nodes - val_s - train_s

            self._train_mask = torch.zeros(num_nodes, device=self._device, dtype=torch.uint8)
            self._val_mask = torch.zeros(num_nodes, device=self._device, dtype=torch.uint8)
            self._test_mask = torch.zeros(num_nodes, device=self._device, dtype=torch.uint8)

            train, test = train_test_split(np.array(range(num_nodes)), test_size=test_s)
            train, val = train_test_split(train, test_size=val_s)
            if train_s != None and len(train) > train_s:
                train, _ = train_test_split(train, train_size=train_s)
            self._train_mask[train] = 1
            self._val_mask[val] = 1
            self._test_mask[test] = 1
        return None

    @property
    def data(self):
        return self._data.clone()

    def set_device(self, device):
        self._device = device
        self._data.to(self._device)
        self._topo_edges.to(self._device)
        self._external_edges.to(self._device)

    def build_architecture(self):

        self.split_train_val_test(train_s=Train, val_s=Val, test_s=Test)
        # self._train_mask = self._data.train_mask
        # self._val_mask = self._data.val_mask
        # self._test_mask = self._data.test_mask

        if self._topo_edges is None:
            self.build_topological_edges()

        if self._params['net'] == 'gcn':
            self._net = GCNNet(self._num_features, self._num_classes,
                                h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                                activation=self._params['activation'])
        elif self._params['net'] == 'siam':
            self._net = SiamNet(self._num_features, self._num_classes,
                                h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                                activation=self._params['activation'])
        elif self._params['net'] == 'gat':
            self._net = GatNet(self._num_features, self._num_classes,
                               h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                               activation=self._params['activation'])
        elif self._params['net'] == 'siam_gat':
            self._net = SiamGAT(self._num_features, self._num_classes,
                               h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                               activation=self._params['activation'], heads=self._params['gat_heads'])
        elif self._params['net'] == 'combined':
            self.build_topo_matrix()
            self._net = GCNCombined(nbow=self._num_features,
                             nfeat=self._topo_mx.shape[1],
                             hlayers=self._params['hidden_sizes'],
                             nclass=self._num_classes,
                             dropout=self._params['dropout_rate'])
        elif self._params['net'] == 'combined_gcn':
            self.build_topo_matrix()
            self._net = CombinedGCN(self._num_features, self._num_classes,
                                    h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                                    activation=self._params['activation'],
                                    num_topology=self._topo_mx.shape[1])
        elif self._params['net'] == 'symmetric':
            self.build_topo_matrix()
            self._net = GCN(nfeat=self._topo_mx.shape[1],
                             hlayers=self._params['hidden_sizes'],
                             nclass=self._num_classes,
                             dropout=self._params['dropout_rate'],
                             layer_type=None)
        elif self._params['net'] == 'asymmetric':
            self.build_topo_matrix()
            self._net = GCN(nfeat=self._topo_mx.shape[1],
                             hlayers=self._params['hidden_sizes'],
                             nclass=self._num_classes,
                             dropout=self._params['dropout_rate'],
                             layer_type=AsymmetricGCN)
        else:
            self._net = SiamNet(self._num_features, self._num_classes,
                                h_layers=self._params['hidden_sizes'], dropout=self._params['dropout_rate'],
                                activation=self._params['activation'])
        self._net.to(self._device)

        # self._criterion = nn.CrossEntropyLoss()
        self._criterion = nn.NLLLoss()
        self._optimizer = optim.Adam(self._net.parameters(), lr=self._params['learning_rate'],
                                     weight_decay=self._params['weight_decay'])

    def build_topo_matrix(self):
        nodes = list(range(self._data.num_nodes))
        edges = list(zip(self._data.edge_index[0].cpu().numpy(), self._data.edge_index[1].cpu().numpy()))
        topo_mx = create_topological_features(nodes, edges, labels=self._data.y, directed=True,
                                              train_set=self._train_mask, data_path=self._data_path,
                                              neighbors=False)
                                              # features=NEIGHBOR_FEATURES)
        self._topo_mx = torch.FloatTensor(topo_mx).to(self._device)
        return self._topo_mx

    def build_topological_edges(self):
        nodes = list(range(self._data.num_nodes))
        edges = list(zip(self._data.edge_index[0].cpu().numpy(), self._data.edge_index[1].cpu().numpy()))

        # construct edges from topological similarity
        topo_path = "./DataSets/{}/edges/topo_{}{}NN".format(self._params['data_name'],
                                                             int(self._params['directed']),
                                                             self._params['knn'][0])
        if os.path.isfile(topo_path):
            self._topo_edges = torch.load(topo_path, map_location=self._device)
        else:
            # getting normalized features
            topo_mx = create_topological_features(nodes, edges, labels=self._data.y, directed=self._params['directed'],
                                                  train_set=self._train_mask, data_path=self._data_path,
                                                  features=NODE_FEATURES)
                                                  # neighbors=self._params['neighbors_ft']
                                                  # directed=self._data.is_directed()

            if self._params['knn'][0] == 0:
                # create edges using threshold on mahalanobis distance
                self._topo_edges = create_topological_edges(topo_mx).to(self._device)
            else:
                # create edges using K-Nearest-Neighbors
                self._topo_edges = create_knn_neighbors(topo_mx, neighbors=self._params['knn'][0],
                                                        directed=self._params['directed']).to(self._device)

            if not os.path.isdir(os.path.dirname(topo_path)):
                os.mkdir(os.path.dirname(topo_path))
            torch.save(self._topo_edges, topo_path)

        # construct another type of edges
        if self._params['net'] == 'siam3':
            # construct edges from labeled neighbors similarity
            if self._params['neighbors_ft']:
                neighbors_path = "./DataSets/{}/edges/neighbors_{}{}NN".format(self._params['data_name'],
                                                                               int(self._params['directed']),
                                                                               self._params['knn'][1])
                if os.path.isfile(neighbors_path):
                    self._external_edges = torch.load(neighbors_path, map_location=self._device)
                else:
                    # getting normalized features
                    neighbors_mx = create_topological_features(nodes, edges, labels=self._data.y, directed=self._params['directed'],
                                                               train_set=self._train_mask, data_path=self._data_path,
                                                               features=NEIGHBOR_FEATURES)

                    if self._params['knn'][1] == 0:
                        # create edges using threshold on mahalanobis distance
                        self._external_edges = create_topological_edges(neighbors_mx).to(self._device)
                    else:
                        # create edges using K-Nearest-Neighbors
                        self._external_edges = create_knn_neighbors(neighbors_mx, neighbors=self._params['knn'][1],
                                                                    directed=self._params['directed']).to(self._device)

                    if not os.path.isdir(os.path.dirname(neighbors_path)):
                        os.mkdir(os.path.dirname(neighbors_path))
                    torch.save(self._external_edges, neighbors_path)
            # construct edges from extenral features similarity
            else:
                externals_path = "./DataSets/{}/edges/externals_{}{}NN".format(self._params['data_name'],
                                                                               int(self._params['directed']),
                                                                               self._params['knn'][1])
                if os.path.isfile(externals_path):
                    self._external_edges = torch.load(externals_path, map_location=self._device)
                else:
                    # getting normalized features
                    ext_feat = self._data.x.cpu().numpy()
                    stds = np.std(ext_feat, axis=0)
                    ext_feat = np.delete(ext_feat, np.where(stds == 0), axis=1)
                    ext_feat = stats.zscore(ext_feat, axis=0)

                    if self._params['knn'][1] == 0:
                        # create edges using threshold on mahalanobis distance
                        self._external_edges = create_topological_edges(ext_feat).to(self._device)
                    else:
                        # create edges using K-Nearest-Neighbors
                        self._external_edges = create_knn_neighbors(ext_feat, neighbors=self._params['knn'][1],
                                                                    directed=self._params['directed']).to(self._device)

                    if not os.path.isdir(os.path.dirname(externals_path)):
                        os.mkdir(os.path.dirname(externals_path))
                    torch.save(self._external_edges, externals_path)



        return None

    def train(self):
        self._net.train()
        labels = self._data.y

        for epoch in range(int(self._params['epochs'])):
            # start = timeit.default_timer()
            self._optimizer.zero_grad()

            if self._params['net'] == 'siam3':
                outputs = self._net(self._data, self._topo_edges, self._external_edges)
            elif self._params['net'] in {'symmetric', 'asymmetric'}:
                outputs = self._net(self._topo_mx, self._adj)
            elif self._params['net'] == 'combined':
                outputs = self._net(self._data.x, self._topo_mx, self._adj)
            elif self._params['net'] == 'combined_gcn':
                outputs = self._net(self._data, self._topo_mx)
            else:
                outputs = self._net(self._data, self._topo_edges)

            loss = self._criterion(outputs[self._train_mask], labels[self._train_mask])

            # initialize parameters for early stopping
            if epoch == 0:
                best_loss = loss
                best_model = copy.deepcopy(self._net)
                best_epoch = 0

            # torch.cuda.empty_cache()
            loss.backward()
            self._optimizer.step()

            # validation
            self._net.eval()
            if self._params['net'] == 'siam3':
                val_outputs = self._net(self._data, self._topo_edges, self._external_edges)
            elif self._params['net'] in {'symmetric', 'asymmetric'}:
                val_outputs = self._net(self._topo_mx, self._adj)
            elif self._params['net'] == 'combined':
                val_outputs = self._net(self._data.x, self._topo_mx, self._adj)
            elif self._params['net'] == 'combined_gcn':
                val_outputs = self._net(self._data, self._topo_mx)
            else:
                val_outputs = self._net(self._data, self._topo_edges)
            val_loss = self._criterion(val_outputs[self._val_mask], labels[self._val_mask])
            if val_loss < best_loss:
                best_loss = val_loss
                # torch.save(self._net, 'myASR.pth')
                best_model = copy.deepcopy(self._net)
                best_epoch = epoch + 1
            self._net.train()

            # print results
            if self._params['verbose'] == 2:
                _, pred = outputs.max(dim=1)
                correct_train = float(pred[self._train_mask].eq(labels[self._train_mask]).sum().item())
                acc_train = correct_train / self._train_mask.sum().item()
                _, val_pred = val_outputs.max(dim=1)
                correct_val = float(val_pred[self._val_mask].eq(labels[self._val_mask]).sum().item())
                acc_val = correct_val / self._val_mask.sum().item()
                correct_test = float(val_pred[self._test_mask].eq(labels[self._test_mask]).sum().item())
                acc_test = correct_test / self._test_mask.sum().item()

                print("epoch: {:3d}, train loss: {:.3f} train acc: {:.3f},"
                      " val loss: {:.3f} val acc: {:.3f}, test acc: {:.3f}".format(epoch + 1, loss, acc_train, val_loss,
                                                                                   acc_val, acc_test))

            # stop = timeit.default_timer()
            # print('finih epoch {}, time:{}'.format(epoch, stop - start))

        print("best model obtained after {} epochs".format(best_epoch))
        self._net = copy.deepcopy(best_model)
        self._net.eval()
        return

    def infer(self, data):
        self._net.eval()
        if self._params['net'] == 'siam3':
            outputs = self._net(data, self._topo_edges, self._external_edges)
        elif self._params['net'] in {'symmetric', 'asymmetric'}:
            outputs = self._net(self._topo_mx, self._adj)
        elif self._params['net'] == 'combined':
            outputs = self._net(self._data.x, self._topo_mx, self._adj)
        elif self._params['net'] == 'combined_gcn':
            outputs = self._net(self._data, self._topo_mx)
        else:
            outputs = self._net(data, self._topo_edges)
        _, pred = outputs.max(dim=1)
        return pred

    def test(self):
        self._net.eval()
        if self._params['net'] == 'siam3':
            outputs = self._net(self._data, self._topo_edges, self._external_edges)
        elif self._params['net'] in {'symmetric', 'asymmetric'}:
            outputs = self._net(self._topo_mx, self._adj)
        elif self._params['net'] == 'combined':
            outputs = self._net(self._data.x, self._topo_mx, self._adj)
        elif self._params['net'] == 'combined_gcn':
            outputs = self._net(self._data, self._topo_mx)
        else:
            outputs = self._net(self._data, self._topo_edges)
        _, pred = outputs.max(dim=1)
        correct = float(pred[self._test_mask].eq(self._data.y[self._test_mask]).sum().item())
        acc = correct / self._test_mask.sum().item()
        Logger.info("test acc: {:.3f}".format(acc))

        return acc


def run_trial(parameters):
    print("Starting Trial")
    print(parameters)
    model = Model(parameters)
    model.load_data()
    acc = []
    for _ in range(parameters['trials']):
        # try:
        model.build_architecture()
        model.train()
        acc.append(model.test())
        # except:
        #     model.set_device("cpu")
        #     # model.build_topological_edges()
        #     model.build_architecture()
        #     model.train()
        #     acc.append(model.test())

    avg_acc = np.mean(acc)
    Logger.info("average test acc: {:.3f}% \n std is: {}".format(avg_acc * 100, np.std(acc) * 100))
    # output for nni - auto ml
    if parameters['is_nni']:
        avg_acc = max(avg_acc, MinValForNNI)
        nni.report_final_result(avg_acc)

    return


# update parameters to best parameters found for the specific architecture and data
def update_params(params, net, data_name=None):
    if net == 'gcn':
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.5,
                       "hidden_sizes": [16],
                       "learning_rate": 0.01,
                       "weight_decay": 5e-4,
                       "epochs": 200,
                       "norm": False})
    elif net == 'siam' and data_name == 'Cora':
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [32, 32],
                       "learning_rate": 0.001,
                       "weight_decay": 0.01,
                       "epochs": 300,
                       "knn": [8, 8],
                       "norm": False})
    elif net == 'siam':
        params.update({"net": net,

                       "activation": "tanh",
                       "dropout_rate": 0.7,
                       "hidden_sizes": [64, 16],
                       "learning_rate": 0.01,
                       "weight_decay": 5e-4,
                       "epochs": 400,
                       "knn": [15, 15],
                       "norm": True})
    elif net == 'gat' and data_name == 'PubMed':
        params.update({"net": net,

                       "activation": "elu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [8],
                       "learning_rate": 0.01,
                       "weight_decay": 0.001,
                       "epochs": 500,
                       # "epochs": 100,
                       "gat_heads": [8, 8],
                       "norm": True})
    elif net == 'gat':
        params.update({"net": net,

                       "activation": "elu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [8],
                       "learning_rate": 0.005,
                       "weight_decay": 5e-4,
                       "epochs": 500,
                       # "epochs": 100,
                       "gat_heads": [8, 1],
                       "norm": True})
    elif net == 'siam_gat' and data_name == 'Cora':
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.7,
                       "hidden_sizes": [16, 8],
                       "learning_rate": 0.01,
                       "weight_decay": 0.001,
                       "epochs": 500,
                       "knn": [10, 10],
                       "gat_heads": [8, 8, 1],
                       "norm": True})
    elif net == 'siam_gat':
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [16, 16],
                       "learning_rate": 0.01,
                       "weight_decay": 5e-4,
                       "epochs": 400,
                       "knn": [15, 15],
                       "gat_heads": [16, 8, 8],
                       "norm": True})
    elif net == 'combined':
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [16],
                       "learning_rate": 0.01,
                       "weight_decay": 0.001,
                       "epochs": 200,
                       "norm": False})
    elif net in {"symmetric", "asymmetric"}:
        params.update({"net": net,

                       "activation": "relu",
                       "dropout_rate": 0.6,
                       "hidden_sizes": [100, 35],
                       "learning_rate": 0.01,
                       "weight_decay": 0.001,
                       "epochs": 200,
                       "norm": False})
    else:
        print("ERROR: there is no {} architecture".format(net))

    return params


# IsNNI must be True if running through Microsoft NNI platform
if __name__ == '__main__':
    # receiving parameters from Microsoft NNI platform
    if IsNNI:
        params_ = nni.get_next_parameter()
        params_.update({"is_nni": True,
                       "data_name": DataSetName,
                       "verbose": 1,
                       "trials": Trials,
                       "neighbors_ft": False,
                       "norm": True,
                       "directed": False
                       })
    else:
        params_ = {"data_name": DataSetName,

                  "net": "siam",            # available networks are: "gcn", "siam", "gat", "siam_gat"
                  "is_nni": False,          # should be True only if running through Microsoft NNI platform
                  "verbose": 1,             # verbose 2 will print scores after each epoch, verbose 1 will print scores only after all epochs
                  "trials": Trials,         # number of trials to run. output is the average score over all trials
                  "neighbors_ft": False,    # set True to use neighbors's labels feature instead of standard topological features
                  "knn": [10, 10],          # number of neighbors in the topological graph
                  "directed": False,

                  "epochs": 300,
                  "activation": "relu",
                  "dropout_rate": 0.6,
                  "hidden_sizes": [16, 16],
                  "learning_rate": 0.01,
                  "weight_decay": 0.005,

                  "norm": False,            # for gat architectures should be True
                  "gat_heads": [8,8,1],     # number of heads at each layers - only for GAT architectures
                }

    params_ = update_params(params_, net=Net, data_name=DataSetName)
    run_trial(params_)

    # nets_ = ["combined", "symmetric", "asymmetric", "gcn", "siam", "combined_gcn"]
    # for net_ in nets_:
    #     print("starting using {} ".format(net_))
    #     params_ = update_params(params_, net=net_, data_name=DataSetName)
    #     for train_s_ in range(5, 90, 10):
    #         Train = train_s_ / 100
    #         print("train size is  {}".format(Train))
    #         run_trial(params_)

    # nets_ = ["gcn", "siam", "gat", "siam_gat"]
    # for net_ in nets_:
    #     print("starting trial using {} ".format(net_))
    #     params_ = update_params(params_, net=net_, data_name=DataSetName)
    #     run_trial(params_)

    print("finish")

