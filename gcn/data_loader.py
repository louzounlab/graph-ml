import os
import pickle

import networkx as nx
import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split

from features_infra.graph_features import GraphFeatures
from layers import AsymmetricGCN, GraphConvolution
from loggers import EmptyLogger

DTYPE = np.float32


def symmetric_normalization(mx):
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum != 0] **= -0.5
    r_inv = rowsum.flatten()
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)  # D^-0.5 * X * D^-0.5


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def handle_matrix_concat(mx, should_normalize=True):
    mx += sparse.eye(mx.shape[0])
    mx_t = mx.transpose()
    if should_normalize:
        mx = symmetric_normalization(mx)
        mx_t = symmetric_normalization(mx_t)

    return sparse.vstack([mx, mx_t])  # vstack: below, hstack: near


def handle_matrix_symmetric(mx):
    # build symmetric adjacency matrix
    mx += (mx.T - mx).multiply(mx.T > mx)
    return symmetric_normalization(mx + sparse.eye(mx.shape[0]))


def sparse_mx_to_torch_sparse_tensor(sparse_mx) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(DTYPE)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# if "citeseer" == dataset:
#     Taking the largest
#     max_subgnx = max(nx.connected_component_subgraphs(self._gnx.to_undirected()), key=len)
#     self._gnx = self._gnx.subgraph(max_subgnx)
class GraphLoader(object):
    def __init__(self, data_dir, features_meta, is_max_connected=False, cuda_num=None, logger=None):
        super(GraphLoader, self).__init__()
        self._logger = EmptyLogger() if logger is None else logger
        self._data_path = data_dir
        self._cuda_num = cuda_num
        self._features_meta = features_meta
        self._is_max_connected = is_max_connected

        self._logger.debug("Loading %s dataset...", self._data_path)
        features_path = self._features_path()
        self._gnx = pickle.load(open(os.path.join(features_path, "gnx.pkl"), "rb"))

        self._nodes_order = sorted(self._gnx)
        self._labels = {i: label for i, label in enumerate(self._gnx.graph["node_labels"])}
        self._ident_labels = self._encode_onehot_gnx()

        self._content = pickle.load(open(os.path.join(self._data_path, "content.pkl"), "rb"))
        bow_mx = np.vstack([self._content[node] for node in self._nodes_order]).astype(DTYPE)
        self._bow_mx = normalize(bow_mx)
        self._topo_mx = None

        # Adjacency matrices
        adj = nx.adjacency_matrix(self._gnx, nodelist=self._nodes_order).astype(DTYPE)
        self._adj = handle_matrix_symmetric(adj)
        self._adj = sparse_mx_to_torch_sparse_tensor(self._adj).to_dense()
        self._adj_rt = handle_matrix_concat(adj, should_normalize=True)
        self._adj_rt = sparse_mx_to_torch_sparse_tensor(self._adj_rt).to_dense()

        self._train_set = self._test_set = None
        self._train_idx = self._test_idx = self._base_train_idx = None
        self._val_idx = None

    def _activate_cuda(self, *items):
        if self._cuda_num is None:
            return items
        if 1 == len(items):
            return items[0].cuda(self._cuda_num)
        return [x.cuda(self._cuda_num) for x in items]

    def _encode_onehot_gnx(self):  # gnx, nodes_order: list = None):
        ident = np.identity(len(self._labels))
        if self._gnx.graph.get('is_index_labels', False):
            labels_dict = {label: ident[i, :] for i, label in self._labels.items()}
        else:
            labels_dict = {i: ident[i, :] for i, label in self._labels.items()}
        return np.array(list(map(lambda n: labels_dict[self._gnx.node[n]['label']], self._nodes_order)), dtype=np.int32)

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def labels(self):
        labels = torch.LongTensor(np.where(self._ident_labels)[1])
        return self._activate_cuda(labels)

    @property
    def train_idx(self):
        train_idx = torch.LongTensor(self._train_idx)
        return self._activate_cuda(train_idx)

    @property
    def val_idx(self):
        val_idx = torch.LongTensor(self._val_idx)
        return self._activate_cuda(val_idx)

    @property
    def test_idx(self):
        test_idx = torch.LongTensor(self._test_idx)
        return self._activate_cuda(test_idx)

    @property
    def bow_mx(self):
        bow_feat = torch.FloatTensor(self._bow_mx)
        return self._activate_cuda(bow_feat)

    @property
    def topo_mx(self):
        assert self._topo_mx is not None, "Split train required"
        topo_feat = torch.FloatTensor(self._topo_mx)
        return self._activate_cuda(topo_feat)

    @property
    def adj_rt_mx(self):
        return self._activate_cuda(self._adj_rt.clone())

    @property
    def adj_mx(self):
        return self._activate_cuda(self._adj.clone())

    def split_test(self, test_p):
        indexes = range(len(self._nodes_order))
        self._train_set, test_set, self._base_train_idx, self._test_idx = train_test_split(self._nodes_order, indexes,
                                                                                           test_size=test_p,
                                                                                           shuffle=True)
        # test_set unused

    def _features_path(self):
        return os.path.join(self._data_path, "features%d" % (self._is_max_connected,))

    def split_train(self, train_p, features_meta=None):
        if features_meta is None:
            features_meta = self._features_meta
        train_set, val_set, self._train_idx, self._val_idx = train_test_split(self._train_set, self._base_train_idx,
                                                                              test_size=1 - train_p, shuffle=True)

        features_path = self._features_path()
        features = GraphFeatures(self._gnx, features_meta, dir_path=features_path,
                                 logger=self._logger, is_max_connected=self._is_max_connected)
        features.build(include=set(train_set), should_dump=False)

        add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.intersection(features_meta))
        self._topo_mx = features.to_matrix(add_ones=add_ones, dtype=np.float64, mtype=np.matrix, should_zscore=True)

        ratio = 10 ** np.ceil(np.log10(abs(np.mean(self._topo_mx) / np.mean(self._bow_mx))))
        self._topo_mx /= ratio


        # def load(self, train_p, features_meta, data_type, feature_path=None):
        #      if feature_path is None:
        #          feature_path = self.dataset_path
        #
        #      train_set, val_set, train_idx, val_idx = train_test_split(self._train_set, self._train_idx,
        #                                                                test_size=1 - train_p, shuffle=True)
        #      train_set = set(train_set)
        #
        #      if data_type in ["symmetric", "asymmetric"]:
        #          features = GraphFeatures(self._gnx, features_meta, dir_path=feature_path, logger=self._logger)
        #          features.build(include=train_set)
        #
        #          add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.difference(features_meta))
        #          topo_mx = features.to_matrix(add_ones=add_ones, dtype=DTYPE, mtype=np.matrix, should_zscore=True)
        #          topo_mx /= 1000
        #      else:
        #          topo_mx = np.matrix([], dtype=DTYPE).reshape(bow_mx.shape[0], 0)
        #
        #      labels = self._encode_onehot_gnx()
        #
        #      adj = nx.adjacency_matrix(self._gnx, nodelist=self._nodes_order).astype(DTYPE)
        #
        #      if data_type == "symmetric":
        #          adj, self.layer_model = handle_matrix_symmetric(adj), GraphConvolution
        #      elif data_type == "asymmetric":
        #          adj1, self.layer_model = handle_matrix_concat(adj, should_normalize=True), AsymmetricGCN
        #          adj2 = handle_matrix_symmetric(adj)
        #      elif data_type == "content":
        #          adj, self.layer_model = handle_matrix_symmetric(adj), GraphConvolution
        #      else:
        #          raise ValueError("data_type should be [symmetric| asymmetric| content")
        #
        #      bow_feat = torch.FloatTensor(bow_mx)
        #      topo_feat = torch.FloatTensor(topo_mx)
        #
        #      labels = torch.LongTensor(np.where(labels)[1])
        #      adj1 = sparse_mx_to_torch_sparse_tensor(adj1).to_dense()
        #      adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()
        #
        #      # train_idx = torch.LongTensor(train_idx)
        #      # val_idx = torch.LongTensor(val_idx)
        #      # idx_test = torch.LongTensor(self._test_idx)
        #
        #      return self.activate_cuda(adj1, adj2, bow_feat, topo_feat, labels, train_idx, val_idx, idx_test)
