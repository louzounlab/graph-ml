import os
import pickle

import networkx as nx
import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from features_infra.graph_features import GraphFeatures
from loggers import EmptyLogger
from pygcn.layers import AsymmetricGCN, GraphConvolution


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
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_to_variable(*args):
    return list(map(Variable, args))


class GraphLoader(object):
    def __init__(self, feature_meta, cuda_num=None, logger=None):
        super(GraphLoader, self).__init__()
        self._logger = EmptyLogger() if logger is None else logger
        self._cuda_num = cuda_num
        self._features_meta = feature_meta

        # self._logger.debug("Loading %s dataset...", self._dataset)
        # self._gnx = pickle.load(open(os.path.join(self.dataset_path, "gnx.pkl"), "rb"))
        # self._content = pickle.load(open(os.path.join(self.dataset_path, "content.pkl"), "rb"))
        # self._nodes_order = sorted(self._gnx)
        self.layer_model = None
        self._cached = {}
        self._train_set = self._test_set = self._train_idx = self._test_idx = None

    @staticmethod
    def _encode_onehot_gnx(gnx, nodes_order):  # gnx, nodes_order: list = None):
        labels = gnx.graph["node_labels"]
        labels_dict = {c: np.identity(len(labels))[i, :] for i, c in enumerate(labels)}
        labels_dict.update({i: labels_dict[c] for i, c in enumerate(labels)})
        return np.array(list(map(lambda n: labels_dict[gnx.node[n]['label']], nodes_order)), dtype=np.int32)

    def split_data(self, test_p, path_iter):
        all_nodes = set()
        common_nodes = None
        for path in path_iter:
            gnx = pickle.load(open(os.path.join(path, "orig_gnx.pkl"), "rb"))
            all_nodes = all_nodes.union(gnx)
            if common_nodes is None:
                common_nodes = set(gnx)
            else:
                common_nodes = common_nodes.intersection(gnx)

        nodes_order = sorted(all_nodes)
        indexes = [(i, node) for i, node in enumerate(nodes_order)]
        common, uncommon = [], []
        for i, node in indexes:
            cur_list = common if node in common_nodes else uncommon
            cur_list.append((i, node))
        c_nodes, c_idx = zip(*common)
        c_train, c_test, c_train_idx, c_test_idx = train_test_split(c_nodes, c_idx, test_size=test_p, shuffle=True)
        uc_nodes, uc_idx = zip(*uncommon)
        uc_train, uc_test, uc_train_idx, uc_test_idx = train_test_split(uc_nodes, uc_idx, test_size=test_p, shuffle=True)

        self._train_set = set(c_train).union(uc_train)
        self._test_set = set(c_test).union(uc_test)
        self._test_idx = c_test_idx + uc_test_idx
        self._train_idx = c_train_idx + uc_train_idx

    def activate_cuda(self, items):
        if self._cuda_num is None:
            return items
        return [x.cuda(self._cuda_num) for x in items]

    def load(self, data_type, feature_path):
        # feat_x = np.vstack([self._content[node] for node in self._nodes_order]).astype(np.float32)  # BOW
        # feat_x = normalize(feat_x)
        gnx = pickle.load(open(os.path.join(feature_path, "gnx.pkl"), "rb"))
        nodes_order = sorted(gnx)

        if data_type in ["symmetric", "asymmetric"]:
            features = GraphFeatures(gnx, self._features_meta, dir_path=feature_path, logger=self._logger)
            features.build(include=self._train_set)

            add_ones = bool(set(self._features_meta).intersection(["first_neighbor_histogram",
                                                                   "second_neighbor_histogram"]))
            topo_x = features.to_matrix(add_ones=add_ones, dtype=np.float32, mtype=np.matrix, should_zscore=True)
            # topo_x /= 1000
            # feature_mx = np.hstack([feature_mx, measures_mx])
        else:
            topo_x = np.matrix([], dtype=np.float32)  # .reshape(feat_x.shape[0], 0)

        feat_x = np.matrix([], dtype=np.float32).reshape(topo_x.shape[0], 0)

        labels = self._encode_onehot_gnx(gnx, nodes_order)

        adj = nx.adjacency_matrix(gnx, nodelist=nodes_order).astype(np.float32)

        if data_type == "symmetric":
            adj, self.layer_model = handle_matrix_symmetric(adj), GraphConvolution
            # feature_mx = normalize(feature_mx)
            # feature_mx[:, measures_mx.shape[1]:] = normalize(feature_mx[:, measures_mx.shape[1]:])
        elif data_type == "asymmetric":
            adj1, self.layer_model = handle_matrix_concat(adj, should_normalize=True), AsymmetricGCN
            adj2 = handle_matrix_symmetric(adj)
            # feature_mx[:, measures_mx.shape[1]:] = normalize(feature_mx[:, measures_mx.shape[1]:])
        elif data_type == "content":
            adj, self.layer_model = handle_matrix_symmetric(adj), GraphConvolution
            # feature_mx = normalize(feature_mx)
        else:
            raise ValueError("data_type should be [symmetric| asymmetric| content")

        if feat_x:
            feat_x = torch.FloatTensor(feat_x)
        else:
            feat_x = torch.FloatTensor()
        topo_x = torch.FloatTensor(topo_x)  # np.array(features.todense()))

        labels = torch.LongTensor(np.where(labels)[1])
        adj1 = sparse_mx_to_torch_sparse_tensor(adj1).to_dense()
        adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()

        train_idx = torch.LongTensor(self._train_idx)
        test_idx = torch.LongTensor(self._test_idx)

        adj1, adj2, feat_x, topo_x, labels = convert_to_variable(adj1, adj2, feat_x, topo_x, labels)
        return self.activate_cuda([adj1, adj2, feat_x, topo_x, labels, train_idx, test_idx])

    def load_cached(self, key, *args, **kwargs):
        if key not in self._cached:
            self._cached[key] = self.load(*args, **kwargs)
        return self._cached[key]
