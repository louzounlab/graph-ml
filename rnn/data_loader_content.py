import os
import pickle

import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn.utils import rnn

from features_infra.feature_calculators import z_scoring
from features_infra.graph_features import GraphFeatures
from loggers import EmptyLogger


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


# borrowed code from https://github.com/pytorch/examples/tree/master/word_language_model
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).contiguous()
    if USE_CUDA:
        data = data.cuda()
    return data


def get_batch(data, seq_length):
    for i in range(0, data.size(1) - seq_length, seq_length):
        inputs = Variable(data[:, i: i + seq_length])
        targets = Variable(data[:, (i + 1): (i + 1) + seq_length].contiguous())
        yield (inputs, targets)


class GraphLoader(object):
    def __init__(self, data_path, feature_meta, test_p, gnx_idx=None, cuda_num=None, logger=None):
        super(GraphLoader, self).__init__()
        self._logger = EmptyLogger() if logger is None else logger
        self._gnx_idx = gnx_idx
        self._cuda_num = cuda_num
        self._test_p = test_p
        # self._features_meta = feature_meta
        self._data_path = data_path

        # self._logger.debug("Loading %s dataset...", self._dataset)
        # self._gnx = pickle.load(open(os.path.join(self.dataset_path, "gnx.pkl"), "rb"))
        # self._content = pickle.load(open(os.path.join(self.dataset_path, "content.pkl"), "rb"))
        # self._nodes_order = sorted(self._gnx)
        self._train_set = self._test_set = self._train_idx = self._test_idx = None
        self._inputs = self._targets = None
        self._nodes_order = []

        self._labels = {i: label for i, label in enumerate(self._get_labels())}
        self._prepare_data()

    def _get_labels(self):
        gnx = pickle.load(open(os.path.join(next(self._get_gnx_paths()), "gnx.pkl"), "rb"))
        return gnx.graph["node_labels"]

    def _get_labels_jsn(self):
        jsn = pickle.load(open(os.path.join(next(self._get_gnx_paths()), "content_jsn.pkl"), "rb"))
        return set([x["top"] for x in jsn.values()])

    @staticmethod
    def _encode_onehot_gnx1(gnx, nodes_order):  # gnx, nodes_order: list = None):
        labels = gnx.graph["node_labels"]
        labels_dict = {c: np.identity(len(labels))[i, :] for i, c in enumerate(labels)}
        labels_dict.update({i: labels_dict[c] for i, c in enumerate(labels)})
        return np.array(list(map(lambda n: labels_dict[gnx.node[n]['label']], nodes_order)), dtype=np.int32)

    def _encode_onehot_gnx(self, gnx, nodes_order):  # gnx, nodes_order: list = None):
        ident = np.identity(len(self._labels))
        labels_dict = {label: ident[j, :] for j, label in self._labels.items()}
        return np.array(list(map(lambda n: labels_dict[gnx.node[n]['label']], nodes_order)), dtype=np.int32)

    def _encode_onehot_jsn(self, jsn, nodes_order):
        ident = np.identity(len(self._labels))
        labels_dict = {label: ident[j, :] for j, label in self._labels.items()}
        return np.array(list(map(lambda n: labels_dict[jsn[n]["top"]], nodes_order)), dtype=np.int32)

    def _join_graphs1(self):
        all_nodes = set()
        common_nodes = None
        for path in self._get_gnx_paths():
            gnx = pickle.load(open(os.path.join(path, "orig_gnx.pkl"), "rb"))
            all_nodes = all_nodes.union(gnx)
            if common_nodes is None:
                common_nodes = set(gnx)
            else:
                common_nodes = common_nodes.intersection(gnx)

        pickle.dump(all_nodes, open(os.path.join(path, "..", "..", "all_nodes.pkl"), "wb"))
        pickle.dump(common_nodes, open(os.path.join(path, "..", "..", "common_nodes.pkl"), "wb"))
        return all_nodes, common_nodes

    def _join_graphs(self):
        path = next(self._get_gnx_paths())
        all_nodes = pickle.load(open(os.path.join(path, "..", "..", "all_nodes.pkl"), "rb"))
        common_nodes = pickle.load(open(os.path.join(path, "..", "..", "common_nodes.pkl"), "rb"))
        return all_nodes, common_nodes

    def _split_data(self):
        feat_path = os.path.join(next(self._get_gnx_paths()), "features_0")
        gnx = pickle.load(open(os.path.join(feat_path, "gnx.pkl"), "rb"))
        self._nodes_order = sorted([node for node in gnx if gnx.node[node]['label'] is not None])
        indexes = [(i, node) for i, node in enumerate(self._nodes_order)]

        idx, nodes = zip(*indexes)
        c_train, c_test, c_train_idx, c_test_idx = train_test_split(nodes, idx, test_size=self._test_p,
                                                                    shuffle=True)

        self._train_set = set(c_train)
        self._test_set = set(c_test)
        self._test_idx = np.array(c_test_idx)
        self._train_idx = np.array(c_train_idx)

    def _split_data_orig(self):
        all_nodes, common_nodes = self._join_graphs()

        self._nodes_order = sorted(all_nodes)
        indexes = [(i, node) for i, node in enumerate(self._nodes_order)]
        common, uncommon = [], []
        for i, node in indexes:
            cur_list = common if node in common_nodes else uncommon
            cur_list.append((i, node))
        c_idx, c_nodes = zip(*common)
        c_train, c_test, c_train_idx, c_test_idx = train_test_split(c_nodes, c_idx, test_size=self._test_p,
                                                                    shuffle=True)
        uc_idx, uc_nodes = zip(*uncommon)
        uc_train, uc_test, uc_train_idx, uc_test_idx = train_test_split(uc_nodes, uc_idx, test_size=self._test_p,
                                                                        shuffle=True)

        self._train_set = set(c_train).union(uc_train)
        self._test_set = set(c_test).union(uc_test)
        self._test_idx = np.array(c_test_idx + uc_test_idx)
        self._train_idx = np.array(c_train_idx + uc_train_idx)

    def _activate_cuda(self, *args):
        if self._cuda_num is None:
            return args
        return [x.cuda(self._cuda_num) for x in args]

    # firms/years/features_0-1
    def _get_gnx_paths(self):
        paths = sorted(os.listdir(self._data_path), key=int)
        if self._gnx_idx is not None:
            # for x in [4, 6]:
            #     yield os.path.join(self._data_path, paths[x])
            yield os.path.join(self._data_path, paths[self._gnx_idx])
            return
        for path in paths:
            yield os.path.join(self._data_path, path)

    def _prepare_data1(self):
        self._split_data()

        self._inputs = self._targets = None
        for path in self._get_gnx_paths():
            feat_path = os.path.join(path, "features_0")
            cur_data = pickle.load(open(os.path.join(feat_path, "data.pkl"), "rb"))
            self._inputs = cur_data if self._inputs is None else np.dstack((self._inputs, cur_data))
            cur_labels = pickle.load(open(os.path.join(feat_path, "labels.pkl"), "rb"))
            self._targets = cur_labels if self._targets is None else np.dstack((self._targets, cur_labels))
        self._inputs = self._inputs.transpose((0, 2, 1))
        self._targets = self._targets.transpose((0, 2, 1))
        self._logger.debug("Finished preparing the data")

    def _prepare_data(self):
        self._split_data()

        self._inputs = self._targets = None
        for path in self._get_gnx_paths():
            # feat_path = os.path.join(path, "features_0")
            vec = pickle.load(open(os.path.join(path, "content_vec.pkl"), "rb"))
            jsn = pickle.load(open(os.path.join(path, "content_json.pkl"), "rb"))
            # gnx = gnx.subgraph(self._nodes_order)

            # features = GraphFeatures(gnx, self._features_meta, dir_path=feat_path, logger=self._logger)
            # features.build(include=self._train_set)
            # add_ones = bool(set(self._features_meta).intersection(["first_neighbor_histogram",
            #                                                        "second_neighbor_histogram"]))
            # cur_data = features.to_matrix(add_ones=add_ones, dtype=np.float32, mtype=np.array, should_zscore=True)
            cur_data = np.vstack([vec[node] for node in self._nodes_order])
            cur_data = z_scoring(cur_data)
            self._inputs = cur_data if self._inputs is None else np.dstack((self._inputs, cur_data))
            # pickle.dump(cur_data, open(os.path.join(feat_path, "data.pkl"), "wb"))

            cur_labels = self._encode_onehot_gnx(jsn, self._nodes_order)
            self._targets = cur_labels if self._targets is None else np.dstack((self._targets, cur_labels))
            # pickle.dump(cur_labels, open(os.path.join(feat_path, "labels.pkl"), "wb"))

        # Arranging data as <batch, seq, feature>
        if self._gnx_idx is None:
            self._inputs = self._inputs.transpose((0, 2, 1))
            self._targets = self._targets.transpose((0, 2, 1))
        self._logger.debug("Finished preparing the data")
        # topo_x = torch.FloatTensor(topo_x)  # np.array(features.todense()))
        #
        # labels = torch.LongTensor(np.where(labels)[1])
        #
        # train_idx = torch.LongTensor(self._train_idx)
        # test_idx = torch.LongTensor(self._test_idx)
        #
        # topo_x, labels = convert_to_variable(topo_x, labels)
        # return self.activate_cuda([topo_x, labels])

    def _load_data(self, indexes, nbatch):
        # for inputs, targets in zip(self._inputs, self._targets):
        inputs, targets = self._inputs[indexes], self._targets[indexes]
        # for i in range(0, int(len(inputs) / nbatch) * nbatch, nbatch):
        for i in range(0, len(inputs), nbatch):
            data, labels = inputs[i: i + nbatch], targets[i: i + nbatch]
            # if self._gnx_idx is not None:
            #     data, labels = data[:, self._gnx_idx, :], labels[:, self._gnx_idx, :]
            data, labels = Variable(torch.FloatTensor(data)), Variable(
                torch.LongTensor(np.where(labels)[self.feat_dim]))
            # labels = labels[labels != reverse_labels[None]]
            yield self._activate_cuda(data, labels)

    def load_train_data(self, nbatch):
        return self._load_data(self._train_idx, nbatch)

    def load_test_data(self, nbatch):
        return self._load_data(self._test_idx, nbatch)

    @property
    def feat_dim(self):
        return 2 if self._gnx_idx is None else 1

    @property
    def num_nodes(self):
        return self._inputs.shape[0]

    @property
    def sequence_len(self):
        return self._inputs.shape[1]

    @property
    def num_features(self):
        return self._inputs.shape[self.feat_dim]

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def labels(self):
        return self._labels

    @property
    def num_layers(self):
        return [100, 20]
        # return [300, 100]

        # def bla(self):
        #     x, lengths = rnn.pad_packed_sequence(x, batch_first=True)
        #     return nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
