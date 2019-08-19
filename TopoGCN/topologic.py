import torch
from scipy.spatial.distance import cdist
from graph_measures.features_infra.graph_features import GraphFeatures
import numpy as np
import os
from graph_measures.loggers import PrintLogger
from graph_measures.feature_meta import NODE_FEATURES, NEIGHBOR_FEATURES
import networkx as nx
from sklearn.neighbors import NearestNeighbors

Threshold = 5
Print_Time_Log = False


def create_topological_features(nodes, edges, labels, features=None, directed=True, train_set=None, data_path=None,
                                neighbors=True):
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    g.graph["node_labels"] = list(set(labels.tolist()))
    for x in nodes:
        g.nodes[x]['label'] = labels[x].item()

    train_set = [i for i, x in enumerate(train_set) if x.item() != 0]

    topo_mx = get_topo_features(g, features_meta=features, train_set=train_set, data_path=data_path,
                                neighbors=neighbors)
    # neighbors_mx = get_nbr_features(g, train_set=train_set, data_path=data_path)

    return topo_mx


def get_topo_features(gnx, features_meta=None, train_set=None, data_path='.', logger=None, neighbors=True):
    if logger is None:
        logger = PrintLogger("MyLogger")
    if features_meta is None:
        features_meta = NODE_FEATURES.copy()
        if neighbors:
            features_meta.update(NEIGHBOR_FEATURES)
    if train_set is not None:
        train_set = set(train_set)

    features_path = os.path.join(data_path, "features{}".format(int(gnx.is_directed())))
    features = GraphFeatures(gnx, features_meta, dir_path=features_path,
                             logger=logger, is_max_connected=False)
    features.build(include=train_set, should_dump=True, print_time=Print_Time_Log)

    feat_mx = features.to_matrix(dtype=np.float64, mtype=np.matrix, should_zscore=True)

    # replace all nan values of attractor basin to 100
    feat_mx[np.isnan(feat_mx)] = 100

    return feat_mx


def create_topological_adjacency(feat_mx, threshold=Threshold):
    inv_cov = np.linalg.pinv(np.cov(np.vstack([feat_mx, feat_mx]).T))
    dists = cdist(feat_mx, feat_mx, 'mahalanobis', VI=inv_cov)
    weights = 1 / dists
    np.fill_diagonal(weights, 0)
    max_w = weights[~np.isinf(weights)].max()
    weights[np.isinf(weights)] = max_w * 2
    mean_w = weights.mean()
    topo_adj = np.where(weights < threshold, 0, 1)
    # topo_adj = np.where(weights < mean_w * threshold, 0, 1)
    # topo_adj = np.where(weights < mean_w*5, 0, weights)

    return topo_adj


def create_topological_edges(topo_mx):
    topo_adj = create_topological_adjacency(topo_mx)
    edge_list = [(i, j) for i in range(len(topo_adj)) for j in range(i) if topo_adj[i, j] != 0]
    edges_idx = torch.LongTensor(list(zip(*edge_list)))
    return edges_idx


def create_knn_neighbors(topo_mx, neighbors=4, directed=False):
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(topo_mx)
    nbrs_mx = nbrs.kneighbors(topo_mx)[1]
    edge_list = []
    for source, targets in enumerate(nbrs_mx):
        for target in targets:
            if directed:
                edge_list.append((source, target))
            elif target >= source:
                edge_list.append((source, target))
    edges_idx = torch.LongTensor(list(zip(*edge_list)))
    return edges_idx
