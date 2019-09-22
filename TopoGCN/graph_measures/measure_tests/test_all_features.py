from graph_measures.features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from graph_measures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from graph_measures.features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.bfs_moments import BfsMomentsCalculator
from graph_measures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from graph_measures.features_algorithms.vertices.communicability_betweenness_centrality import \
    CommunicabilityBetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.eccentricity import EccentricityCalculator
from graph_measures.features_algorithms.vertices.fiedler_vector import FiedlerVectorCalculator
from graph_measures.features_algorithms.vertices.flow import FlowCalculator
from graph_measures.features_algorithms.vertices.general import GeneralCalculator
from graph_measures.features_algorithms.vertices.hierarchy_energy import HierarchyEnergyCalculator
from graph_measures.eatures_algorithms.vertices.k_core import KCoreCalculator
from graph_measures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from graph_measures.features_algorithms.vertices.louvain import LouvainCalculator
from graph_measures.features_algorithms.vertices.motifs import nth_edges_motif
from graph_measures.features_algorithms.vertices.page_rank import PageRankCalculator
from graph_measures.measure_tests.specific_feature_test import test_specific_feature


FEATURE_CLASSES = [
    AttractorBasinCalculator,
    AverageNeighborDegreeCalculator,
    BetweennessCentralityCalculator,
    BfsMomentsCalculator,
    ClosenessCentralityCalculator,
    CommunicabilityBetweennessCentralityCalculator,
    EccentricityCalculator,
    FiedlerVectorCalculator,
    FlowCalculator,
    GeneralCalculator,
    HierarchyEnergyCalculator,
    KCoreCalculator,
    LoadCentralityCalculator,
    LouvainCalculator,
    nth_edges_motif(3),
    PageRankCalculator,
]


def test_all():
    for cls in FEATURE_CLASSES:
        test_specific_feature(cls, is_max_connected=True)


if __name__ == "__main__":
    test_all()
