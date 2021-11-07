import networkx as nx
from iteration_utilities import duplicates
from sklearn.model_selection import train_test_split
from stellargraph.data import EdgeSplitter

from ..utils.log import get_logger

LOG = get_logger(__name__)


def get_edge_list(G, sort=True):
    if sort:
        # sort by node 1
        sorted_edges = sorted(G.edges(), key=lambda x: x[0], reverse=True)
        return sorted_edges
    else:
        return list(G.edges())


def list_duplicate_edges(G):
    dup = list(duplicates(get_edge_list(G)))
    if len(dup) > 0:
        LOG.info(f"{len(dup)} duplicates found")
    return dup


def remove_nodes_with_low_degree(G, n):
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    # keep top n nodes
    node_list_remove = [node for node, v in degree_sorted[n::]]
    G.remove_nodes_from(node_list_remove)
    return G


def set_attributes(G, **kwargs):
    for key in kwargs.keys():
        nx.set_node_attributes(G, kwargs[key], key)
    return G


def edge_splitter_graph_train_test(graph):
    edge_splitter_test = EdgeSplitter(graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global"
    )

    print(graph_test.info())

    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    print(graph_train.info())

    return (
        graph_train,
        graph_test,
        examples_train,
        examples_test,
        examples_model_selection,
        labels_train,
        labels_test,
        labels_model_selection,
    )


def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]
