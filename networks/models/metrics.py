from operator import itemgetter

import networkx as nx
import prefect
from sklearn.metrics import roc_auc_score

logger = prefect.context.get("logger")


def compute_centrality_metrics(G):
    closeness_centrality = nx.centrality.closeness_centrality(G)
    degree_centrality = nx.centrality.degree_centrality(G)
    betweenness_centrality = nx.centrality.betweenness_centrality(G)
    eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
    return (
        closeness_centrality,
        degree_centrality,
        betweenness_centrality,
        eigenvector_centrality,
    )


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def node_with_largest_degree(G):
    node_and_degree = G.degree()
    return sorted(node_and_degree, key=itemgetter(1))[-1]


def connected_components(G):
    components = nx.connected_components(G)
    print(components)
    largest_component = max(components, key=len)
    return components, largest_component
