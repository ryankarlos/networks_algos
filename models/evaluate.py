import networkx as nx
import numpy as np
import prefect
from networkx.algorithms import community
from prefect import task
from sklearn.metrics import roc_auc_score

from preprocess.build_features import link_examples_to_features

logger = prefect.context.get("logger")


@task
def compute_metrics(G):
    """
    degree, betweeness, communities
    """
    degree_dict = dict(G.degree(G.nodes()))
    betweenness_dict = nx.betweenness_centrality(G)
    communities = community.greedy_modularity_communities(G)
    modularity_dict = {}
    for i, c in enumerate(communities):
        for name in c:
            modularity_dict[name] = i

    return communities, {
        "degree": degree_dict,
        "betweeness": betweenness_dict,
        "modularity": modularity_dict,
    }


@task
def shortest_path(G, source_id, target_id):
    shortest_path = nx.shortest_path(G, source=source_id, target=target_id)
    logger.info("Shortest path between user1 and user2:", shortest_path)
    return shortest_path


@task
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0
