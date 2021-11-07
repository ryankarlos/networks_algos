import networkx as nx
import prefect
from sklearn.metrics import roc_auc_score

logger = prefect.context.get("logger")


def centrality(G):
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
