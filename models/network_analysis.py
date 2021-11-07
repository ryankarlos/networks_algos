import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def community_detection(G):
    colors = ["" for x in range(G.number_of_nodes())]
    counter = 0
    for com in nx.community.label_propagation_communities(G):
        color = "#%06X" % np.randint(0, 0xFFFFFF)
        counter += 1
        for node in list(com):
            colors[node] = color
    print(f"Number of communities detected: {counter}")


def shortest_paths(G):
    shortest_paths = nx.shortest_path(G)
    frequencies = [0 for i in range(nx.diameter(G))]
    for node_start in shortest_paths.keys():
        for path in shortest_paths.get(node_start).values():
            path_length = len(path) - 1
            if path_length > 0:
                frequencies[path_length - 1] += 1
    frequencies = [num / sum(frequencies) for num in frequencies]
    return frequencies


def clustering_effects(G):
    nx.average_clustering(G)
    plt.figure(figsize=(15, 8))
    plt.hist(nx.clustering(G).values(), bins=50)
    plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
    plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
    plt.ylabel("Counts", fontdict={"size": 20})
    plt.show()


def closeness_centrality(G):
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
