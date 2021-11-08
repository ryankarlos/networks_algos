import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import prefect
from gensim.models import Word2Vec
from networkx.algorithms import community, tree
from prefect import task
from stellargraph.data import BiasedRandomWalk


def community_detection(G):
    colors = ["" for x in range(G.number_of_nodes())]
    counter = 0
    for com in community.label_propagation_communities(G):
        color = "#%06X" % np.randint(0, 0xFFFFFF)
        counter += 1
        for node in list(com):
            colors[node] = color
    print(f"Number of communities detected: {counter}")


def clustering_effects(G):
    nx.average_clustering(G)
    plt.figure(figsize=(15, 8))
    plt.hist(nx.clustering(G).values(), bins=50)
    plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
    plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
    plt.ylabel("Counts", fontdict={"size": 20})
    plt.show()


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


def max_spanning_tree(G, algorithm="kruskal"):
    mst = tree.maximum_spanning_edges(G, algorithm=algorithm, data=False)
    edge_list = list(mst)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G


@task
def node2vec_embedding(graph, name, **kwargs):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(
        graph.nodes(),
        n=kwargs["num_walks"],
        length=kwargs["walk_length"],
        p=kwargs["p"],
        q=kwargs["q"],
    )
    logger = prefect.context.get("logger")
    logger.info(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=kwargs["dimensions"],
        window=kwargs["window_size"],
        min_count=0,
        sg=1,
        workers=kwargs["workers"],
        epochs=kwargs["num_iter"],
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding
