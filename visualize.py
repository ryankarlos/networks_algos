import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.generators.ego import ego_graph
from pyvis.network import Network
from sklearn.decomposition import PCA


def draw_network(G, ax, edge_list=None, color="red"):
    if edge_list is not None:
        G = nx.Graph()
        G.add_edges_from(edge_list)
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, node_color=color, ax=ax)
        print(nx.info(G))
        return G
    else:
        nx.draw_networkx(G, node_color=color)
        print(nx.info(G))
        return G


def plot_subgraph(G, node_list, ax):
    subgraph = G.subgraph(node_list)
    diameter = nx.diameter(subgraph)
    print("Network diameter of largest component:", diameter)
    draw_network(subgraph, color="red", ax=ax)
    return subgraph


def plot_ego_network(G, n, radius, ax):
    """
    plot ego network around a node n depending
    on radius setting i.e. only include upto
    n nodes directly or indirectly connected to
    this node
    """
    ego_nx = ego_graph(G, n, radius=radius)
    draw_network(ego_nx, ax=ax, color="red")
    return ego_nx


def plot_centrality_hist(centrality, name):
    plt.figure(figsize=(15, 8))
    plt.hist(centrality.values(), bins=60)
    plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])
    plt.title(f"Histogram - {name} ", fontdict={"size": 35}, loc="center")
    plt.xlabel(f"{name}", fontdict={"size": 20})
    plt.ylabel("Counts", fontdict={"size": 20})
    plt.show()


def plot_interactive_network(model, title):
    net = Network(notebook=True)
    net.from_nx(model)
    net.show_buttons(filter=["physics"])
    filename = f"{title}.html"
    net.show(filename)
    return net


def plot_community_class_count(communities):
    count_list = []
    class_list = []
    for i, c in enumerate(communities):
        class_list.append(i)
        count_list.append(len(list(c)))

    df = pd.DataFrame({"class": class_list, "count": count_list})
    df.plot.bar(x="class", y="count")
    return df


def plot_link_features_projection(n_components, link_features, labels_test):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(link_features)
    plt.figure(figsize=(16, 12))
    col = []
    for label in labels_test:
        if label == 1:
            col.append("red")
        else:
            col.append("blue")

    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=col,
        alpha=0.5,
    )
    plt.show()


def plot_shortest_paths_hist(frequencies):
    plt.figure(figsize=(15, 8))
    plt.bar(x=[i + 1 for i in range(8)], height=frequencies)
    plt.title(
        "Percentages of Shortest Path Lengths", fontdict={"size": 35}, loc="center"
    )
    plt.xlabel("Shortest Path Length", fontdict={"size": 22})
    plt.ylabel("Percentage", fontdict={"size": 22})
    plt.show()
