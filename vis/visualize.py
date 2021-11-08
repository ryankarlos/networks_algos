import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.generators.ego import ego_graph
from pyvis.network import Network
from sklearn.decomposition import PCA


def plot_network_with_edge_weights(G, figsize=(10, 10)):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if (d["weight"] > 0.8)]
    emedium = [
        (u, v) for (u, v, d) in G.edges(data=True) if (0.8 >= d["weight"] >= 0.5)
    ]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if (d["weight"] < 0.5)]
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="red", node_size=300)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=8, alpha=0.2)
    nx.draw_networkx_edges(G, pos, edgelist=emedium, width=5, alpha=0.2)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, alpha=0.2)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        font_family="sans-serif",
        font_color="white",
    )
    plt.axis("off")
    plt.show()


def plot_ego_network(G, n, radius, **options):
    """
    plot ego network around a node n depending
    on radius setting i.e. only include upto
    n nodes directly or indirectly connected to
    this node
    """
    hub_ego = ego_graph(G, n, radius=radius)
    pos = nx.spring_layout(hub_ego)
    nx.draw(hub_ego, pos, node_color="b", node_size=50, with_labels=False)
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[n], **options)
    plt.show()
    return hub_ego


def plot_centrality_hist(centrality, name):
    plt.figure(figsize=(15, 8))
    plt.hist(centrality.values(), bins=60)
    plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])
    plt.title(f"Histogram - {name} ", fontdict={"size": 35}, loc="center")
    plt.xlabel(f"{name}", fontdict={"size": 20})
    plt.ylabel("Counts", fontdict={"size": 20})
    plt.show()


def interactive_network_vis(
    dag, *widgets, options=None, weights=False, notebook=True, directed=True
):
    nt = Network("800px", "800px", directed=directed, notebook=notebook)

    nt.from_nx(dag)
    if weights:
        for edge in nt.edges:
            edge["value"] = edge["weight"]
        if options is not None:
            nt.set_options(options=options)
            return nt
        else:
            nt.show_buttons(filter=widgets)
            return nt


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


def plot_degree_freq_log_log(G, m=0):
    degree_freq = G.degree_historgam(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees[m:], degree_freq[m:], "go-")
    plt.title("log log plot for degree freq")
    plt.xlabel("degree")
    plt.ylabel("frequency")
    plt.show()
