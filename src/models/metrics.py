import networkx as nx
from networkx.algorithms import community

from ..utils.log import get_logger

LOG = get_logger()


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


def shortest_path(G, source_id, target_id):
    shortest_path = nx.shortest_path(G, source=source_id, target=target_id)
    LOG.info("Shortest path between user1 and user2:", shortest_path)
    return shortest_path
