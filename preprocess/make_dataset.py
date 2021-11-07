import pandas as pd
from IPython.display import HTML, display
from stellargraph import datasets


def cora_dataset():
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)
    return graph


def reader(path_edges, path_target):
    df_edges = pd.read_csv(path_edges)
    df_target = pd.read_csv(path_target)
    return df_edges, df_target
