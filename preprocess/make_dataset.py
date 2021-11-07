from IPython.display import HTML, display
from prefect import task
from stellargraph import datasets


@task
def cora_dataset():
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)
    return graph
