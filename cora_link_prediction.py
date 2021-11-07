import multiprocessing

import numpy as np
import pandas as pd
import prefect
from prefect import Flow, task

from models.link_prediction import (
    evaluate_link_prediction_model,
    node2vec_embedding,
    train_link_prediction_model,
)
from preprocess.build_features import (
    edge_splitter_graph_train_test,
    link_examples_to_features,
)
from preprocess.make_dataset import cora_dataset
from visualize import plot_link_features_projection

logger = prefect.context.get("logger")


params = {
    "p": 1.0,
    "q": 1.0,
    "dimensions": 128,
    "num_walks": 10,
    "walk_length": 80,
    "window_size": 10,
    "num_iter": 1,
    "workers": multiprocessing.cpu_count(),
}


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


@task(nout=8)
def load_data_and_construct_splits():
    graph = cora_dataset()
    return edge_splitter_graph_train_test(graph)


@task
def train_link_classifier(
    examples_train,
    labels_train,
    embedding_train,
):
    results = []
    ops = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    for op in ops:
        clf = train_link_prediction_model(
            examples_train, labels_train, embedding_train, op
        )
        results.append(
            evaluate_link_prediction_model(
                clf, examples_train, labels_train, embedding_train, op
            )
        )
    return results


@task(nout=2)
def compute_best_results(results):
    best_result = max(results, key=lambda result: result["score"])
    logger.info(f"Best result from '{best_result['binary_operator'].__name__}'")
    print(
        pd.DataFrame(
            [
                (result["binary_operator"].__name__, result["score"])
                for result in results
            ],
            columns=("name", "ROC AUC score"),
        ).set_index("name")
    )
    return best_result, best_result["binary_operator"].__name__


@task
def evaluate_on_test_data(examples_test, labels_test, embedding_test, best_result):
    test_score = (
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_test,
        best_result["binary_operator"],
    )
    logger.info(f"ROC AUC score on test set using " f"'{name}': {test_score}")


@task
def visualise_link_embeddings(examples_test, embedding_test, labels_test, best_result):
    link_features = link_examples_to_features(
        examples_test, embedding_test, best_result["binary_operator"]
    )
    plot_link_features_projection(
        n_components=2, link_features=link_features, labels_test=labels_test
    )


if __name__ == "__main__":
    with Flow("link-prediction") as flow:
        graph_train_test_labels = load_data_and_construct_splits()
        (
            graph_train,
            graph_test,
            examples_train,
            examples_test,
            examples_model_selection,
            labels_train,
            labels_test,
            labels_model_selection,
        ) = graph_train_test_labels
        embedding_train = node2vec_embedding(graph_train, "Train Graph", **params)
        embedding_test = node2vec_embedding(graph_test, "Test Graph", **params)
        results = train_link_classifier(examples_train, labels_train, embedding_train)
        best_result, name = compute_best_results(results)
        evaluate_on_test_data(examples_test, labels_test, embedding_test, best_result)
        visualise_link_embeddings(
            examples_test, embedding_test, labels_test, best_result
        )
    # flow.visualize()
    flow.run()
