import multiprocessing

import pandas as pd
import prefect
from prefect import Flow, task

from models.embeddings import node2vec_embedding
from models.evaluate import (
    evaluate_link_prediction_model,
    operator_avg,
    operator_hadamard,
    operator_l1,
    operator_l2,
)
from models.train import train_link_prediction_model
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


@task
def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


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


if __name__ == "__main__":
    with Flow("link-prediction") as flow:
        graph = cora_dataset()
        (
            graph_train,
            graph_test,
            examples_train,
            examples_test,
            examples_model_selection,
            labels_train,
            labels_test,
            labels_model_selection,
        ) = edge_splitter_graph_train_test(graph)
        embedding_train = node2vec_embedding(graph_train, "Train Graph", **params)
        binary_operators = operator_hadamard, operator_l1, operator_l2, operator_avg
        results = [run_link_prediction(op) for op in binary_operators]
        best_result, name = compute_best_results(results)
        embedding_test = node2vec_embedding(graph_test, "Test Graph", **params)
        test_score = evaluate_link_prediction_model(
            best_result["classifier"],
            examples_test,
            labels_test,
            embedding_test,
            best_result["binary_operator"],
        )
        logger.info(f"ROC AUC score on test set using " f"'{name}': {test_score}")

        link_features = link_examples_to_features(
            examples_test, embedding_test, best_result["binary_operator"]
        )
        plot_link_features_projection(
            n_components=2, link_features=link_features, labels_test=labels_test
        )
    flow.visualize()
    # flow.run()
