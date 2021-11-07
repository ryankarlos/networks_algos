import multiprocessing

import pandas as pd
from node2vec import node2vec_embedding
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.make_dataset import cora_dataset
from src.models.metrics import (
    evaluate_link_prediction_model,
    operator_avg,
    operator_hadamard,
    operator_l1,
    operator_l2,
)
from src.processing.build_features import (
    edge_splitter_graph_train_test,
    link_examples_to_features,
)
from src.utils.log import get_logger
from src.visualization.visualize import plot_link_features_projection

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

LOG = get_logger(__name__)


def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


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


if __name__ == "__main__":
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
    LOG.info("Run node2vec walk on train graph")
    embedding_train = node2vec_embedding(graph_train, "Train Graph", **params)
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    LOG.info("Train link prediction model")
    results = [run_link_prediction(op) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])

    LOG.info(f"Best result from '{best_result['binary_operator'].__name__}'")

    print(
        pd.DataFrame(
            [
                (result["binary_operator"].__name__, result["score"])
                for result in results
            ],
            columns=("name", "ROC AUC score"),
        ).set_index("name")
    )
    LOG.info("Run node2vec walk on test graph")
    embedding_test = node2vec_embedding(graph_test, "Test Graph", **params)
    LOG.info("Evaluate link prediction model on test set")
    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_test,
        best_result["binary_operator"],
    )
    LOG.info(
        f"ROC AUC score on test set using "
        f"'{best_result['binary_operator'].__name__}': {test_score}"
    )

    link_features = link_examples_to_features(
        examples_test, embedding_test, best_result["binary_operator"]
    )
    plot_link_features_projection(
        n_components=2, link_features=link_features, labels_test=labels_test
    )
