import prefect
from gensim.models import Word2Vec
from prefect import task
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stellargraph.data import BiasedRandomWalk

from models.metrics import evaluate_roc_auc
from preprocess.build_features import link_examples_to_features


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


def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }
