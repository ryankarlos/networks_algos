import prefect
from gensim.models import Word2Vec
from prefect import task
from stellargraph.data import BiasedRandomWalk


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
