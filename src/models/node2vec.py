from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk

from src.utils.log import get_logger

LOG = get_logger(__name__)


def node2vec_embedding(graph, name, **kwargs):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(
        graph.nodes(),
        n=kwargs["num_walks"],
        length=kwargs["walk_length"],
        p=kwargs["p"],
        q=kwargs["q"],
    )
    LOG.info(f"Number of random walks for '{name}': {len(walks)}")

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
