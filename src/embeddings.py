from sentence_transformers import (
    SentenceTransformer
)

from config import (
    EMBEDDING_MODEL
)


class EmbeddingGenerator:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def generate(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=True
        )