import chromadb
from sentence_transformers import SentenceTransformer


class ComplaintRetriever:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="vector_store/chroma_db"
        )

        self.collection = (
            self.client.get_collection(
                "complaints"
            )
        )

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def retrieve(
        self,
        query,
        k=5
    ):

        query_embedding = (
            self.model.encode(query)
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k
        )

        return results