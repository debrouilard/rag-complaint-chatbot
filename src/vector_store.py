import chromadb
import pandas as pd

from chunking import chunk_text
from embeddings import EmbeddingGenerator
from config import VECTOR_DB_PATH


class ComplaintVectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH
        )

        self.collection = (
            self.client.get_or_create_collection(
                "complaints"
            )
        )

        self.embedding_model = (
            EmbeddingGenerator()
        )

    def add_complaints(self, dataframe):

        total_chunks = 0

        for idx, row in dataframe.iterrows():

            complaint_id = str(idx)

            product = row["Product"]

            chunks = chunk_text(
                row["clean_narrative"]
            )

            embeddings = (
                self.embedding_model.generate(
                    chunks
                )
            )

            ids = []
            metadata = []

            for i, chunk in enumerate(chunks):

                ids.append(
                    f"{complaint_id}_{i}"
                )

                metadata.append(
                    {
                        "complaint_id":
                        complaint_id,

                        "product_category":
                        product,

                        "chunk_index":
                        i
                    }
                )

            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadata
            )

            total_chunks += len(chunks)

        print(
            f"Stored {total_chunks} chunks."
        )


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/sampled_complaints.csv"
    )

    store = ComplaintVectorStore()

    store.add_complaints(df)