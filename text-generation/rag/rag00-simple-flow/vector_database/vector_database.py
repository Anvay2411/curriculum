from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from typing import List
import os


class VectorDB:
    def __init__(
        self,
        path: str = "data",
        collection_name: str = "RAG",
        embedding_size: int = 1024,
        similarity_method=Distance.COSINE,
        verbose: bool = True,
    ):
        self.collection_name = collection_name
        self.verboseprint = print if verbose else lambda *a: None
        self.client = QdrantClient(path=path)

        if os.path.exists(os.path.join(path, "collection", collection_name)):
            self.verboseprint("Database Loaded")
        else:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_size, distance=similarity_method
                ),
            )
            self.verboseprint(f"Vector Database {self.collection_name} initialized.")

    def insert_docs_embeddings(self, embeddings: List, metadata=None):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector=embeds.tolist(),
                    payload={"text": text},
                )
                for idx, (embeds, text) in enumerate(zip(embeddings, metadata))
            ],
        )
        self.verboseprint("Embedding stored.")

    def query_embeddings(self, query_embedding, top_k=5):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        )

        # 🔒 Version-safe handling
        if isinstance(result, list):
            # Older qdrant-client
            points = result
        else:
            # Newer qdrant-client (QueryResponse object)
            points = result.points

        return points
