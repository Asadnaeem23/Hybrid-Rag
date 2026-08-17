from pinecone import Pinecone

from src.config import PINECONE_API_KEY


class PineconeStore:
    def __init__(self, index_name: str, api_key: str | None = None):
        self.pc = Pinecone(api_key=api_key or PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def search(self, vector, top_k=5):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
