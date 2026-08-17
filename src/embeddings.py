from typing import List, Optional
import voyageai
from langchain_core.embeddings import Embeddings

from src.config import VOYAGE_API_KEY


class VoyageEmbedder(Embeddings):
    def __init__(self, model: str = "voyage-4", api_key: Optional[str] = None):
        self.model = model
        key = api_key or VOYAGE_API_KEY
        self.client = voyageai.Client(api_key=key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document",
        )
        return response.embeddings

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
        )
        return response.embeddings[0]
