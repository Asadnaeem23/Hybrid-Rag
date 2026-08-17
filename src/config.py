from dotenv import load_dotenv
from dataclasses import dataclass
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-rag-dense")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

GROQ_API_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

OPENROUTER_API_KEYS = [k for k in [
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2"),
    os.getenv("OPENROUTER_API_KEY_3"),
] if k]

GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else None
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else None


DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DENSE_TOP_K = 10
DEFAULT_SPARSE_TOP_K = 10
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RETRIEVAL_MIN_SCORE = 0.50
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_INDEX_BATCH_SIZE = 100


@dataclass(frozen=True)
class RuntimeConfig:
    voyage_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index: str = PINECONE_INDEX_NAME
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    generation_provider: str = "OpenRouter"
    generation_model: str = DEFAULT_OPENROUTER_MODEL
    generation_temperature: float = DEFAULT_TEMPERATURE
    dense_top_k: int = DEFAULT_DENSE_TOP_K
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    retrieval_min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE
    enable_query_rewrite: bool = True
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    index_batch_size: int = DEFAULT_INDEX_BATCH_SIZE

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            voyage_api_key=VOYAGE_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index=PINECONE_INDEX_NAME,
            groq_api_key=GROQ_API_KEY,
            openrouter_api_key=OPENROUTER_API_KEY,
        )

    @property
    def provider_key(self) -> str:
        return self.generation_provider.strip().lower()

    @property
    def generation_api_key(self) -> str | None:
        if self.provider_key == "groq":
            return self.groq_api_key
        if self.provider_key == "openrouter":
            return self.openrouter_api_key
        return None

    def require_chat_settings(self) -> None:
        if not self.voyage_api_key:
            raise ValueError("Voyage API key is required for retrieval and reranking.")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required for retrieval.")
        if not self.pinecone_index:
            raise ValueError("Pinecone index name is required.")
        if self.provider_key not in {"groq", "openrouter"}:
            raise ValueError("Generation provider must be Groq or OpenRouter.")
        if not self.generation_model:
            raise ValueError("Generation model is required.")
        if not 0 <= self.generation_temperature <= 2:
            raise ValueError("Generation temperature must be between 0 and 2.")
        if self.dense_top_k < 1 or self.sparse_top_k < 1 or self.rerank_top_k < 1:
            raise ValueError("Retrieval result counts must be at least 1.")
        if self.rerank_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("Final reranked sources cannot exceed dense plus sparse retrieval results.")
        if not 0 <= self.retrieval_min_score <= 1:
            raise ValueError("Retrieval confidence threshold must be between 0 and 1.")
        if not self.generation_api_key:
            raise ValueError(f"{self.generation_provider} API key is required for answer generation.")

    def require_indexing_settings(self) -> None:
        if not self.voyage_api_key:
            raise ValueError("Voyage API key is required for document embedding.")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required for indexing documents.")
        if not self.pinecone_index:
            raise ValueError("Pinecone index name is required.")
        if self.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 characters.")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        if self.index_batch_size < 1:
            raise ValueError("Index batch size must be at least 1.")
