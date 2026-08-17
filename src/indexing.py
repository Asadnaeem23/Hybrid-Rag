from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from langchain_core.documents import Document

from src.chunking import create_chunks
from src.config import RuntimeConfig
from src.embeddings import VoyageEmbedder
from src.ingestion import load_document
from src.vector_store import PineconeStore


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class IndexingResult:
    documents_indexed: int = 0
    chunks_indexed: int = 0
    chunks: list[Document] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)


def _batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def index_document_paths(
    file_paths: list[str | Path],
    runtime_config: RuntimeConfig,
    batch_size: int | None = None,
) -> IndexingResult:
    runtime_config.require_indexing_settings()
    batch_size = batch_size or runtime_config.index_batch_size

    result = IndexingResult()
    embedder = VoyageEmbedder(api_key=runtime_config.voyage_api_key)
    vector_store = PineconeStore(
        runtime_config.pinecone_index,
        api_key=runtime_config.pinecone_api_key,
    )

    for file_path in file_paths:
        path = Path(file_path)

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.unsupported_files.append(path.name)
            continue

        documents = [
            document
            for document in load_document(str(path))
            if document.page_content and document.page_content.strip()
        ]

        if not documents:
            result.empty_files.append(path.name)
            continue

        chunks = [
            chunk
            for chunk in create_chunks(
                documents,
                chunk_size=runtime_config.chunk_size,
                chunk_overlap=runtime_config.chunk_overlap,
            )
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not chunks:
            result.empty_files.append(path.name)
            continue

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.embed_documents(texts)
        upload_id = uuid4().hex

        records = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{path.stem}_{upload_id}_chunk_{index}"
            metadata = {
                **chunk.metadata,
                "source": path.name,
                "text": chunk.page_content,
                "chunk_id": chunk_id,
            }
            chunk.metadata.update(metadata)
            records.append(
                {
                    "id": chunk_id,
                    "values": vector,
                    "metadata": metadata,
                }
            )

        for batch in _batched(records, batch_size):
            vector_store.upsert(batch)

        result.documents_indexed += 1
        result.chunks_indexed += len(records)
        result.chunks.extend(chunks)

    return result
