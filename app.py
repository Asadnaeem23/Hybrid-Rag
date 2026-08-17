from pathlib import Path
from uuid import uuid4

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_GROQ_MODEL,
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    DEFAULT_SPARSE_TOP_K,
    DEFAULT_TEMPERATURE,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RuntimeConfig,
    VOYAGE_API_KEY,
)
from src.indexing import SUPPORTED_EXTENSIONS, index_document_paths
from src.runtime import build_rag_pipeline


UPLOAD_DIR = Path("data") / "uploads"


def init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "sparse_chunks" not in st.session_state:
        st.session_state.sparse_chunks = []
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "upload_batch_id" not in st.session_state:
        st.session_state.upload_batch_id = str(uuid4())


def reset_chat() -> None:
    thread_id = st.session_state.get("thread_id")
    checkpointer = st.session_state.get("checkpointer")

    if thread_id and checkpointer:
        try:
            checkpointer.delete_thread(thread_id)
        except Exception:
            pass

    st.session_state.thread_id = str(uuid4())
    st.session_state.chat_messages = []


def clean_error(error: Exception, runtime_config: RuntimeConfig) -> str:
    message = str(error) or error.__class__.__name__

    for secret in [
        runtime_config.voyage_api_key,
        runtime_config.pinecone_api_key,
        runtime_config.groq_api_key,
        runtime_config.openrouter_api_key,
    ]:
        if secret:
            message = message.replace(secret, "[redacted]")

    return message


def build_runtime_config() -> RuntimeConfig:
    provider = st.session_state.get("generation_provider", "OpenRouter")
    default_model = (
        DEFAULT_GROQ_MODEL
        if provider == "Groq"
        else DEFAULT_OPENROUTER_MODEL
    )

    return RuntimeConfig(
        voyage_api_key=st.session_state.get("voyage_api_key") or VOYAGE_API_KEY,
        pinecone_api_key=st.session_state.get("pinecone_api_key") or PINECONE_API_KEY,
        pinecone_index=st.session_state.get("pinecone_index") or PINECONE_INDEX_NAME,
        groq_api_key=st.session_state.get("groq_api_key") or GROQ_API_KEY,
        openrouter_api_key=st.session_state.get("openrouter_api_key") or OPENROUTER_API_KEY,
        generation_provider=provider,
        generation_model=st.session_state.get("generation_model") or default_model,
        generation_temperature=st.session_state.get("generation_temperature", DEFAULT_TEMPERATURE),
        dense_top_k=st.session_state.get("dense_top_k", DEFAULT_DENSE_TOP_K),
        sparse_top_k=st.session_state.get("sparse_top_k", DEFAULT_SPARSE_TOP_K),
        rerank_top_k=st.session_state.get("rerank_top_k", DEFAULT_RERANK_TOP_K),
        retrieval_min_score=st.session_state.get(
            "retrieval_min_score",
            DEFAULT_RETRIEVAL_MIN_SCORE,
        ),
        enable_query_rewrite=st.session_state.get("enable_query_rewrite", True),
        chunk_size=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
        chunk_overlap=st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        index_batch_size=st.session_state.get("index_batch_size", DEFAULT_INDEX_BATCH_SIZE),
    )


def render_sidebar() -> RuntimeConfig:
    with st.sidebar:
        st.header("API Configuration")
        st.text_input(
            "Voyage AI API Key",
            type="password",
            placeholder="Paste your Voyage AI key for embeddings and reranking.",
            help="Required for document embeddings, query embeddings, and reranking retrieved sources.",
            key="voyage_api_key",
        )
        st.text_input(
            "Pinecone API Key",
            type="password",
            placeholder="Paste your Pinecone API key for vector storage.",
            help="Required to index uploaded documents and search the dense vector index.",
            key="pinecone_api_key",
        )
        st.text_input(
            "Pinecone Index Name",
            value=st.session_state.get("pinecone_index", PINECONE_INDEX_NAME),
            placeholder="Enter the Pinecone index name, for example hybrid-rag-dense.",
            help="Use an existing Pinecone index that matches the embedding dimension used by this project.",
            key="pinecone_index",
        )
        st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq key for Groq generation or query rewriting.",
            help="Required when Provider is Groq. Also used for query rewriting when that option is enabled.",
            key="groq_api_key",
        )
        st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="Paste your OpenRouter key for OpenRouter generation.",
            help="Required when Provider is OpenRouter.",
            key="openrouter_api_key",
        )

        st.header("LLM Configuration")
        provider = st.selectbox(
            "Provider",
            ["OpenRouter", "Groq"],
            help="Choose which service generates the final answer.",
            key="generation_provider",
        )
        model_placeholder = (
            f"Enter a Groq model, for example {DEFAULT_GROQ_MODEL}."
            if provider == "Groq"
            else f"Enter an OpenRouter model, for example {DEFAULT_OPENROUTER_MODEL}."
        )
        st.text_input(
            "Model",
            placeholder=model_placeholder,
            help="Leave empty to use the default model for the selected provider.",
            key="generation_model",
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("generation_temperature", DEFAULT_TEMPERATURE),
            step=0.1,
            help="Lower values are more focused. Higher values are more varied but may be less predictable.",
            key="generation_temperature",
        )

        with st.expander("Advanced RAG Settings"):
            dense_top_k = st.number_input(
                "Dense Results",
                min_value=1,
                max_value=50,
                value=st.session_state.get("dense_top_k", DEFAULT_DENSE_TOP_K),
                step=1,
                help="How many vector-search matches to fetch from Pinecone before fusion.",
                key="dense_top_k",
            )
            sparse_top_k = st.number_input(
                "Sparse Results",
                min_value=1,
                max_value=50,
                value=st.session_state.get("sparse_top_k", DEFAULT_SPARSE_TOP_K),
                step=1,
                help="How many keyword/BM25 matches to fetch from uploaded in-memory chunks.",
                key="sparse_top_k",
            )
            st.number_input(
                "Final Sources",
                min_value=1,
                max_value=max(1, dense_top_k + sparse_top_k),
                value=min(
                    st.session_state.get("rerank_top_k", DEFAULT_RERANK_TOP_K),
                    dense_top_k + sparse_top_k,
                ),
                step=1,
                help="How many fused results to keep after Voyage reranking and send to the answer model.",
                key="rerank_top_k",
            )
            st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get(
                    "retrieval_min_score",
                    DEFAULT_RETRIEVAL_MIN_SCORE,
                ),
                step=0.05,
                help="Minimum rerank score treated as good retrieval before optional query rewriting.",
                key="retrieval_min_score",
            )
            st.toggle(
                "Enable Query Rewrite",
                value=st.session_state.get("enable_query_rewrite", True),
                help="When retrieval quality is low, let a Groq rewriter try one improved search query.",
                key="enable_query_rewrite",
            )

        st.header("Upload Documents")
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=4000,
            value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            step=50,
            help="Approximate character length for each document chunk created during upload processing.",
            key="chunk_size",
        )
        st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=max(0, chunk_size - 1),
            value=min(
                st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
                chunk_size - 1,
            ),
            step=10,
            help="Characters repeated between neighboring chunks. Keep this smaller than chunk size.",
            key="chunk_overlap",
        )
        st.number_input(
            "Index Batch Size",
            min_value=1,
            max_value=500,
            value=st.session_state.get("index_batch_size", DEFAULT_INDEX_BATCH_SIZE),
            step=10,
            help="Number of vector records sent to Pinecone per upload batch.",
            key="index_batch_size",
        )
        uploaded_files = st.file_uploader(
            "Files",
            type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, TXT, or Markdown files to add them to the RAG knowledge base.",
        )

        runtime_config = build_runtime_config()

        if st.button("Process Uploads", type="primary", use_container_width=True):
            process_uploads(uploaded_files, runtime_config)

        st.divider()
        if st.button("New Chat", use_container_width=True):
            reset_chat()
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            reset_chat()
            st.rerun()

    return runtime_config


def process_uploads(uploaded_files, runtime_config: RuntimeConfig) -> None:
    if not uploaded_files:
        st.info("Choose one or more supported files first.")
        return

    try:
        runtime_config.require_indexing_settings()
    except ValueError as error:
        st.error(clean_error(error, runtime_config))
        return

    batch_dir = UPLOAD_DIR / st.session_state.upload_batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        extension = Path(safe_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            st.warning(f"Unsupported file skipped: {safe_name}")
            continue

        target = batch_dir / f"{uuid4().hex}_{safe_name}"
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)

    if not saved_paths:
        st.warning("No supported files were selected.")
        return

    with st.status("Processing document...", expanded=True) as status:
        try:
            result = index_document_paths(
                saved_paths,
                runtime_config=runtime_config,
                batch_size=runtime_config.index_batch_size,
            )
        except Exception as error:
            status.update(label="Upload failed", state="error")
            st.error(clean_error(error, runtime_config))
            return

        st.session_state.sparse_chunks.extend(result.chunks)
        status.update(
            label=(
                f"Successfully indexed {result.documents_indexed} documents / "
                f"{result.chunks_indexed} chunks."
            ),
            state="complete",
        )

    for name in result.unsupported_files:
        st.warning(f"Unsupported file skipped: {name}")
    for name in result.empty_files:
        st.warning(f"Empty document skipped: {name}")

    st.session_state.upload_batch_id = str(uuid4())


def display_sources(sources: list[dict]) -> None:
    with st.expander("Retrieved Sources"):
        if not sources:
            st.caption("No sources retrieved.")
            return

        for source in sources:
            metadata = source.get("metadata") or {}
            raw_source = str(metadata.get("source", "Unknown source"))
            source_name = Path(raw_source).name
            chunk_id = metadata.get("chunk_id") or source.get("id", "Unknown chunk")
            rerank_score = source.get("rerank_score")

            if rerank_score is None:
                score_text = "n/a"
            else:
                score_text = f"{float(rerank_score):.4f}"

            st.markdown(
                f"- **{source_name}** | `{chunk_id}` | rerank score: `{score_text}`"
            )


def run_chat(runtime_config: RuntimeConfig) -> None:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                display_sources(message.get("sources", []))

    user_query = st.chat_input("Ask a question about your uploaded documents.")

    if not user_query:
        return

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            runtime_config.require_chat_settings()
            pipeline = build_rag_pipeline(
                runtime_config=runtime_config,
                sparse_chunks=st.session_state.sparse_chunks,
                checkpointer=st.session_state.checkpointer,
            )

            result = pipeline.graph.invoke(
                {
                    "query": user_query,
                    "rewritten_query": "",
                    "retrieved_documents": [],
                    "rewrite_attempted": False,
                    "retrieval_quality": "",
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                    }
                },
            )
        except Exception as error:
            answer = clean_error(error, runtime_config)
            sources = []
            st.error(answer)
        else:
            answer = result.get("answer") or "No answer was generated."
            sources = result.get("retrieved_documents", [])
            st.markdown(answer)
            display_sources(sources)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="Hybrid RAG Assistant",
        layout="wide",
    )
    init_session_state()

    st.title("Hybrid RAG Assistant")
    runtime_config = render_sidebar()
    run_chat(runtime_config)


if __name__ == "__main__":
    main()
