from typing import Optional

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


REWRITE_MODEL = "llama-3.3-70b-versatile"


def get_query_rewriter(api_key: str | None = None):

    return ChatGroq(
        model=REWRITE_MODEL,
        api_key=api_key or GROQ_API_KEY,
        temperature=0,
    )


def rewrite_query(
    query: str,
    llm,
    knowledge_context: Optional[str] = None,
) -> str:

    knowledge_context = knowledge_context or """
The knowledge base contains technical documentation.
"""

    prompt = f"""
You are a query rewriting component inside a RAG system.

Your job is to rewrite a user's query ONLY when necessary
so that it becomes more useful for retrieving relevant
documents.

Knowledge base context:
{knowledge_context}

Rules:

1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not invent facts.
4. Resolve vague references when the available knowledge-base
   context makes the intended meaning reasonably clear.
5. Add useful technical terminology when it is strongly implied.
6. Do not unnecessarily expand an already good query.
7. Return ONLY the rewritten search query.
8. Do not add explanations, labels, quotes, or commentary.

USER QUERY:
{query}
"""

    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    # Safety fallback
    if not rewritten:
        return query

    return rewritten
