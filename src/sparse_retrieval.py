from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = None

        if not chunks:
            return

        tokenized_documents = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def retrieve(self, query: str, top_k: int = 5):
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:

            results.append({
                "id": f"doc_0_chunk_{index}",
                "score": float(scores[index]),
                "text": self.chunks[index].page_content,
                "metadata": self.chunks[index].metadata,
            })

        return results
