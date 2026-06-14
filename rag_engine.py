import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self):
        # Load the model locally — downloads once (~80 MB) and caches automatically.
        print("Loading sentence-transformers model locally...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")

        # Your corporate knowledge base
        self.documents = [
            {
                "title": "Refunds & Returns Policy",
                "content": (
                    "Policy: Customers can return items within 30 days of purchase for a full refund. "
                    "Items must be undamaged and in original packaging. "
                    "Damaged items upon arrival qualify for an instant free replacement."
                ),
            },
            {
                "title": "Shipping & Delivery Guarantees",
                "content": (
                    "Policy: Standard shipping takes 3-5 business days. "
                    "Expedited shipping takes 1-2 business days. "
                    "If an order is delayed past 7 business days, the customer is automatically issued a $15 credit."
                ),
            },
            {
                "title": "Subscription & Cancellation Terms",
                "content": (
                    "Policy: Subscriptions can be canceled at any time via the user dashboard. "
                    "Cancellations made mid-billing cycle will keep access active until the end of the current period. "
                    "No partial refunds given."
                ),
            },
        ]

        print("Initialising RAG Knowledge Base — indexing policy documents...")
        # Pre-compute embeddings for all documents once at startup.
        # encode() returns a numpy array directly — no parsing needed.
        self._doc_embeddings = self.model.encode(
            [doc["content"] for doc in self.documents],
            show_progress_bar=False,
        )
        print("RAG Knowledge Base successfully indexed!")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text string using the local model."""
        return self.model.encode(text, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        query_embedding = self._get_embedding(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self._doc_embeddings)[0]

        results = []
        for idx, doc in enumerate(self.documents):
            doc_copy = doc.copy()
            doc_copy["similarity"] = float(similarities[idx])
            results.append(doc_copy)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_context_string(self, query: str, top_k: int = 2) -> str:
        matched_docs = self.retrieve(query, top_k=top_k)
        context_blocks = [
            f"[{doc['title']}]\n{doc['content']}\n"
            for doc in matched_docs
        ]
        return "\n".join(context_blocks)