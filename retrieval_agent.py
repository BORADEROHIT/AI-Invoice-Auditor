import hashlib
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

class SimpleHashEmbeddings(Embeddings):
    def __init__(self, size: int = 384):
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.size

        for word in text.lower().split():
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.size
            vector[index] += 1.0

        total = sum(vector) or 1.0
        return [value / total for value in vector]

class RetrievalAgent:
    """
    RAG Retrieval Agent
    """

    def __init__(self, index_dir: str = "./outputs/faiss_index"):
        self.index_dir = Path(index_dir)
        self.embeddings = SimpleHashEmbeddings()

    def retrieve(self, query: str, top_k: int = 3):
        if not self.index_dir.exists():
            return []

        vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store.similarity_search(query, k=top_k)
