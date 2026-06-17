import json
import hashlib
from pathlib import Path

from langchain_core.documents import Document
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
    
class IndexingAgent:
    """
    RAG Indexing Agent
    """

    def __init__(
        self,
        reports_dir: str = "./outputs/reports",
        index_dir: str = "./outputs/faiss_index"
    ):
        self.reports_dir = Path(reports_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = SimpleHashEmbeddings()

    def index_reports(self) -> dict:
        documents = []

        for report_file in self.reports_dir.glob("*_report.json"):
            with open(report_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            content = json.dumps(data, indent=2)
            chunks = self._chunk_text(content)

            for index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(report_file),
                            "chunk_id": index
                        }
                    )
                )

        if not documents:
            return {
                "status": "no_reports_found",
                "message": "No report files found for indexing."
            }

        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(str(self.index_dir))

        return {
            "status": "indexed",
            "documents_indexed": len(documents),
            "index_dir": str(self.index_dir)
        }

    def _chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks
