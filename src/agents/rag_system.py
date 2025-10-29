"""
RAG (Retrieval-Augmented Generation) System

Provides context retrieval for agents using:
- FAISS vector store
- Sentence transformers for embeddings
- Document chunking and indexing
- Semantic search
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """
    RAG system for financial document retrieval

    Features:
    - Document chunking with overlap
    - Semantic embeddings via sentence transformers
    - FAISS vector indexing for fast search
    - Metadata filtering (symbol, document type, date)
    - Top-k retrieval with similarity scores
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        index_path: Optional[Path] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize RAG system

        Args:
            embedding_model: Sentence transformer model
            index_path: Path to save/load FAISS index
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []  # Store document metadata

        # Try to load existing index
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        logger.info(f"RAG system initialized with {len(self.documents)} documents")

    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add documents to the RAG system

        Args:
            texts: List of document texts
            metadata: List of metadata dicts (one per document)
                     Should include: symbol, doc_type, date, title
        """
        logger.info(f"Adding {len(texts)} documents to RAG system")

        if metadata is None:
            metadata = [{"symbol": "UNKNOWN", "doc_type": "unknown"} for _ in texts]

        # Chunk documents
        chunks = []
        chunk_metadata = []

        for text, meta in zip(texts, metadata):
            doc_chunks = self._chunk_text(text)

            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)

                # Add chunk metadata
                chunk_meta = meta.copy()
                chunk_meta["chunk_id"] = i
                chunk_meta["total_chunks"] = len(doc_chunks)
                chunk_metadata.append(chunk_meta)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store document metadata
        for chunk, meta in zip(chunks, chunk_metadata):
            self.documents.append({"text": chunk, "metadata": meta})

        logger.info(f"Added {len(chunks)} chunks. Total documents: {len(self.documents)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        symbol: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            symbol: Filter by stock symbol
            doc_type: Filter by document type (e.g., "10-K", "news")

        Returns:
            List of retrieved documents with text, metadata, and similarity score
        """
        if len(self.documents) == 0:
            logger.warning("No documents in RAG system")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
        )[0]

        # Search FAISS index (retrieve more than top_k for filtering)
        search_k = min(top_k * 5, len(self.documents))
        distances, indices = self.index.search(
            query_embedding.astype(np.float32).reshape(1, -1),
            search_k,
        )

        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]

                # Apply filters
                if symbol and doc["metadata"].get("symbol") != symbol:
                    continue
                if doc_type and doc["metadata"].get("doc_type") != doc_type:
                    continue

                results.append(
                    {
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "similarity_score": float(1 / (1 + dist)),  # Convert distance to similarity
                    }
                )

                if len(results) >= top_k:
                    break

        logger.info(f"Retrieved {len(results)} documents for query")
        return results

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping segments

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk:
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

        return chunks

    def save_index(self, path: Path):
        """Save FAISS index and documents to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save documents metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Saved RAG index to {path}")

    def load_index(self, path: Path):
        """Load FAISS index and documents from disk"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")

        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Load documents metadata
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        logger.info(f"Loaded RAG index from {path} ({len(self.documents)} documents)")

    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        doc_types = {}
        symbols = {}

        for doc in self.documents:
            meta = doc["metadata"]

            doc_type = meta.get("doc_type", "unknown")
            symbol = meta.get("symbol", "UNKNOWN")

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            symbols[symbol] = symbols.get(symbol, 0) + 1

        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "doc_types": doc_types,
            "symbols": symbols,
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(chunk_size=512, chunk_overlap=50)

    # Sample financial documents
    documents = [
        "Apple Inc. reported strong Q4 earnings with revenue up 12% YoY. iPhone sales exceeded expectations.",
        "The company's Services revenue reached an all-time high of $20 billion, growing at 15% annually.",
        "Apple's balance sheet remains robust with $170 billion in cash and investments.",
        "Management expressed confidence in the upcoming product cycle with new AI features.",
        "Analysts are concerned about slowing iPhone growth in China due to competition.",
    ]

    metadata = [
        {"symbol": "AAPL", "doc_type": "earnings", "date": "2024-11-01", "title": "Q4 Earnings"},
        {"symbol": "AAPL", "doc_type": "earnings", "date": "2024-11-01", "title": "Q4 Earnings"},
        {"symbol": "AAPL", "doc_type": "10-K", "date": "2024-10-30", "title": "Annual Report"},
        {"symbol": "AAPL", "doc_type": "earnings", "date": "2024-11-01", "title": "Q4 Call"},
        {"symbol": "AAPL", "doc_type": "news", "date": "2024-11-02", "title": "Analyst Note"},
    ]

    # Add documents
    rag.add_documents(documents, metadata)

    # Retrieve relevant context
    query = "What are Apple's revenue growth and earnings trends?"
    results = rag.retrieve(query, top_k=3, symbol="AAPL")

    print(f"\nQuery: {query}\n")
    print(f"Retrieved {len(results)} documents:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']}")
        print(f"   Type: {result['metadata']['doc_type']}, "
              f"Similarity: {result['similarity_score']:.3f}\n")

    # Get stats
    stats = rag.get_stats()
    print(f"\nRAG Stats: {stats}")
