"""
Base adapter interface for all vector database implementations.

This ensures PocketPaw can support multiple vector databases
(ChromaDB, Qdrant, SQLite-vec, etc.) using the same interface.
"""

from abc import ABC, abstractmethod


class VectorDBAdapter(ABC):
    """
    Abstract base class defining the required methods
    for any vector database adapter.
    """

    @abstractmethod
    def add(self, text: str):
        """
        Store a piece of memory text in the vector database.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5):
        """
        Search for the top-k semantically similar memories.
        """
        pass