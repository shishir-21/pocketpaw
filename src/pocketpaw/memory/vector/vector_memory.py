"""
Vector memory manager.

This module selects the correct vector database adapter
based on environment configuration.
"""

import os
from .chroma_adapter import ChromaAdapter


class VectorMemory:
    """
    High-level memory interface used by the agent.
    """

    def __init__(self, user_id: str = "default", data_path=None):
        # Save configuration
        self.user_id = user_id
        self.data_path = data_path
        
        # Read vector store configuration
        vector_db = os.getenv("POCKETPAW_VECTOR_STORE", "chromadb")

        # Select adapter based on configuration
        if vector_db == "chromadb":
            self.adapter = ChromaAdapter()
        else:
            raise ValueError(f"Unsupported vector store: {vector_db}")

    def add_memory(self, text: str):
        """
        Add memory to vector database using the selected adapter.
        """
        self.adapter.add(text)

    def search_memory(self, query: str, k: int = 5):
        """
        Search relevant memories using semantic similarity.
        """
        return self.adapter.search(query, k)