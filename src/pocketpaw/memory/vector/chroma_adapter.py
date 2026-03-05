"""
ChromaDB adapter implementation.

This adapter connects PocketPaw's memory system to ChromaDB
while following the VectorDBAdapter interface.
"""

import chromadb
from .base import VectorDBAdapter


class ChromaAdapter(VectorDBAdapter):
    """
    Adapter class for ChromaDB.
    """

    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.Client()

        # Create or get a collection to store memory vectors
        self.collection = self.client.get_or_create_collection(
            name="pocketpaw_memory"
        )

    def add(self, text: str):
        """
        Add a memory text to the vector database.
        """
        self.collection.add(
            documents=[text],
            ids=[str(hash(text))]  # simple unique ID
        )

    def search(self, query: str, k: int = 5):
        """
        Retrieve top-k similar memories based on semantic similarity.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        return results.get("documents", [])