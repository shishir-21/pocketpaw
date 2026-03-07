"""
ChromaDB adapter implementation.

This adapter connects PocketPaw's vector database layer to ChromaDB
while following the VectorStoreProtocol interface.
"""

import chromadb

from pocketpaw.vectordb.protocol import VectorStoreProtocol


class ChromaAdapter(VectorStoreProtocol):
    """
    Adapter class for ChromaDB.
    """

    def __init__(self, data_path):
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(data_path))

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="pocketpaw_memory"
        )

    async def add(self, text: str, id: str) -> None:
        """
        Store text in the vector database.
        """
        self.collection.add(
            documents=[text],
            ids=[id],
        )

    async def search(self, query: str, limit: int = 5) -> list[str]:
        """
        Search for similar memories.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
        )

        return results.get("documents", [[]])[0]

    async def delete(self, id: str) -> bool:
        """
        Delete a stored vector entry.
        """
        try:
            self.collection.delete(ids=[id])
            return True
        except Exception:
            return False
