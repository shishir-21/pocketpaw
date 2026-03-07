"""
ChromaDB adapter implementation.

This adapter connects PocketPaw's vector database layer to ChromaDB
while following the VectorStoreProtocol interface.
"""

import asyncio


class ChromaAdapter:
    """
    Adapter class for ChromaDB.
    """

    def __init__(self, data_path):

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install pocketpaw[vector]"
            )

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=str(data_path))

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="pocketpaw_memory"
        )

    async def add(self, text: str, id: str) -> None:
        """
        Store text in the vector database.
        """

        await asyncio.to_thread(
            self.collection.add,
            documents=[text],
            ids=[id],
        )

    async def search(self, query: str, limit: int = 10) -> list[str]:
        """
        Search for similar memories.
        """

        results = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query],
            n_results=limit,
        )

        return results.get("documents", [[]])[0]

    async def get_by_id(self, id: str):

        results = await asyncio.to_thread(
            self.collection.get,
            ids=[id],
        )

        docs = results.get("documents", [])

        if not docs:
            return None

        return docs[0]

    async def delete(self, id: str) -> bool:
        """
        Delete a stored vector entry.
        """

        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=[id],
            )
            return True
        except Exception:
            return False
