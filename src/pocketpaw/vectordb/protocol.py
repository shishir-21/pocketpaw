"""
Vector store protocol.

Defines a generic interface for vector database implementations
like ChromaDB, Qdrant, or others.
"""

from typing import Protocol


class VectorStoreProtocol(Protocol):
    """
    Protocol defining required methods for vector databases.
    """

    async def add(self, text: str, id: str) -> None:
        """
        Store text in the vector database.
        """
        ...

    async def search(self, query: str, limit: int = 10) -> list[str]:
        """
        Return the most similar stored texts.
        """
        ...

    async def get_by_id(self, id: str) -> str | None:
        """
        Retrieve a stored entry by its exact ID.
        """
        ...

    async def delete(self, id: str) -> bool:
        """
        Delete a stored vector entry.
        """
        ...
