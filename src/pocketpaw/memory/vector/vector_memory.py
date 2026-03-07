"""
Vector memory backend for PocketPaw.

Implements MemoryStoreProtocol so it integrates with MemoryManager.
"""

import hashlib
from pathlib import Path

from pocketpaw.memory.protocol import MemoryEntry, MemoryStoreProtocol, MemoryType
from pocketpaw.vectordb.chroma_adapter import ChromaAdapter


class VectorMemory(MemoryStoreProtocol):
    """
    Vector-based memory backend using a vector database adapter.
    """

    def __init__(self, user_id: str = "default", data_path: Path | None = None):
        self.user_id = user_id
        self.data_path = data_path

        # simple session memory store
        self.sessions: dict[str, list[MemoryEntry]] = {}

        # initialize vector adapter
        db_path = data_path or (Path.home() / ".pocketpaw" / "vector_memory")
        self.adapter = ChromaAdapter(db_path)

    def _generate_id(self, text: str) -> str:
        """
        Generate deterministic ID for stored memory.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    async def save(self, entry: MemoryEntry) -> str:
        """
        Save memory entry to vector database.
        """
        entry_id = self._generate_id(entry.content)

        # store in vector db
        await self.adapter.add(entry.content, entry_id)

        # optional session storage
        if entry.session_key:
            self.sessions.setdefault(entry.session_key, []).append(entry)

        return entry_id

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """
        Retrieve memory by ID (approximate via vector search).
        """
        results = await self.adapter.search(entry_id, limit=1)

        if not results:
            return None

        return MemoryEntry(
            id=entry_id,
            content=results[0],
            type=MemoryType.LONG_TERM,
        )

    async def delete(self, entry_id: str) -> bool:
        """
        Delete memory entry from vector database.
        """
        try:
            return await self.adapter.delete(entry_id)
        except Exception:
            return False

    async def search(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        tags=None,
        limit: int = 5,
    ):
        """
        Semantic search using vector similarity.
        """
        docs = await self.adapter.search(query or "", limit=limit)

        return [
            MemoryEntry(
                id=self._generate_id(text),
                content=text,
                type=MemoryType.LONG_TERM,
            )
            for text in docs
        ]

    async def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 50,
        user_id=None,
    ):
        """
        Retrieve memories filtered by type.
        """
        docs = await self.adapter.search("", limit=limit)

        return [
            MemoryEntry(
                id=self._generate_id(text),
                content=text,
                type=memory_type,
            )
            for text in docs
        ]

    async def get_session(self, session_key: str):
        """
        Return session conversation history.
        """
        return self.sessions.get(session_key, [])

    async def clear_session(self, session_key: str):
        """
        Clear session history.
        """
        count = len(self.sessions.get(session_key, []))
        self.sessions[session_key] = []
        return count
    