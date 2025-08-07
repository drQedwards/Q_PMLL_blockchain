"""
PostgreSQL storage backend for PMLL
"""

import pickle
from typing import Any, Optional
from .base import StorageBackend

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class PostgreSQLStorageBackend(StorageBackend):
    """PostgreSQL-based storage backend"""
    
    def __init__(self, postgres_url: str):
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("asyncpg not available. Install with: pip install asyncpg")
        
        self.postgres_url = postgres_url
        self.pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL"""
        self.pool = await asyncpg.create_pool(self.postgres_url)
        
        # Create table if not exists
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS pmll_storage (
                    key VARCHAR PRIMARY KEY,
                    value BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            ''')
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
    
    async def store(self, key: str, value: Any) -> None:
        """Store a value in PostgreSQL"""
        serialized = pickle.dumps(value)
        async with self.pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO pmll_storage (key, value) VALUES ($1, $2) '
                'ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()',
                key, serialized
            )
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from PostgreSQL"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT value FROM pmll_storage WHERE key = $1', key)
            if row is None:
                return None
            return pickle.loads(row['value'])
    
    async def delete(self, key: str) -> bool:
        """Delete a value from PostgreSQL"""
        async with self.pool.acquire() as conn:
            result = await conn.execute('DELETE FROM pmll_storage WHERE key = $1', key)
            return result.split()[-1] != '0'
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in PostgreSQL"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT 1 FROM pmll_storage WHERE key = $1', key)
            return row is not None