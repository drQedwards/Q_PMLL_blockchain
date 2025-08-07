"""
Redis storage backend for PMLL
"""

import pickle
from typing import Any, Optional
from .base import StorageBackend

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisStorageBackend(StorageBackend):
    """Redis-based storage backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise RuntimeError("aioredis not available. Install with: pip install aioredis")
        
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self) -> None:
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
    
    async def store(self, key: str, value: Any) -> None:
        """Store a value in Redis"""
        serialized = pickle.dumps(value)
        await self.redis.set(key, serialized)
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from Redis"""
        data = await self.redis.get(key)
        if data is None:
            return None
        return pickle.loads(data)
    
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis"""
        result = await self.redis.delete(key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        return await self.redis.exists(key) > 0