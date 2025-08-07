"""
Abstract base class for PMLL storage backends
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the storage backend"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the storage backend"""
        pass
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> None:
        """Store a value with the given key"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value by key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists"""
        pass