"""
PMLL Memory Pool - Core memory management system

Implementation of the Persistent Memory Logic Loop memory pool with hierarchical
compression and queue-theoretic promise semantics.
"""

import uuid
import time
import math
import threading
from typing import Dict, Optional, List, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .compression import MemoryCompressionEngine
from .promises import Promise, PromiseQueue
from .hash_functions import ConsistentHasher
from .metrics import MemoryMetrics


class CompressionLevel(Enum):
    """Compression levels for memory optimization"""
    NONE = 0
    LOW = 1
    BALANCED = 2
    AGGRESSIVE = 3
    MAX = 4


@dataclass
class MemorySlot:
    """Individual memory slot in the pool"""
    slot_id: str
    tensor_data: Optional[Any] = None
    compressed_data: Optional[bytes] = None
    metadata: Dict[str, Any] = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    compression_level: CompressionLevel = CompressionLevel.NONE
    promise_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.last_accessed = time.time()


class PMLLMemoryPool:
    """
    Persistent Memory Logic Loop Memory Pool
    
    Implements hierarchical memory management with compression, promise-based
    queuing, and Ouroboros-pattern caching for optimal LLM memory utilization.
    """
    
    def __init__(
        self,
        dim: int = 768,
        slots: int = 8192,
        compression_level: CompressionLevel = CompressionLevel.BALANCED,
        compression_threshold: float = 0.75,
        ttl_seconds: int = 3600,
        device: str = "cpu",
        enable_ouroboros: bool = True
    ):
        """
        Initialize PMLL Memory Pool
        
        Args:
            dim: Dimensionality of tensors (e.g., 768 for BERT)
            slots: Number of memory slots
            compression_level: Default compression level
            compression_threshold: Threshold for automatic compression
            ttl_seconds: Time-to-live for cached items
            device: PyTorch device ('cpu', 'cuda', 'mps')
            enable_ouroboros: Enable Ouroboros recursive caching
        """
        self.dim = dim
        self.slots = slots
        self.compression_level = compression_level
        self.compression_threshold = compression_threshold
        self.ttl_seconds = ttl_seconds
        self.device = device
        self.enable_ouroboros = enable_ouroboros
        
        # Core storage
        self.memory_slots: Dict[str, MemorySlot] = {}
        self.slot_assignments: Dict[str, str] = {}  # tensor_id -> slot_id
        
        # Hash-based slot assignment
        self.hasher = ConsistentHasher(slots)
        
        # Compression engine
        self.compression_engine = MemoryCompressionEngine()
        
        # Promise-based queuing system
        self.promise_queue = PromiseQueue()
        
        # Metrics tracking
        self.metrics = MemoryMetrics()
        
        # Threading lock
        self._lock = threading.RLock()
        
        # Ouroboros cache hierarchy
        if enable_ouroboros:
            self._ouroboros_cache = {}
            self._cache_depth = int(math.log2(max(slots, 2)))
        
    def store(
        self, 
        tensor: Union[np.ndarray, 'torch.Tensor', bytes],
        tensor_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress: Optional[bool] = None
    ) -> str:
        """
        Store tensor in memory pool with optional compression
        
        Args:
            tensor: Input tensor data
            tensor_id: Optional tensor ID (generated if not provided)
            metadata: Optional metadata
            compress: Force compression (None = automatic)
            
        Returns:
            Tensor ID for retrieval
        """
        if tensor_id is None:
            tensor_id = str(uuid.uuid4())
            
        with self._lock:
            # Determine slot assignment using consistent hashing
            slot_id = self.hasher.get_slot(tensor_id)
            
            # Handle slot collision/eviction
            if slot_id in self.memory_slots:
                self._evict_slot(slot_id)
                
            # Create memory slot
            memory_slot = MemorySlot(
                slot_id=slot_id,
                metadata=metadata or {},
                compression_level=self.compression_level
            )
            
            # Determine if compression should be used
            should_compress = compress
            if should_compress is None:
                memory_usage = len(self.memory_slots) / self.slots
                should_compress = memory_usage > self.compression_threshold
                
            # Store tensor data
            if should_compress:
                memory_slot.compressed_data = self.compression_engine.compress(
                    tensor, 
                    self.compression_level
                )
                memory_slot.tensor_data = None
            else:
                memory_slot.tensor_data = self._normalize_tensor(tensor)
                memory_slot.compressed_data = None
                
            # Store in memory pool
            self.memory_slots[slot_id] = memory_slot
            self.slot_assignments[tensor_id] = slot_id
            
            # Update Ouroboros cache if enabled
            if self.enable_ouroboros:
                self._update_ouroboros_cache(tensor_id, slot_id)
                
            # Update metrics
            self.metrics.record_store(
                tensor_id=tensor_id,
                compressed=should_compress,
                slot_id=slot_id
            )
            
            return tensor_id
            
    def retrieve(self, tensor_id: str) -> Optional[Any]:
        """
        Retrieve tensor from memory pool
        
        Args:
            tensor_id: Tensor ID
            
        Returns:
            Retrieved tensor or None if not found
        """
        with self._lock:
            # Check if tensor exists
            if tensor_id not in self.slot_assignments:
                self.metrics.record_miss(tensor_id)
                return None
                
            slot_id = self.slot_assignments[tensor_id]
            
            if slot_id not in self.memory_slots:
                # Slot was evicted, remove assignment
                del self.slot_assignments[tensor_id]
                self.metrics.record_miss(tensor_id)
                return None
                
            memory_slot = self.memory_slots[slot_id]
            
            # Update access statistics
            memory_slot.last_accessed = time.time()
            memory_slot.access_count += 1
            
            # Retrieve tensor data
            if memory_slot.compressed_data is not None:
                # Decompress data
                tensor_data = self.compression_engine.decompress(
                    memory_slot.compressed_data,
                    memory_slot.compression_level
                )
            else:
                tensor_data = memory_slot.tensor_data
                
            # Update metrics
            self.metrics.record_hit(tensor_id, slot_id)
            
            # Update Ouroboros cache access pattern
            if self.enable_ouroboros:
                self._access_ouroboros_cache(tensor_id, slot_id)
                
            return tensor_data
            
    def store_async(
        self,
        tensor: Union[np.ndarray, 'torch.Tensor', bytes],
        tensor_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress: Optional[bool] = None
    ) -> Promise:
        """
        Store tensor asynchronously using promise-based queuing
        
        Args:
            tensor: Input tensor data
            tensor_id: Optional tensor ID
            metadata: Optional metadata
            compress: Force compression
            
        Returns:
            Promise that resolves to tensor ID
        """
        def store_operation():
            return self.store(tensor, tensor_id, metadata, compress)
            
        promise = self.promise_queue.enqueue(store_operation)
        return promise
        
    def retrieve_async(self, tensor_id: str) -> Promise:
        """
        Retrieve tensor asynchronously using promise-based queuing
        
        Args:
            tensor_id: Tensor ID
            
        Returns:
            Promise that resolves to tensor data
        """
        def retrieve_operation():
            return self.retrieve(tensor_id)
            
        promise = self.promise_queue.enqueue(retrieve_operation)
        return promise
        
    def delete(self, tensor_id: str) -> bool:
        """
        Delete tensor from memory pool
        
        Args:
            tensor_id: Tensor ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if tensor_id not in self.slot_assignments:
                return False
                
            slot_id = self.slot_assignments[tensor_id]
            
            # Remove from slot assignments
            del self.slot_assignments[tensor_id]
            
            # Remove from memory slots if it exists
            if slot_id in self.memory_slots:
                del self.memory_slots[slot_id]
                
            # Update Ouroboros cache
            if self.enable_ouroboros and tensor_id in self._ouroboros_cache:
                del self._ouroboros_cache[tensor_id]
                
            self.metrics.record_delete(tensor_id)
            return True
            
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries based on TTL
        
        Returns:
            Number of expired entries removed
        """
        current_time = time.time()
        expired_count = 0
        
        with self._lock:
            expired_slots = []
            
            for slot_id, memory_slot in self.memory_slots.items():
                if current_time - memory_slot.last_accessed > self.ttl_seconds:
                    expired_slots.append(slot_id)
                    
            for slot_id in expired_slots:
                # Find tensor_id for this slot
                tensor_id = None
                for tid, sid in self.slot_assignments.items():
                    if sid == slot_id:
                        tensor_id = tid
                        break
                        
                if tensor_id:
                    self.delete(tensor_id)
                    expired_count += 1
                    
        return expired_count
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics"""
        with self._lock:
            total_slots = len(self.memory_slots)
            compressed_slots = sum(
                1 for slot in self.memory_slots.values() 
                if slot.compressed_data is not None
            )
            
            return {
                "pool_config": {
                    "dimensions": self.dim,
                    "total_slots": self.slots,
                    "compression_level": self.compression_level.name,
                    "compression_threshold": self.compression_threshold,
                    "ttl_seconds": self.ttl_seconds,
                    "device": self.device,
                    "ouroboros_enabled": self.enable_ouroboros
                },
                "usage": {
                    "occupied_slots": total_slots,
                    "free_slots": self.slots - total_slots,
                    "utilization": total_slots / self.slots,
                    "compressed_slots": compressed_slots,
                    "compression_ratio": compressed_slots / max(1, total_slots)
                },
                "performance": self.metrics.get_summary(),
                "ouroboros_cache": {
                    "enabled": self.enable_ouroboros,
                    "cache_depth": getattr(self, '_cache_depth', 0),
                    "cached_entries": len(getattr(self, '_ouroboros_cache', {}))
                }
            }
            
    def _normalize_tensor(self, tensor: Union[np.ndarray, 'torch.Tensor', bytes]) -> Any:
        """Normalize tensor to consistent format"""
        if isinstance(tensor, bytes):
            return tensor
        elif TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            # Try to convert to numpy array
            return np.array(tensor)
            
    def _evict_slot(self, slot_id: str) -> None:
        """Evict existing slot using LRU policy"""
        if slot_id in self.memory_slots:
            # Find tensor_id for eviction
            tensor_id_to_evict = None
            for tid, sid in self.slot_assignments.items():
                if sid == slot_id:
                    tensor_id_to_evict = tid
                    break
                    
            if tensor_id_to_evict:
                self.delete(tensor_id_to_evict)
                self.metrics.record_eviction(tensor_id_to_evict)
                
    def _update_ouroboros_cache(self, tensor_id: str, slot_id: str) -> None:
        """Update Ouroboros recursive cache pattern"""
        if not self.enable_ouroboros:
            return
            
        self._ouroboros_cache[tensor_id] = {
            "slot_id": slot_id,
            "depth": 0,
            "parent": None,
            "children": [],
            "access_pattern": []
        }
        
        # Implement recursive caching pattern
        cache_key = hash(tensor_id) % len(self._ouroboros_cache)
        if cache_key != 0:  # Avoid infinite recursion
            parent_key = cache_key // 2
            if parent_key in self._ouroboros_cache:
                self._ouroboros_cache[tensor_id]["parent"] = parent_key
                
    def _access_ouroboros_cache(self, tensor_id: str, slot_id: str) -> None:
        """Update Ouroboros cache access patterns"""
        if not self.enable_ouroboros or tensor_id not in self._ouroboros_cache:
            return
            
        cache_entry = self._ouroboros_cache[tensor_id]
        cache_entry["access_pattern"].append(time.time())
        
        # Keep only recent access patterns
        cache_entry["access_pattern"] = cache_entry["access_pattern"][-10:]