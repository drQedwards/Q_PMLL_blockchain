"""
Hash Functions and Consistent Hashing for PMLL

Implementation of hash functions for optimal slot assignment in memory pools.
"""

import hashlib
import bisect
from typing import List, Dict, Any, Optional


class ConsistentHasher:
    """
    Consistent hashing implementation for memory slot assignment
    
    Provides even distribution and minimal reshuffling when slots are added/removed.
    """
    
    def __init__(self, num_slots: int, replicas: int = 150):
        """
        Initialize consistent hasher
        
        Args:
            num_slots: Number of memory slots
            replicas: Number of virtual replicas per slot (for better distribution)
        """
        self.num_slots = num_slots
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        
        self._build_ring()
        
    def _build_ring(self) -> None:
        """Build the consistent hash ring"""
        self.ring.clear()
        self.sorted_keys.clear()
        
        for slot_idx in range(self.num_slots):
            slot_id = f"slot_{slot_idx}"
            
            for replica in range(self.replicas):
                key = self._hash(f"{slot_id}:{replica}")
                self.ring[key] = slot_id
                
        self.sorted_keys = sorted(self.ring.keys())
        
    def _hash(self, data: str) -> int:
        """Generate hash value for given data"""
        return int(hashlib.md5(data.encode()).hexdigest(), 16)
        
    def get_slot(self, key: str) -> str:
        """
        Get assigned slot for given key
        
        Args:
            key: Key to hash
            
        Returns:
            Assigned slot ID
        """
        if not self.sorted_keys:
            return "slot_0"
            
        hash_value = self._hash(key)
        
        # Find the first slot with hash >= hash_value
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        if idx == len(self.sorted_keys):
            # Wrap around to the beginning of the ring
            idx = 0
            
        return self.ring[self.sorted_keys[idx]]
        
    def add_slot(self) -> None:
        """Add a new slot to the ring"""
        self.num_slots += 1
        self._build_ring()
        
    def remove_slot(self, slot_id: str) -> None:
        """Remove a slot from the ring"""
        if self.num_slots <= 1:
            return  # Keep at least one slot
            
        # Remove all replicas of this slot
        keys_to_remove = [
            key for key, sid in self.ring.items() 
            if sid == slot_id
        ]
        
        for key in keys_to_remove:
            del self.ring[key]
            
        self.sorted_keys = sorted(self.ring.keys())
        self.num_slots -= 1
        
    def get_distribution_stats(self) -> Dict[str, int]:
        """Get distribution statistics for debugging"""
        distribution = {}
        
        # Sample 10000 random keys to see distribution
        import random
        test_keys = [f"test_key_{i}" for i in range(10000)]
        
        for key in test_keys:
            slot = self.get_slot(key)
            distribution[slot] = distribution.get(slot, 0) + 1
            
        return distribution


class BloomFilter:
    """
    Bloom filter for fast membership testing in memory pools
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter
        
        Args:
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal parameters
        self.size = self._calculate_size(expected_items, false_positive_rate)
        self.hash_count = self._calculate_hash_count(self.size, expected_items)
        
        # Bit array
        self.bit_array = [False] * self.size
        self.items_added = 0
        
    def _calculate_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size"""
        import math
        return int(-n * math.log(p) / (math.log(2) ** 2))
        
    def _calculate_hash_count(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions"""
        import math
        return int(m * math.log(2) / n)
        
    def _hash_functions(self, item: str) -> List[int]:
        """Generate multiple hash values for an item"""
        hashes = []
        
        # Use different hash algorithms
        hash1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        hash2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        
        for i in range(self.hash_count):
            # Combine hashes with different coefficients
            combined_hash = (hash1 + i * hash2) % self.size
            hashes.append(combined_hash)
            
        return hashes
        
    def add(self, item: str) -> None:
        """Add an item to the bloom filter"""
        hashes = self._hash_functions(item)
        
        for hash_val in hashes:
            self.bit_array[hash_val] = True
            
        self.items_added += 1
        
    def contains(self, item: str) -> bool:
        """
        Check if item might be in the set
        
        Returns:
            True if item might be present (or false positive)
            False if item is definitely not present
        """
        hashes = self._hash_functions(item)
        
        for hash_val in hashes:
            if not self.bit_array[hash_val]:
                return False
                
        return True
        
    def estimated_false_positive_rate(self) -> float:
        """Calculate current estimated false positive rate"""
        if self.items_added == 0:
            return 0.0
            
        import math
        
        # Calculate the probability that a bit is still 0
        prob_zero = (1 - 1/self.size) ** (self.hash_count * self.items_added)
        
        # False positive rate
        return (1 - prob_zero) ** self.hash_count
        
    def get_stats(self) -> Dict[str, Any]:
        """Get bloom filter statistics"""
        bits_set = sum(self.bit_array)
        
        return {
            "size": self.size,
            "hash_count": self.hash_count,
            "items_added": self.items_added,
            "bits_set": bits_set,
            "bits_set_ratio": bits_set / self.size,
            "expected_false_positive_rate": self.false_positive_rate,
            "estimated_false_positive_rate": self.estimated_false_positive_rate()
        }


class LocalitySensitiveHasher:
    """
    Locality-Sensitive Hashing for similar tensor detection
    """
    
    def __init__(self, num_bands: int = 20, rows_per_band: int = 5):
        """
        Initialize LSH
        
        Args:
            num_bands: Number of hash bands
            rows_per_band: Number of rows per band
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.hash_tables: List[Dict[str, List[str]]] = [
            {} for _ in range(num_bands)
        ]
        
    def _minhash_signature(self, data: Any, signature_size: int = 100) -> List[int]:
        """Generate MinHash signature for data"""
        import random
        
        # Convert data to string representation
        if hasattr(data, 'tobytes'):
            data_str = str(data.tobytes())
        else:
            data_str = str(data)
            
        # Generate hash functions
        signature = []
        random.seed(42)  # For reproducibility
        
        for i in range(signature_size):
            # Create different hash functions using random coefficients
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            
            hash_val = (a * hash(data_str + str(i)) + b) % (2**32)
            signature.append(hash_val)
            
        return signature
        
    def add_item(self, item_id: str, data: Any) -> None:
        """Add item to LSH index"""
        signature = self._minhash_signature(data)
        
        # Split signature into bands
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            band_signature = tuple(signature[start_idx:end_idx])
            band_hash = str(hash(band_signature))
            
            if band_hash not in self.hash_tables[band_idx]:
                self.hash_tables[band_idx][band_hash] = []
                
            self.hash_tables[band_idx][band_hash].append(item_id)
            
    def find_similar(self, data: Any, threshold: float = 0.8) -> List[str]:
        """
        Find similar items to given data
        
        Args:
            data: Data to find similarities for
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar item IDs
        """
        signature = self._minhash_signature(data)
        candidates = set()
        
        # Find candidates from all bands
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            band_signature = tuple(signature[start_idx:end_idx])
            band_hash = str(hash(band_signature))
            
            if band_hash in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_hash])
                
        return list(candidates)
        
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity between two signatures"""
        if len(sig1) != len(sig2):
            return 0.0
            
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)