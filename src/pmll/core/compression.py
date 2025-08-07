"""
Memory Compression Engine for PMLL

Advanced compression algorithms for optimal memory utilization in LLM operations.
"""

import zlib
import pickle
import lzma
import bz2
from typing import Any, Union, Optional, Dict
from enum import Enum
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .memory_pool import CompressionLevel


class CompressionAlgorithm(Enum):
    """Available compression algorithms"""
    ZLIB = "zlib"
    LZMA = "lzma"
    BZ2 = "bz2"
    CUSTOM_LLM = "custom_llm"


class MemoryCompressionEngine:
    """
    Advanced memory compression engine optimized for LLM tensors
    """
    
    def __init__(self):
        self.algorithm_map = {
            CompressionLevel.NONE: None,
            CompressionLevel.LOW: CompressionAlgorithm.ZLIB,
            CompressionLevel.BALANCED: CompressionAlgorithm.LZMA,
            CompressionLevel.AGGRESSIVE: CompressionAlgorithm.BZ2,
            CompressionLevel.MAX: CompressionAlgorithm.CUSTOM_LLM
        }
        
    def compress(
        self, 
        data: Union[np.ndarray, 'torch.Tensor', bytes, Any],
        compression_level: CompressionLevel = CompressionLevel.BALANCED
    ) -> bytes:
        """
        Compress data using specified compression level
        
        Args:
            data: Data to compress
            compression_level: Compression level to use
            
        Returns:
            Compressed bytes
        """
        if compression_level == CompressionLevel.NONE:
            return pickle.dumps(data)
            
        # Convert to serializable format
        serialized_data = self._serialize_data(data)
        
        algorithm = self.algorithm_map[compression_level]
        
        if algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(serialized_data, level=6)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.compress(serialized_data, preset=6)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.compress(serialized_data, compresslevel=6)
        elif algorithm == CompressionAlgorithm.CUSTOM_LLM:
            return self._llm_specific_compression(serialized_data)
        else:
            return serialized_data
            
    def decompress(
        self, 
        compressed_data: bytes,
        compression_level: CompressionLevel = CompressionLevel.BALANCED
    ) -> Any:
        """
        Decompress data using specified compression level
        
        Args:
            compressed_data: Compressed bytes
            compression_level: Compression level used
            
        Returns:
            Original data
        """
        if compression_level == CompressionLevel.NONE:
            return pickle.loads(compressed_data)
            
        algorithm = self.algorithm_map[compression_level]
        
        try:
            if algorithm == CompressionAlgorithm.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.LZMA:
                decompressed = lzma.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.BZ2:
                decompressed = bz2.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.CUSTOM_LLM:
                decompressed = self._llm_specific_decompression(compressed_data)
            else:
                decompressed = compressed_data
                
            return self._deserialize_data(decompressed)
            
        except Exception as e:
            raise RuntimeError(f"Decompression failed: {e}")
            
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        if isinstance(data, bytes):
            return data
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # Special handling for PyTorch tensors
            return pickle.dumps({
                'type': 'torch_tensor',
                'data': data.cpu().numpy(),
                'dtype': str(data.dtype),
                'device': str(data.device)
            })
        elif isinstance(data, np.ndarray):
            # Special handling for numpy arrays
            return pickle.dumps({
                'type': 'numpy_array',
                'data': data,
                'dtype': str(data.dtype),
                'shape': data.shape
            })
        else:
            return pickle.dumps(data)
            
    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize bytes back to original data"""
        try:
            data_dict = pickle.loads(data_bytes)
            
            if isinstance(data_dict, dict) and 'type' in data_dict:
                if data_dict['type'] == 'torch_tensor' and TORCH_AVAILABLE:
                    numpy_data = data_dict['data']
                    tensor = torch.from_numpy(numpy_data)
                    return tensor
                elif data_dict['type'] == 'numpy_array':
                    return data_dict['data']
                    
            return data_dict
            
        except Exception:
            # Fallback to direct pickle loading
            return pickle.loads(data_bytes)
            
    def _llm_specific_compression(self, data: bytes) -> bytes:
        """
        Custom compression optimized for LLM tensor patterns
        """
        # Apply multiple compression stages
        stage1 = zlib.compress(data, level=9)
        stage2 = lzma.compress(stage1, preset=9)
        
        # Add custom header
        header = b'PMLL_LLM_COMP_V1'
        return header + stage2
        
    def _llm_specific_decompression(self, data: bytes) -> bytes:
        """
        Custom decompression for LLM-optimized compression
        """
        header = b'PMLL_LLM_COMP_V1'
        
        if not data.startswith(header):
            raise ValueError("Invalid LLM compression format")
            
        compressed_data = data[len(header):]
        stage1 = lzma.decompress(compressed_data)
        return zlib.decompress(stage1)
        
    def get_compression_ratio(
        self, 
        original_data: Any,
        compression_level: CompressionLevel = CompressionLevel.BALANCED
    ) -> float:
        """
        Calculate compression ratio for given data and level
        
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        original_bytes = self._serialize_data(original_data)
        compressed_bytes = self.compress(original_data, compression_level)
        
        return len(original_bytes) / len(compressed_bytes)