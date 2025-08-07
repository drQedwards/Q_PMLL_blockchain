"""
PMLL - Persistent Memory Logic Loop

Memory optimization system for Large Language Models with hierarchical compression,
queue-theoretic promise semantics, and production-ready deployment capabilities.

Based on the formal proof that P = NP using the PMLL Algorithm by Dr. Josef Kurk Edwards.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core.memory_pool import PMLLMemoryPool
from .core.enums import CompressionLevel
from .core.compression import MemoryCompressionEngine
from .core.promises import Promise, PromiseQueue
from .core.pmll_solver import PMLLSATSolver
# Storage
from .storage.base import StorageBackend
# Optional backends - import only if available
try:
    from .storage.redis_backend import RedisStorageBackend
except ImportError:
    RedisStorageBackend = None

try:
    from .storage.postgres_backend import PostgreSQLStorageBackend  
except ImportError:
    PostgreSQLStorageBackend = None

# API components
from .api.app import create_app

# Integrations
from .integrations.pytorch import PMLLMultiheadAttention
from .integrations.huggingface import PMLLBertAttention
from .integrations.transformer_engine import TransformerEngine

__all__ = [
    "__version__",
    # Core components
    "PMLLMemoryPool",
    "CompressionLevel", 
    "MemoryCompressionEngine",
    "Promise",
    "PromiseQueue",
    "PMLLSATSolver",
    # Storage
    "StorageBackend",
    "RedisStorageBackend", 
    "PostgreSQLStorageBackend",
    # API
    "create_app",
    # Integrations
    "PMLLMultiheadAttention",
    "PMLLBertAttention",
    "TransformerEngine",
]

# Package metadata
__author__ = "PMLL Team"
__email__ = "team@pmll.ai"
__license__ = "Apache 2.0"
__description__ = "Persistent Memory Logic Loop for LLM memory optimization"

# Integration with pypm concepts
PMLL_VERSION = __version__
PYPM_COMPATIBILITY = True