"""
Common enums for PMLL core components
"""

from enum import Enum


class CompressionLevel(Enum):
    """Compression levels for memory optimization"""
    NONE = 0
    LOW = 1
    BALANCED = 2
    AGGRESSIVE = 3
    MAX = 4


class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"      # < 1B parameters
    MEDIUM = "medium"    # 1B - 7B parameters  
    LARGE = "large"      # 7B - 30B parameters
    XLARGE = "xlarge"    # 30B+ parameters