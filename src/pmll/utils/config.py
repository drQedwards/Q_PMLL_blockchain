"""
Configuration management for PMLL

Based on the configuration system from the problem statement and PPM tarball.
"""

import os
import toml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class StorageConfig:
    """Storage backend configuration"""
    backend_type: str = "redis"
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://user:pass@localhost/pmll"
    connection_pool_size: int = 10
    connection_timeout: float = 30.0


@dataclass
class MemoryConfig:
    """Memory pool configuration"""
    default_dim: int = 768
    default_slots: int = 8192
    compression_threshold: float = 0.75
    default_ttl: int = 1000
    enable_metrics: bool = True
    device: str = "cpu"


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_prometheus: bool = True
    enable_grafana: bool = False
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None


@dataclass
class ModelConfig:
    """AI model configuration"""
    default_model: str = "gpt2"
    model_cache_dir: str = "./models"
    enable_8bit: bool = False
    max_memory_mb: int = 4096
    device: str = "auto"


@dataclass
class PMLLConfig:
    """Main PMLL configuration"""
    environment: Environment = Environment.DEVELOPMENT
    storage: StorageConfig = field(default_factory=StorageConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'PMLLConfig':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix == '.toml':
            with open(config_path, 'r') as f:
                data = toml.load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PMLLConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'environment' in data:
            config.environment = Environment(data['environment'])
        
        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])
        
        if 'memory' in data:
            config.memory = MemoryConfig(**data['memory'])
        
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
            
        if 'models' in data:
            config.models = ModelConfig(**data['models'])
        
        return config
    
    @classmethod
    def from_env(cls) -> 'PMLLConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Environment
        config.environment = Environment(
            os.getenv('PMLL_ENVIRONMENT', Environment.DEVELOPMENT)
        )
        
        # Storage
        config.storage.backend_type = os.getenv('PMLL_STORAGE_BACKEND', 'redis')
        config.storage.redis_url = os.getenv('REDIS_URL', config.storage.redis_url)
        config.storage.postgres_url = os.getenv('DATABASE_URL', config.storage.postgres_url)
        
        # Memory
        config.memory.default_dim = int(os.getenv('PMLL_DEFAULT_DIM', config.memory.default_dim))
        config.memory.default_slots = int(os.getenv('PMLL_DEFAULT_SLOTS', config.memory.default_slots))
        config.memory.device = os.getenv('PMLL_DEVICE', config.memory.device)
        
        # API
        config.api.host = os.getenv('PMLL_HOST', config.api.host)
        config.api.port = int(os.getenv('PMLL_PORT', config.api.port))
        config.api.workers = int(os.getenv('PMLL_WORKERS', config.api.workers))
        
        # Monitoring  
        config.monitoring.log_level = os.getenv('LOG_LEVEL', config.monitoring.log_level)
        config.monitoring.enable_prometheus = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'
        
        # Models
        config.models.default_model = os.getenv('PMLL_DEFAULT_MODEL', config.models.default_model)
        config.models.device = os.getenv('PMLL_MODEL_DEVICE', config.models.device)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'storage': self.storage.__dict__,
            'memory': self.memory.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__,
            'models': self.models.__dict__,
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        data = self.to_dict()
        
        if config_path.suffix == '.toml':
            with open(config_path, 'w') as f:
                toml.dump(data, f)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def load_config() -> PMLLConfig:
    """Load configuration from various sources"""
    # Try config file first
    config_paths = [
        Path("pmll.toml"),
        Path("config/pmll.toml"),
        Path("/etc/pmll/config.toml"),
        Path.home() / ".pmll" / "config.toml"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            return PMLLConfig.from_file(config_path)
    
    # Fall back to environment variables
    return PMLLConfig.from_env()


def create_default_config() -> None:
    """Create default configuration file"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config = PMLLConfig()
    config_file = config_dir / "pmll.toml"
    
    config.save_to_file(config_file)
    print(f"Created default configuration: {config_file}")


# Load global configuration
try:
    global_config = load_config()
except Exception:
    # Use default config if loading fails
    global_config = PMLLConfig()