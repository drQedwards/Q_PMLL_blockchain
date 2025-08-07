"""
Performance Metrics and Monitoring for PMLL

Comprehensive metrics collection and analysis for memory pool operations.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric measurement"""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    

class MemoryMetrics:
    """
    Memory pool performance metrics collector
    """
    
    def __init__(self, history_size: int = 10000):
        """
        Initialize metrics collector
        
        Args:
            history_size: Number of historical points to keep
        """
        self.history_size = history_size
        self._lock = threading.Lock()
        
        # Metric storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Operation tracking
        self.operations: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "stores": 0,
            "deletes": 0
        }
        
        # Performance tracking
        self.start_time = time.time()
        
    def record_store(self, tensor_id: str, compressed: bool, slot_id: str) -> None:
        """Record a store operation"""
        with self._lock:
            self.counters["stores_total"] += 1
            self.cache_stats["stores"] += 1
            
            if compressed:
                self.counters["stores_compressed"] += 1
                
            # Record operation
            self.operations["store"].append(MetricPoint(
                timestamp=time.time(),
                value=1,
                labels={"tensor_id": tensor_id, "slot_id": slot_id, "compressed": str(compressed)}
            ))
            
            # Keep history manageable
            if len(self.operations["store"]) > self.history_size:
                self.operations["store"] = self.operations["store"][-self.history_size:]
                
    def record_hit(self, tensor_id: str, slot_id: str) -> None:
        """Record a cache hit"""
        with self._lock:
            self.counters["hits_total"] += 1
            self.cache_stats["hits"] += 1
            
            self.operations["hit"].append(MetricPoint(
                timestamp=time.time(),
                value=1,
                labels={"tensor_id": tensor_id, "slot_id": slot_id}
            ))
            
            if len(self.operations["hit"]) > self.history_size:
                self.operations["hit"] = self.operations["hit"][-self.history_size:]
                
    def record_miss(self, tensor_id: str) -> None:
        """Record a cache miss"""
        with self._lock:
            self.counters["misses_total"] += 1
            self.cache_stats["misses"] += 1
            
            self.operations["miss"].append(MetricPoint(
                timestamp=time.time(),
                value=1,
                labels={"tensor_id": tensor_id}
            ))
            
            if len(self.operations["miss"]) > self.history_size:
                self.operations["miss"] = self.operations["miss"][-self.history_size:]
                
    def record_eviction(self, tensor_id: str) -> None:
        """Record a cache eviction"""
        with self._lock:
            self.counters["evictions_total"] += 1
            self.cache_stats["evictions"] += 1
            
            self.operations["eviction"].append(MetricPoint(
                timestamp=time.time(),
                value=1,
                labels={"tensor_id": tensor_id}
            ))
            
    def record_delete(self, tensor_id: str) -> None:
        """Record a delete operation"""
        with self._lock:
            self.counters["deletes_total"] += 1
            self.cache_stats["deletes"] += 1
            
    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation execution time"""
        with self._lock:
            if operation not in self.timers:
                self.timers[operation] = []
                
            self.timers[operation].append(duration)
            
            # Keep recent measurements
            if len(self.timers[operation]) > self.history_size:
                self.timers[operation] = self.timers[operation][-self.history_size:]
                
            # Update histogram
            self.histograms[f"{operation}_duration"].append(duration)
            
    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value"""
        with self._lock:
            self.gauges[name] = value
            
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment counter"""
        with self._lock:
            self.counters[name] += value
            
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        return self.cache_stats["hits"] / total_requests
        
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        if operation not in self.timers or not self.timers[operation]:
            return {}
            
        durations = self.timers[operation]
        
        return {
            "count": len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "avg_time": statistics.mean(durations),
            "median_time": statistics.median(durations),
            "p95_time": self._percentile(durations, 95),
            "p99_time": self._percentile(durations, 99),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }
        
    def get_throughput_stats(self, window_seconds: int = 300) -> Dict[str, float]:
        """
        Get throughput statistics for recent time window
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Throughput statistics
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        stats = {}
        
        for operation_type, points in self.operations.items():
            recent_points = [
                p for p in points 
                if p.timestamp >= cutoff_time
            ]
            
            if recent_points:
                stats[f"{operation_type}_per_second"] = len(recent_points) / window_seconds
                stats[f"{operation_type}_count"] = len(recent_points)
            else:
                stats[f"{operation_type}_per_second"] = 0.0
                stats[f"{operation_type}_count"] = 0
                
        return stats
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": uptime,
            "cache_statistics": {
                "hit_rate": self.get_cache_hit_rate(),
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "evictions": self.cache_stats["evictions"],
                "stores": self.cache_stats["stores"],
                "deletes": self.cache_stats["deletes"]
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "throughput": self.get_throughput_stats(),
            "operation_performance": {}
        }
        
        # Add operation performance statistics
        for operation in self.timers.keys():
            summary["operation_performance"][operation] = self.get_operation_stats(operation)
            
        return summary
        
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.operations.clear()
            
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "stores": 0,
                "deletes": 0
            }
            
            self.start_time = time.time()
            
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


class PerformanceProfiler:
    """
    Performance profiler for PMLL operations
    """
    
    def __init__(self):
        self.active_timers: Dict[str, float] = {}
        self.metrics = MemoryMetrics()
        self._lock = threading.Lock()
        
    def start_timer(self, name: str) -> None:
        """Start timing an operation"""
        with self._lock:
            self.active_timers[name] = time.time()
            
    def end_timer(self, name: str) -> float:
        """End timing and record duration"""
        with self._lock:
            if name not in self.active_timers:
                return 0.0
                
            start_time = self.active_timers.pop(name)
            duration = time.time() - start_time
            
            self.metrics.record_operation_time(name, duration)
            return duration
            
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return TimerContext(self, operation_name)
        
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        return self.metrics.get_summary()


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        
    def __enter__(self):
        self.profiler.start_timer(self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.operation_name)


# Global metrics instance
global_metrics = MemoryMetrics()
global_profiler = PerformanceProfiler()