"""
Promise-based Queue System for PMLL

Implementation of the Q_promise system from the PPM-6.5.0 tarball,
integrated with the PMLL memory optimization framework.
"""

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Dict, Generic, TypeVar
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import queue

T = TypeVar('T')
U = TypeVar('U')


class PromiseState(Enum):
    """States of a promise"""
    PENDING = "pending"
    FULFILLED = "fulfilled" 
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class QMemNode:
    """
    Memory node in the promise chain - based on Q_promises.c implementation
    """
    index: int
    payload: Any = None
    next: Optional['QMemNode'] = None
    created_at: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return f"QMemNode({self.index}, {self.payload})"


class Promise(Generic[T]):
    """
    PMLL Promise implementation with thenable memory-chain simulation
    
    Based on the Q_promises library from PPM-6.5.0, enhanced for PMLL.
    """
    
    def __init__(self, executor: Optional[Callable] = None):
        self.promise_id = str(uuid.uuid4())
        self.state = PromiseState.PENDING
        self.value: Optional[T] = None
        self.error: Optional[Exception] = None
        self.created_at = time.time()
        
        # Promise chain components
        self.then_callbacks: List[Callable[[T], Any]] = []
        self.catch_callbacks: List[Callable[[Exception], Any]] = []
        self.finally_callbacks: List[Callable[[], None]] = []
        
        # Memory chain for thenable pattern
        self.memory_chain: Optional[QMemNode] = None
        self.chain_length = 0
        
        # Threading support
        self._lock = threading.Lock()
        self._future: Optional[Future] = None
        
        # Auto-execute if executor provided
        if executor:
            self.execute(executor)
            
    def then(self, callback: Callable[[T], U]) -> 'Promise[U]':
        """
        Add success callback (thenable pattern)
        
        Args:
            callback: Function to call when promise is fulfilled
            
        Returns:
            New promise for chaining
        """
        with self._lock:
            if self.state == PromiseState.FULFILLED:
                # Already fulfilled, execute immediately
                try:
                    result = callback(self.value)
                    return Promise.resolve(result)
                except Exception as e:
                    return Promise.reject(e)
            elif self.state == PromiseState.REJECTED:
                # Already rejected, propagate rejection
                return Promise.reject(self.error)
            else:
                # Still pending, add callback
                new_promise = Promise()
                
                def wrapped_callback(value):
                    try:
                        result = callback(value)
                        new_promise._fulfill(result)
                    except Exception as e:
                        new_promise._reject(e)
                        
                self.then_callbacks.append(wrapped_callback)
                return new_promise
                
    def catch(self, callback: Callable[[Exception], T]) -> 'Promise[T]':
        """
        Add error callback
        
        Args:
            callback: Function to call when promise is rejected
            
        Returns:
            New promise for chaining
        """
        with self._lock:
            if self.state == PromiseState.REJECTED:
                # Already rejected, execute immediately
                try:
                    result = callback(self.error)
                    return Promise.resolve(result)
                except Exception as e:
                    return Promise.reject(e)
            elif self.state == PromiseState.FULFILLED:
                # Already fulfilled, propagate fulfillment
                return Promise.resolve(self.value)
            else:
                # Still pending, add callback
                new_promise = Promise()
                
                def wrapped_callback(error):
                    try:
                        result = callback(error)
                        new_promise._fulfill(result)
                    except Exception as e:
                        new_promise._reject(e)
                        
                self.catch_callbacks.append(wrapped_callback)
                return new_promise
                
    def finally_callback(self, callback: Callable[[], None]) -> 'Promise[T]':
        """
        Add finally callback that executes regardless of outcome
        
        Args:
            callback: Function to call when promise settles
            
        Returns:
            This promise for chaining
        """
        with self._lock:
            if self.state != PromiseState.PENDING:
                # Already settled, execute immediately
                try:
                    callback()
                except Exception:
                    pass  # Finally callbacks don't affect promise outcome
            else:
                self.finally_callbacks.append(callback)
                
        return self
        
    def execute(self, executor: Callable) -> None:
        """
        Execute the promise with given executor function
        
        Args:
            executor: Function that performs the async operation
        """
        def run_executor():
            try:
                result = executor()
                self._fulfill(result)
            except Exception as e:
                self._reject(e)
                
        # Run in thread pool to avoid blocking
        self._future = PromiseQueue.get_thread_pool().submit(run_executor)
        
    def wait(self, timeout: Optional[float] = None) -> T:
        """
        Wait for promise to complete (blocking)
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Promise value
            
        Raises:
            Exception: If promise is rejected
            TimeoutError: If timeout exceeded
        """
        if self._future:
            try:
                self._future.result(timeout=timeout)
            except Exception:
                pass  # Handled by promise state
                
        # Wait for state change
        start_time = time.time()
        while self.state == PromiseState.PENDING:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Promise timeout exceeded")
            time.sleep(0.001)  # Small sleep to avoid busy waiting
            
        if self.state == PromiseState.FULFILLED:
            return self.value
        elif self.state == PromiseState.REJECTED:
            raise self.error
        else:
            raise RuntimeError(f"Promise in unexpected state: {self.state}")
            
    def is_pending(self) -> bool:
        """Check if promise is still pending"""
        return self.state == PromiseState.PENDING
        
    def is_fulfilled(self) -> bool:
        """Check if promise is fulfilled"""
        return self.state == PromiseState.FULFILLED
        
    def is_rejected(self) -> bool:
        """Check if promise is rejected"""
        return self.state == PromiseState.REJECTED
        
    def _fulfill(self, value: T) -> None:
        """Internal method to fulfill the promise"""
        with self._lock:
            if self.state != PromiseState.PENDING:
                return
                
            self.state = PromiseState.FULFILLED
            self.value = value
            
            # Create memory chain for thenable pattern
            self._create_memory_chain()
            
            # Execute callbacks
            for callback in self.then_callbacks:
                try:
                    callback(value)
                except Exception:
                    pass  # Callback errors don't affect promise state
                    
            for callback in self.finally_callbacks:
                try:
                    callback()
                except Exception:
                    pass
                    
    def _reject(self, error: Exception) -> None:
        """Internal method to reject the promise"""
        with self._lock:
            if self.state != PromiseState.PENDING:
                return
                
            self.state = PromiseState.REJECTED
            self.error = error
            
            # Execute callbacks
            for callback in self.catch_callbacks:
                try:
                    callback(error)
                except Exception:
                    pass
                    
            for callback in self.finally_callbacks:
                try:
                    callback()
                except Exception:
                    pass
                    
    def _create_memory_chain(self) -> None:
        """Create memory chain for thenable pattern (from Q_promises.c)"""
        if not self.then_callbacks:
            return
            
        chain_length = len(self.then_callbacks) + 1
        head = None
        prev = None
        
        for i in range(chain_length):
            node = QMemNode(
                index=i,
                payload="Known" if i % 2 == 0 else "Unknown"
            )
            
            if prev:
                prev.next = node
            else:
                head = node
                
            prev = node
            
        self.memory_chain = head
        self.chain_length = chain_length
        
    @staticmethod
    def resolve(value: T) -> 'Promise[T]':
        """Create a fulfilled promise with the given value"""
        promise = Promise()
        promise._fulfill(value)
        return promise
        
    @staticmethod
    def reject(error: Exception) -> 'Promise':
        """Create a rejected promise with the given error"""
        promise = Promise()
        promise._reject(error)
        return promise
        
    @staticmethod
    def all(promises: List['Promise']) -> 'Promise[List[Any]]':
        """
        Wait for all promises to complete
        
        Args:
            promises: List of promises to wait for
            
        Returns:
            Promise that resolves to list of all results
        """
        def executor():
            results = []
            for promise in promises:
                results.append(promise.wait())
            return results
            
        return Promise(executor)
        
    @staticmethod
    def race(promises: List['Promise']) -> 'Promise':
        """
        Return the first promise to complete
        
        Args:
            promises: List of promises to race
            
        Returns:
            Promise that resolves to the first completed result
        """
        def executor():
            futures = [p._future for p in promises if p._future]
            if not futures:
                raise RuntimeError("No executable promises provided")
                
            # Use ThreadPoolExecutor's wait functionality
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                return future.result()
                
        return Promise(executor)


class PromiseQueue:
    """
    Queue system for managing promise execution with PMLL optimization
    """
    
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _instance: Optional['PromiseQueue'] = None
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.queue: queue.Queue = queue.Queue()
        self.active_promises: Dict[str, Promise] = {}
        self.completed_promises: Dict[str, Promise] = {}
        self.metrics = {
            "enqueued": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "average_execution_time": 0.0
        }
        
        # Initialize thread pool
        if PromiseQueue._thread_pool is None:
            PromiseQueue._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            
    @classmethod
    def get_instance(cls) -> 'PromiseQueue':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    @classmethod
    def get_thread_pool(cls) -> ThreadPoolExecutor:
        """Get shared thread pool"""
        if cls._thread_pool is None:
            cls._thread_pool = ThreadPoolExecutor(max_workers=10)
        return cls._thread_pool
        
    def enqueue(self, operation: Callable) -> Promise:
        """
        Enqueue an operation for asynchronous execution
        
        Args:
            operation: Function to execute asynchronously
            
        Returns:
            Promise that will resolve when operation completes
        """
        promise = Promise(operation)
        
        self.active_promises[promise.promise_id] = promise
        self.metrics["enqueued"] += 1
        
        # Add completion handler
        def on_complete():
            if promise.promise_id in self.active_promises:
                completed_promise = self.active_promises.pop(promise.promise_id)
                self.completed_promises[promise.promise_id] = completed_promise
                
                if completed_promise.is_fulfilled():
                    self.metrics["completed"] += 1
                elif completed_promise.is_rejected():
                    self.metrics["failed"] += 1
                    
        promise.finally_callback(on_complete)
        return promise
        
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "max_workers": self.max_workers,
            "active_promises": len(self.active_promises),
            "completed_promises": len(self.completed_promises),
            "queue_size": self.queue.qsize(),
            "metrics": self.metrics.copy()
        }
        
    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Clean up old completed promises"""
        current_time = time.time()
        to_remove = []
        
        for promise_id, promise in self.completed_promises.items():
            if current_time - promise.created_at > max_age_seconds:
                to_remove.append(promise_id)
                
        for promise_id in to_remove:
            del self.completed_promises[promise_id]
            
        return len(to_remove)


# Integration with Q_promises.c functionality
def q_then_callback(index: int, payload: str) -> None:
    """Callback function compatible with Q_promises.c thenable pattern"""
    print(f"Q_then callback: Node {index} -> {payload}")
    

def create_memory_chain(length: int) -> Optional[QMemNode]:
    """
    Create memory chain compatible with Q_promises.c
    
    Args:
        length: Length of the chain to create
        
    Returns:
        Head node of the created chain
    """
    if length == 0:
        return None
        
    head = None
    prev = None
    
    for i in range(length):
        node = QMemNode(
            index=i,
            payload="Known" if i % 2 == 0 else "Unknown"
        )
        
        if prev:
            prev.next = node
        else:
            head = node
            
        prev = node
        
    return head


def q_then(head: QMemNode, callback: Callable[[int, str], None]) -> None:
    """
    Execute callback for each node in the chain (Q_promises.c compatibility)
    
    Args:
        head: Head node of the chain
        callback: Callback function to execute
    """
    current = head
    while current:
        if callback:
            callback(current.index, current.payload)
        current = current.next


def free_memory_chain(head: QMemNode) -> None:
    """
    Free memory chain (Q_promises.c compatibility)
    
    Args:
        head: Head node of the chain to free
    """
    # Python garbage collection handles memory automatically
    # This is provided for compatibility with C implementation
    current = head
    while current:
        next_node = current.next
        current.next = None  # Break reference
        current = next_node