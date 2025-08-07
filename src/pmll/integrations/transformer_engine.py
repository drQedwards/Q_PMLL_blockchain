"""
PMLL Transformer Engine - AI Model Integration

Complete AI inference engine using PMLL memory optimization with transformer models.
Based on the Transformer.py from PPM-6.5.0 tarball.
"""

import os
import sys
import subprocess
import asyncio
import time
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModel,
        pipeline,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.memory_pool import PMLLMemoryPool
from ..core.enums import CompressionLevel
from ..core.promises import Promise, PromiseQueue
from ..core.pmll_solver import PMLLSATSolver
from ..core.metrics import global_metrics, global_profiler


from ..core.enums import ModelSize


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    model_id: str
    model_size: ModelSize
    device: str = "auto"
    use_8bit: bool = False
    use_pmll_memory: bool = True
    max_memory_mb: int = 4096
    compression_level: CompressionLevel = CompressionLevel.BALANCED


class TransformerEngine:
    """
    PMLL-optimized transformer engine for AI inference
    
    Integrates PMLL memory optimization with transformer models for efficient AI inference.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Transformer Engine
        
        Args:
            config: Model configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers not available. Install with: pip install transformers torch"
            )
            
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.embedding_model = None
        
        # PMLL components
        self.memory_pool = None
        self.promise_queue = PromiseQueue.get_instance()
        self.sat_solver = None
        
        # Model state
        self.is_loaded = False
        self.generation_config = None
        
        # Performance tracking
        self.inference_count = 0
        self.total_tokens_generated = 0
        
        # Initialize PMLL memory if enabled
        if config.use_pmll_memory:
            self._initialize_pmll_memory()
            
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
        
    def _initialize_pmll_memory(self) -> None:
        """Initialize PMLL memory optimization"""
        # Calculate dimensions based on model
        if "gpt2" in self.config.model_id.lower():
            dim = 768
        elif "bert" in self.config.model_id.lower():
            dim = 768
        elif "llama" in self.config.model_id.lower():
            dim = 4096
        elif "mistral" in self.config.model_id.lower():
            dim = 4096
        else:
            dim = 768  # Default
            
        # Calculate slots based on memory limit
        slots = (self.config.max_memory_mb * 1024 * 1024) // (dim * 4)  # 4 bytes per float
        slots = min(max(slots, 1024), 65536)  # Clamp between 1K and 64K
        
        self.memory_pool = PMLLMemoryPool(
            dim=dim,
            slots=slots,
            compression_level=self.config.compression_level,
            device=self.device,
            enable_ouroboros=True
        )
        
        # Initialize SAT solver for optimization problems
        self.sat_solver = PMLLSATSolver(
            num_vars=100,  # Start with small problem size
            enable_ouroboros=True
        )
        
    async def load_model_async(self) -> None:
        """Load model asynchronously"""
        def load_operation():
            return self._load_model_sync()
            
        promise = self.promise_queue.enqueue(load_operation)
        await asyncio.create_task(self._promise_to_task(promise))
        
    def _load_model_sync(self) -> bool:
        """Synchronous model loading"""
        try:
            with global_profiler.time_operation("model_loading"):
                print(f"Loading model: {self.config.model_id}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id, 
                    use_fast=True
                )
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                # Determine model dtype
                dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
                
                # Load model with potential quantization
                load_kwargs = {"torch_dtype": dtype}
                
                if self.config.use_8bit:
                    try:
                        import bitsandbytes as bnb
                        load_kwargs["load_in_8bit"] = True
                        load_kwargs["device_map"] = "auto"
                        print("Using 8-bit quantization")
                    except ImportError:
                        print("bitsandbytes not available, using standard precision")
                        
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
                
                # Create generation pipeline
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False
                )
                
                # Load embedding model for similarity tasks
                try:
                    embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
                    self.embedding_model = AutoModel.from_pretrained(embed_model_id)
                    self.embedding_model.eval()
                    if self.device != "cpu":
                        self.embedding_model.to(self.device)
                    print(f"Loaded embedding model: {embed_model_id}")
                except Exception as e:
                    print(f"Could not load embedding model: {e}")
                    
                # Set generation config
                self.generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                self.is_loaded = True
                global_metrics.set_gauge("model_loaded", 1.0)
                print("Model loaded successfully")
                return True
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            global_metrics.set_gauge("model_loaded", 0.0)
            return False
            
    async def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_pmll_cache: bool = True
    ) -> str:
        """
        Generate text using PMLL-optimized inference
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_pmll_cache: Use PMLL memory caching
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            await self.load_model_async()
            
        def generate_operation():
            return self._generate_text_sync(
                prompt, max_new_tokens, temperature, top_p, use_pmll_cache
            )
            
        promise = self.promise_queue.enqueue(generate_operation)
        return await self._promise_to_task(promise)
        
    def _generate_text_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_pmll_cache: bool
    ) -> str:
        """Synchronous text generation"""
        try:
            with global_profiler.time_operation("text_generation"):
                # Check PMLL cache first
                cache_key = f"gen_{hash(prompt + str(max_new_tokens) + str(temperature))}"
                
                if use_pmll_cache and self.memory_pool:
                    cached_result = self.memory_pool.retrieve(cache_key)
                    if cached_result is not None:
                        global_metrics.increment_counter("cache_hits")
                        return cached_result
                        
                # Generate text
                result = self.generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=max(0.0, float(temperature)),
                    top_p=min(1.0, max(0.0, float(top_p))),
                    pad_token_id=self.tokenizer.eos_token_id,
                    generation_config=self.generation_config
                )
                
                generated_text = result[0]["generated_text"]
                
                # Cache result in PMLL memory
                if use_pmll_cache and self.memory_pool:
                    self.memory_pool.store(
                        generated_text,
                        tensor_id=cache_key,
                        metadata={
                            "prompt": prompt,
                            "max_tokens": max_new_tokens,
                            "temperature": temperature,
                            "timestamp": time.time()
                        }
                    )
                    
                # Update metrics
                self.inference_count += 1
                self.total_tokens_generated += len(self.tokenizer.encode(generated_text))
                global_metrics.increment_counter("text_generations")
                
                return generated_text
                
        except Exception as e:
            print(f"Text generation failed: {e}")
            return f"Error: {str(e)}"
            
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Generated text chunks
        """
        if not self.is_loaded:
            await self.load_model_async()
            
        # For now, simulate streaming by yielding chunks
        # Real streaming would require more complex integration
        full_text = await self.generate_text(
            prompt, max_new_tokens, temperature, use_pmll_cache=False
        )
        
        # Yield text in chunks
        chunk_size = max(1, len(full_text) // 20)  # 20 chunks
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.01)  # Small delay to simulate streaming
            
    async def embed_text(self, text: str, use_pmll_cache: bool = True) -> List[float]:
        """
        Generate embeddings for text
        
        Args:
            text: Input text
            use_pmll_cache: Use PMLL memory caching
            
        Returns:
            Text embeddings
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")
            
        def embed_operation():
            return self._embed_text_sync(text, use_pmll_cache)
            
        promise = self.promise_queue.enqueue(embed_operation)
        return await self._promise_to_task(promise)
        
    def _embed_text_sync(self, text: str, use_pmll_cache: bool) -> List[float]:
        """Synchronous text embedding"""
        try:
            cache_key = f"embed_{hash(text)}"
            
            # Check cache
            if use_pmll_cache and self.memory_pool:
                cached_result = self.memory_pool.retrieve(cache_key)
                if cached_result is not None:
                    return cached_result
                    
            # Generate embedding
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                )
                
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                # Get embeddings
                outputs = self.embedding_model(**inputs)
                
                # Mean pooling
                embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                
                mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                masked_embeddings = embeddings * mask
                summed = masked_embeddings.sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = summed / counts
                
                # Convert to list
                embedding_list = mean_embeddings[0].cpu().float().numpy().tolist()
                
                # Cache result
                if use_pmll_cache and self.memory_pool:
                    self.memory_pool.store(
                        embedding_list,
                        tensor_id=cache_key,
                        metadata={"text": text, "timestamp": time.time()}
                    )
                    
                return embedding_list
                
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return []
            
    async def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage with PMLL solver
        
        Returns:
            Optimization results
        """
        if not self.memory_pool or not self.sat_solver:
            return {"error": "PMLL components not initialized"}
            
        def optimize_operation():
            # Create optimization problem
            # Variables represent memory slots to compress/evict
            num_slots = len(self.memory_pool.memory_slots)
            
            if num_slots == 0:
                return {"optimized_slots": 0, "memory_saved": 0}
                
            # Simple constraint: compress at least half the slots
            clauses = []
            
            # At least compress 50% of slots
            positive_literals = list(range(1, min(num_slots + 1, 100)))  # Limit problem size
            clauses.append(positive_literals[:len(positive_literals)//2])
            
            # Add clauses to SAT solver
            self.sat_solver = PMLLSATSolver(len(positive_literals), enable_ouroboros=True)
            for clause in clauses:
                self.sat_solver.add_clause(clause)
                
            # Solve optimization problem
            is_satisfiable = self.sat_solver.solve()
            
            if is_satisfiable:
                solution = self.sat_solver.get_solution()
                optimized_count = sum(1 for v in solution.values() if v)
                
                return {
                    "optimized_slots": optimized_count,
                    "total_slots": len(positive_literals),
                    "memory_saved_estimated": optimized_count * 0.5,  # Estimate 50% savings
                    "sat_solver_stats": self.sat_solver.get_stats()
                }
            else:
                return {"error": "No optimization solution found"}
                
        promise = self.promise_queue.enqueue(optimize_operation)
        return await self._promise_to_task(promise)
        
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        stats = {
            "model_config": {
                "model_id": self.config.model_id,
                "device": self.device,
                "is_loaded": self.is_loaded,
                "use_8bit": self.config.use_8bit,
                "use_pmll_memory": self.config.use_pmll_memory
            },
            "performance": {
                "inference_count": self.inference_count,
                "total_tokens_generated": self.total_tokens_generated,
                "avg_tokens_per_inference": (
                    self.total_tokens_generated / max(1, self.inference_count)
                )
            },
            "memory_pool": None,
            "promise_queue": self.promise_queue.get_stats()
        }
        
        if self.memory_pool:
            stats["memory_pool"] = self.memory_pool.get_stats()
            
        return stats
        
    async def _promise_to_task(self, promise: Promise) -> Any:
        """Convert promise to asyncio task"""
        # Simple polling approach - could be improved with proper async integration
        while promise.is_pending():
            await asyncio.sleep(0.01)
            
        if promise.is_fulfilled():
            return promise.value
        elif promise.is_rejected():
            raise promise.error
        else:
            raise RuntimeError("Promise in unexpected state")


class ModelManager:
    """
    Manager for multiple AI models with PMLL optimization
    """
    
    def __init__(self):
        self.models: Dict[str, TransformerEngine] = {}
        self.default_model: Optional[str] = None
        
    async def load_model(self, model_id: str, config: Optional[ModelConfig] = None) -> TransformerEngine:
        """Load a model with given configuration"""
        if config is None:
            config = ModelConfig(
                model_id=model_id,
                model_size=ModelSize.SMALL,
                use_pmll_memory=True
            )
            
        engine = TransformerEngine(config)
        await engine.load_model_async()
        
        self.models[model_id] = engine
        
        if self.default_model is None:
            self.default_model = model_id
            
        return engine
        
    def get_model(self, model_id: Optional[str] = None) -> Optional[TransformerEngine]:
        """Get loaded model"""
        if model_id is None:
            model_id = self.default_model
            
        return self.models.get(model_id)
        
    async def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using specified or default model"""
        model = self.get_model(model_id)
        if model is None:
            raise RuntimeError(f"Model not loaded: {model_id}")
            
        return await model.generate_text(prompt, **kwargs)
        
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all loaded models"""
        return {
            model_id: model.get_model_stats() 
            for model_id, model in self.models.items()
        }


# Global model manager instance
model_manager = ModelManager()