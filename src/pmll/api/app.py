"""
PMLL FastAPI Application

Complete web interface for the PMLL system with AI model inference endpoints.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from ..integrations.transformer_engine import ModelManager, ModelConfig, ModelSize
from ..core.memory_pool import PMLLMemoryPool, CompressionLevel
from ..core.promises import PromiseQueue
from ..core.pmll_solver import PMLLSATSolver
from ..core.metrics import global_metrics, global_profiler
from ..utils.config import load_config


# Pydantic models for API
class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., min_length=1, max_length=8192)
    model_id: Optional[str] = None
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    use_pmll_cache: bool = True
    stream: bool = False


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    model_id: str
    inference_time: float
    tokens_generated: int
    cached: bool = False


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings"""
    text: str = Field(..., min_length=1, max_length=8192)
    use_pmll_cache: bool = True


class EmbeddingResponse(BaseModel):
    """Response model for text embeddings"""
    embeddings: List[float]
    dimension: int
    inference_time: float
    cached: bool = False


class ModelLoadRequest(BaseModel):
    """Request model for loading models"""
    model_id: str
    device: str = "auto"
    use_8bit: bool = False
    use_pmll_memory: bool = True
    max_memory_mb: int = Field(4096, ge=512, le=32768)
    compression_level: str = "balanced"
    
    @validator('compression_level')
    def validate_compression_level(cls, v):
        valid_levels = ["none", "low", "balanced", "aggressive", "max"]
        if v not in valid_levels:
            raise ValueError(f"Compression level must be one of {valid_levels}")
        return v


class MemoryOptimizeRequest(BaseModel):
    """Request model for memory optimization"""
    target_compression: Optional[float] = Field(None, ge=0.1, le=0.9)
    cleanup_expired: bool = True


class SATSolveRequest(BaseModel):
    """Request model for SAT problem solving"""
    num_variables: int = Field(..., ge=1, le=10000)
    clauses: List[List[int]]
    enable_ouroboros: bool = True
    max_depth: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    uptime: float
    version: str
    models_loaded: int
    memory_usage: Dict[str, Any]
    performance_stats: Dict[str, Any]


# Global state
model_manager = ModelManager()
memory_pool = None
promise_queue = PromiseQueue.get_instance()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global memory_pool
    config = load_config()
    
    # Initialize PMLL memory pool
    memory_pool = PMLLMemoryPool(
        dim=config.memory.default_dim,
        slots=config.memory.default_slots,
        compression_level=getattr(CompressionLevel, config.memory.compression_threshold, CompressionLevel.BALANCED),
        device=config.memory.device,
        enable_ouroboros=True
    )
    
    print("PMLL API Server starting...")
    print(f"Memory pool initialized with {config.memory.default_slots} slots")
    
    yield
    
    # Shutdown
    print("PMLL API Server shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="PMLL API",
        description="Persistent Memory Logic Loop - AI Model Inference API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Routes
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Basic health check"""
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            uptime=time.time() - global_metrics.start_time,
            version="1.0.0",
            models_loaded=len(model_manager.models),
            memory_usage=memory_pool.get_stats() if memory_pool else {},
            performance_stats=global_metrics.get_summary()
        )
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with comprehensive metrics"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "uptime": time.time() - global_metrics.start_time,
                "version": "1.0.0",
                "memory_pool": memory_pool.get_stats() if memory_pool else None,
                "promise_queue": promise_queue.get_stats(),
                "models": model_manager.get_all_stats()
            },
            "performance": global_profiler.get_profile_summary(),
            "metrics": global_metrics.get_summary()
        }
    
    @app.post("/models/load")
    async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
        """Load an AI model"""
        try:
            # Convert compression level string to enum
            compression_map = {
                "none": CompressionLevel.NONE,
                "low": CompressionLevel.LOW,
                "balanced": CompressionLevel.BALANCED,
                "aggressive": CompressionLevel.AGGRESSIVE,
                "max": CompressionLevel.MAX
            }
            
            config = ModelConfig(
                model_id=request.model_id,
                model_size=ModelSize.SMALL,  # Could be determined automatically
                device=request.device,
                use_8bit=request.use_8bit,
                use_pmll_memory=request.use_pmll_memory,
                max_memory_mb=request.max_memory_mb,
                compression_level=compression_map[request.compression_level]
            )
            
            # Load model asynchronously
            engine = await model_manager.load_model(request.model_id, config)
            
            return {
                "status": "success",
                "model_id": request.model_id,
                "message": "Model loaded successfully",
                "stats": engine.get_model_stats()
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    @app.get("/models")
    async def list_models():
        """List all loaded models"""
        return {
            "models": list(model_manager.models.keys()),
            "default_model": model_manager.default_model,
            "stats": model_manager.get_all_stats()
        }
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate_text(request: GenerateRequest):
        """Generate text using AI model"""
        try:
            start_time = time.time()
            
            # Get model
            model = model_manager.get_model(request.model_id)
            if model is None:
                available_models = list(model_manager.models.keys())
                if not available_models:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No models loaded. Load a model first using /models/load"
                    )
                # Use first available model
                model = model_manager.get_model(available_models[0])
            
            # Generate text
            if request.stream:
                # For streaming, we'll return the full response here
                # Real streaming would need Server-Sent Events
                generated_text = await model.generate_text(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    use_pmll_cache=request.use_pmll_cache
                )
            else:
                generated_text = await model.generate_text(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    use_pmll_cache=request.use_pmll_cache
                )
            
            inference_time = time.time() - start_time
            
            # Estimate token count
            tokens_generated = len(generated_text.split()) * 1.3  # Rough estimate
            
            return GenerateResponse(
                generated_text=generated_text,
                model_id=request.model_id or model_manager.default_model or "unknown",
                inference_time=inference_time,
                tokens_generated=int(tokens_generated),
                cached=False  # Could check if result came from cache
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text generation failed: {str(e)}"
            )
    
    @app.post("/generate/stream")
    async def generate_text_stream(request: GenerateRequest):
        """Generate text with streaming response"""
        try:
            model = model_manager.get_model(request.model_id)
            if model is None:
                available_models = list(model_manager.models.keys())
                if not available_models:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No models loaded"
                    )
                model = model_manager.get_model(available_models[0])
            
            async def generate_stream():
                async for chunk in model.generate_stream(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature
                ):
                    yield f"data: {chunk}\n\n"
                    
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Streaming generation failed: {str(e)}"
            )
    
    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed_text(request: EmbeddingRequest):
        """Generate embeddings for text"""
        try:
            start_time = time.time()
            
            # Get any available model with embedding capability
            model = None
            for engine in model_manager.models.values():
                if engine.embedding_model is not None:
                    model = engine
                    break
                    
            if model is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No model with embedding capability loaded"
                )
            
            embeddings = await model.embed_text(
                text=request.text,
                use_pmll_cache=request.use_pmll_cache
            )
            
            inference_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=embeddings,
                dimension=len(embeddings),
                inference_time=inference_time,
                cached=False
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}"
            )
    
    @app.post("/memory/optimize")
    async def optimize_memory(request: MemoryOptimizeRequest):
        """Optimize memory usage with PMLL algorithms"""
        try:
            if not memory_pool:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Memory pool not initialized"
                )
            
            results = {"optimizations": []}
            
            # Cleanup expired entries if requested
            if request.cleanup_expired:
                expired_count = memory_pool.cleanup_expired()
                results["optimizations"].append({
                    "type": "cleanup_expired",
                    "items_removed": expired_count
                })
            
            # Use SAT solver for memory optimization if model available
            for model_id, model in model_manager.models.items():
                if hasattr(model, 'optimize_memory'):
                    optimization_result = await model.optimize_memory()
                    results["optimizations"].append({
                        "type": "sat_optimization",
                        "model_id": model_id,
                        "result": optimization_result
                    })
            
            # Get updated stats
            results["memory_stats"] = memory_pool.get_stats()
            results["timestamp"] = time.time()
            
            return results
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Memory optimization failed: {str(e)}"
            )
    
    @app.post("/sat/solve")
    async def solve_sat_problem(request: SATSolveRequest):
        """Solve SAT problem using PMLL algorithm"""
        try:
            # Create SAT solver
            solver = PMLLSATSolver(
                num_vars=request.num_variables,
                enable_ouroboros=request.enable_ouroboros,
                max_depth=request.max_depth
            )
            
            # Add clauses
            for clause in request.clauses:
                # Validate clause
                for literal in clause:
                    if abs(literal) > request.num_variables or literal == 0:
                        raise ValueError(f"Invalid literal {literal}")
                solver.add_clause(clause)
            
            # Solve
            start_time = time.time()
            is_satisfiable = solver.solve()
            solve_time = time.time() - start_time
            
            result = {
                "satisfiable": is_satisfiable,
                "solution": solver.get_solution() if is_satisfiable else None,
                "solver_stats": solver.get_stats(),
                "solve_time": solve_time,
                "polynomial_bound": solver._calculate_phi_n()
            }
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAT solving failed: {str(e)}"
            )
    
    @app.get("/metrics")
    async def get_metrics():
        """Get comprehensive system metrics"""
        return {
            "timestamp": time.time(),
            "global_metrics": global_metrics.get_summary(),
            "profiler_stats": global_profiler.get_profile_summary(),
            "memory_pool": memory_pool.get_stats() if memory_pool else None,
            "promise_queue": promise_queue.get_stats(),
            "models": model_manager.get_all_stats()
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "PMLL API",
            "description": "Persistent Memory Logic Loop - AI Model Inference API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "models": "/models",
                "generate": "/generate",
                "embed": "/embed",
                "sat_solve": "/sat/solve",
                "memory_optimize": "/memory/optimize",
                "metrics": "/metrics",
                "docs": "/docs"
            },
            "status": "operational"
        }
    
    return app


# For running directly
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Use 8001 to match supervisor configuration
        log_level="info"
    )