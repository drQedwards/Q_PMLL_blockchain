#!/usr/bin/env python3
"""
Basic PMLL Usage Example

Demonstrates how to use the PMLL system for AI model inference.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pmll.integrations.transformer_engine import ModelManager, ModelConfig, ModelSize
from pmll.core.memory_pool import PMLLMemoryPool, CompressionLevel
from pmll.core.pmll_solver import PMLLSATSolver


async def main():
    print("🧠 PMLL Basic Usage Example")
    print("=" * 50)
    
    # 1. Create PMLL Memory Pool
    print("\n1. Initializing PMLL Memory Pool...")
    memory_pool = PMLLMemoryPool(
        dim=768,
        slots=4096,
        compression_level=CompressionLevel.BALANCED,
        enable_ouroboros=True
    )
    
    print(f"   Memory pool created with {memory_pool.slots} slots")
    print(f"   Ouroboros caching: {memory_pool.enable_ouroboros}")
    
    # 2. Test SAT Solver (P = NP proof implementation)
    print("\n2. Testing PMLL SAT Solver...")
    sat_solver = PMLLSATSolver(
        num_vars=3,
        enable_ouroboros=True
    )
    
    # Add simple SAT problem: (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3)
    sat_solver.add_clause([1, -2, 3])  # x1 or not x2 or x3
    sat_solver.add_clause([-1, 2, -3])  # not x1 or x2 or not x3
    
    is_satisfiable = sat_solver.solve()
    print(f"   SAT Problem satisfiable: {is_satisfiable}")
    
    if is_satisfiable:
        solution = sat_solver.get_solution()
        print(f"   Solution: {solution}")
        
    stats = sat_solver.get_stats()
    print(f"   Solver steps: {stats['refinement_steps']}/{stats['max_possible_steps']}")
    
    # 3. Load AI Model (if transformers available)
    try:
        print("\n3. Loading AI Model...")
        model_manager = ModelManager()
        
        config = ModelConfig(
            model_id="gpt2",
            model_size=ModelSize.SMALL,
            use_pmll_memory=True,
            device="cpu"  # Use CPU for compatibility
        )
        
        model = await model_manager.load_model("gpt2", config)
        print("   ✅ GPT-2 model loaded successfully")
        
        # 4. Generate Text
        print("\n4. Generating Text...")
        prompt = "The future of AI is"
        generated = await model.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        
        print(f"   Prompt: {prompt}")
        print(f"   Generated: {generated}")
        
        # 5. Test Memory Optimization
        print("\n5. Testing Memory Optimization...")
        optimization_result = await model.optimize_memory()
        print(f"   Optimization result: {optimization_result}")
        
        # 6. Show Statistics
        print("\n6. System Statistics:")
        model_stats = model.get_model_stats()
        print(f"   Model inferences: {model_stats['performance']['inference_count']}")
        print(f"   Memory pool usage: {model_stats['memory_pool']['usage']['utilization']:.2%}")
        
    except ImportError:
        print("\n3. ⚠️  Transformers not available - install with: pip install transformers torch")
        print("   Skipping AI model demonstration")
    
    except Exception as e:
        print(f"\n3. ❌ Error loading model: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 PMLL Basic Usage Example Complete!")
    print("\nNext steps:")
    print("- Try the API server: python scripts/run_pmll_server.py")
    print("- Use the CLI: python -m pmll.cli.main --help")
    print("- Check out API docs at: http://localhost:8001/docs")


if __name__ == "__main__":
    asyncio.run(main())