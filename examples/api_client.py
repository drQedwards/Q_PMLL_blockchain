#!/usr/bin/env python3
"""
PMLL API Client Example

Demonstrates how to interact with the PMLL API server.
"""

import requests
import json
import time
from typing import Dict, Any


class PMLLClient:
    """Client for PMLL API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API server health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def load_model(
        self, 
        model_id: str, 
        device: str = "cpu",
        use_pmll_memory: bool = True,
        compression_level: str = "balanced"
    ) -> Dict[str, Any]:
        """Load an AI model"""
        data = {
            "model_id": model_id,
            "device": device,
            "use_pmll_memory": use_pmll_memory,
            "compression_level": compression_level
        }
        
        response = self.session.post(f"{self.base_url}/models/load", json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_text(
        self,
        prompt: str,
        model_id: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        use_pmll_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate text using AI model"""
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "use_pmll_cache": use_pmll_cache
        }
        
        if model_id:
            data["model_id"] = model_id
        
        response = self.session.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        return response.json()
    
    def embed_text(self, text: str, use_pmll_cache: bool = True) -> Dict[str, Any]:
        """Generate embeddings for text"""
        data = {
            "text": text,
            "use_pmll_cache": use_pmll_cache
        }
        
        response = self.session.post(f"{self.base_url}/embed", json=data)
        response.raise_for_status()
        return response.json()
    
    def solve_sat(
        self,
        num_variables: int,
        clauses: list,
        enable_ouroboros: bool = True
    ) -> Dict[str, Any]:
        """Solve SAT problem"""
        data = {
            "num_variables": num_variables,
            "clauses": clauses,
            "enable_ouroboros": enable_ouroboros
        }
        
        response = self.session.post(f"{self.base_url}/sat/solve", json=data)
        response.raise_for_status()
        return response.json()
    
    def optimize_memory(self, cleanup_expired: bool = True) -> Dict[str, Any]:
        """Optimize memory usage"""
        data = {"cleanup_expired": cleanup_expired}
        
        response = self.session.post(f"{self.base_url}/memory/optimize", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate PMLL API client usage"""
    print("🌐 PMLL API Client Example")
    print("=" * 50)
    
    # Initialize client
    client = PMLLClient()
    
    try:
        # 1. Health Check
        print("\n1. Health Check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        
        # 2. Load Model
        print("\n2. Loading GPT-2 model...")
        load_result = client.load_model(
            model_id="gpt2",
            device="cpu",
            use_pmll_memory=True,
            compression_level="balanced"
        )
        print(f"   Status: {load_result['status']}")
        
        # Wait a moment for model to load
        time.sleep(2)
        
        # 3. Generate Text
        print("\n3. Generating Text...")
        generation_result = client.generate_text(
            prompt="The PMLL algorithm demonstrates that",
            max_new_tokens=50,
            temperature=0.7
        )
        
        print(f"   Generated: {generation_result['generated_text']}")
        print(f"   Inference time: {generation_result['inference_time']:.3f}s")
        
        # 4. Test SAT Solver
        print("\n4. Testing SAT Solver...")
        sat_result = client.solve_sat(
            num_variables=3,
            clauses=[[1, -2, 3], [-1, 2, -3]],
            enable_ouroboros=True
        )
        
        print(f"   Satisfiable: {sat_result['satisfiable']}")
        print(f"   Solve time: {sat_result['solve_time']:.3f}s")
        if sat_result['satisfiable']:
            print(f"   Solution: {sat_result['solution']}")
        
        # 5. Memory Optimization
        print("\n5. Memory Optimization...")
        optimize_result = client.optimize_memory(cleanup_expired=True)
        print(f"   Optimizations applied: {len(optimize_result['optimizations'])}")
        
        # 6. System Metrics
        print("\n6. System Metrics...")
        metrics = client.get_metrics()
        global_metrics = metrics['global_metrics']
        print(f"   Cache hit rate: {global_metrics['cache_statistics']['hit_rate']:.2%}")
        print(f"   Total operations: {sum(global_metrics['counters'].values())}")
        
        print("\n" + "=" * 50)
        print("🎉 API Client Example Complete!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to PMLL API server")
        print("   Start the server with: python scripts/run_pmll_server.py")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")


if __name__ == "__main__":
    main()