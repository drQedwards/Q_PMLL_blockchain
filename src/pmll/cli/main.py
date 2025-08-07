#!/usr/bin/env python3
"""
PMLL Command Line Interface

Comprehensive CLI for the PMLL system, combining concepts from the original pypm
CLI structure with PMLL-specific functionality.
"""

import sys
import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
import requests

from ..utils.logging import setup_logging
from ..utils.config import load_config, create_default_config
from ..api.app import create_app
from ..integrations.transformer_engine import ModelManager, ModelConfig, ModelSize
from ..core.memory_pool import PMLLMemoryPool, CompressionLevel
from ..core.pmll_solver import PMLLSATSolver
from .. import __version__


def start_server(args: argparse.Namespace) -> None:
    """Start the PMLL API server"""
    config = load_config()
    
    # Override config with command line arguments
    host = args.host or config.api.host
    port = args.port or config.api.port
    workers = args.workers or config.api.workers
    reload = args.reload if hasattr(args, 'reload') else config.api.reload
    log_level = args.log_level or config.monitoring.log_level.lower()
    
    # Setup logging
    setup_logging(level=log_level.upper())
    
    logging.info(f"Starting PMLL server on {host}:{port}")
    logging.info(f"Workers: {workers}, Reload: {reload}")
    
    # Create FastAPI app
    app = create_app()
    
    # Uvicorn configuration
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "log_level": log_level,
        "reload": reload,
        "workers": workers if not reload else 1,
    }
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server failed to start: {e}")
        sys.exit(1)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run PMLL benchmarks"""
    from ..benchmarks.suite import PMLLBenchmarkSuite
    
    logging.info("Starting PMLL benchmark suite")
    
    suite = PMLLBenchmarkSuite(
        output_dir=args.output_dir,
        iterations=args.iterations,
        pool_sizes=args.pool_sizes,
        model_ids=getattr(args, 'model_ids', ['gpt2'])
    )
    
    try:
        results = asyncio.run(suite.run_all_benchmarks())
        
        if args.report:
            suite.generate_report(results)
            logging.info(f"Benchmark report generated in {args.output_dir}")
        
        logging.info("Benchmark suite completed successfully")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


def migrate_database(args: argparse.Namespace) -> None:
    """Run database migrations"""
    from ..storage.migrations import MigrationRunner
    
    logging.info(f"Running database migrations: {args.direction}")
    
    try:
        runner = MigrationRunner(database_url=args.database_url)
        asyncio.run(runner.run_migrations(direction=args.direction))
        logging.info("Database migrations completed")
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        sys.exit(1)


def health_check(args: argparse.Namespace) -> None:
    """Perform health check"""
    url = f"http://{args.host}:{args.port}/health/detailed"
    
    try:
        response = requests.get(url, timeout=args.timeout)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Pretty print health data
            import json
            print(json.dumps(health_data, indent=2))
            
            if health_data.get("status") == "healthy":
                logging.info("Service is healthy")
                sys.exit(0)
            else:
                logging.error("Service is not healthy")
                sys.exit(1)
        else:
            logging.error(f"Health check failed: HTTP {response.status_code}")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Health check error: {e}")
        sys.exit(1)


def solve_sat(args: argparse.Namespace) -> None:
    """Solve SAT problem using PMLL algorithm"""
    logging.info("Solving SAT problem with PMLL algorithm")
    
    try:
        # Read SAT problem from file if provided
        if args.input_file:
            clauses = read_sat_file(args.input_file)
            num_vars = max(abs(lit) for clause in clauses for lit in clause)
        else:
            # Use command line clauses
            clauses = args.clauses
            num_vars = args.num_variables
        
        # Create SAT solver
        solver = PMLLSATSolver(
            num_vars=num_vars,
            enable_ouroboros=args.ouroboros,
            max_depth=args.max_depth
        )
        
        # Add clauses
        for clause in clauses:
            solver.add_clause(clause)
        
        # Solve
        import time
        start_time = time.time()
        is_satisfiable = solver.solve()
        solve_time = time.time() - start_time
        
        # Output results
        print(f"Problem: {num_vars} variables, {len(clauses)} clauses")
        print(f"Result: {'SATISFIABLE' if is_satisfiable else 'UNSATISFIABLE'}")
        print(f"Solve time: {solve_time:.3f} seconds")
        
        if is_satisfiable:
            solution = solver.get_solution()
            if args.show_solution:
                print("Solution:")
                for var, value in sorted(solution.items()):
                    print(f"  x{var} = {value}")
        
        # Show solver statistics
        if args.verbose:
            stats = solver.get_stats()
            print("\nSolver Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Save results to file if requested
        if args.output_file:
            results = {
                "satisfiable": is_satisfiable,
                "solution": solver.get_solution() if is_satisfiable else None,
                "solve_time": solve_time,
                "stats": solver.get_stats()
            }
            
            with open(args.output_file, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logging.error(f"SAT solving failed: {e}")
        sys.exit(1)


def load_model(args: argparse.Namespace) -> None:
    """Load an AI model"""
    logging.info(f"Loading model: {args.model_id}")
    
    try:
        # Create model config
        compression_map = {
            "none": CompressionLevel.NONE,
            "low": CompressionLevel.LOW,
            "balanced": CompressionLevel.BALANCED,
            "aggressive": CompressionLevel.AGGRESSIVE,
            "max": CompressionLevel.MAX
        }
        
        config = ModelConfig(
            model_id=args.model_id,
            model_size=ModelSize(args.model_size),
            device=args.device,
            use_8bit=args.use_8bit,
            use_pmll_memory=args.use_pmll,
            max_memory_mb=args.max_memory,
            compression_level=compression_map[args.compression]
        )
        
        # Create model manager and load
        manager = ModelManager()
        engine = asyncio.run(manager.load_model(args.model_id, config))
        
        print(f"Successfully loaded model: {args.model_id}")
        
        # Show model stats
        if args.verbose:
            stats = engine.get_model_stats()
            print("\nModel Statistics:")
            import json
            print(json.dumps(stats, indent=2))
        
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        sys.exit(1)


def read_sat_file(file_path: str) -> List[List[int]]:
    """Read SAT problem from DIMACS format file"""
    clauses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and problem line
            if line.startswith('c') or line.startswith('p'):
                continue
            
            # Parse clause
            if line:
                literals = [int(x) for x in line.split() if x != '0']
                if literals:
                    clauses.append(literals)
    
    return clauses


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        prog="pmll",
        description="PMLL - Persistent Memory Logic Loop CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"pmll {__version__}"
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Set logging level"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", help="Host to bind to")
    server_parser.add_argument("--port", type=int, help="Port to bind to")
    server_parser.add_argument("--workers", type=int, help="Number of workers")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    server_parser.set_defaults(func=start_server)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--output-dir", default="./benchmark_results", 
                                help="Output directory for results")
    benchmark_parser.add_argument("--iterations", type=int, default=3,
                                help="Number of benchmark iterations")
    benchmark_parser.add_argument("--pool-sizes", type=int, nargs="+", 
                                default=[512, 1024, 2048],
                                help="Pool sizes to benchmark")
    benchmark_parser.add_argument("--model-ids", nargs="+", default=["gpt2"],
                                help="Model IDs to benchmark")
    benchmark_parser.add_argument("--report", action="store_true",
                                help="Generate HTML report")
    benchmark_parser.set_defaults(func=run_benchmark)
    
    # Migration command
    migrate_parser = subparsers.add_parser("migrate", help="Database migrations")
    migrate_parser.add_argument("--database-url", required=True,
                              help="Database connection URL")
    migrate_parser.add_argument("--direction", choices=["up", "down"], default="up",
                              help="Migration direction")
    migrate_parser.set_defaults(func=migrate_database)
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Health check")
    health_parser.add_argument("--host", default="localhost", help="Server host")
    health_parser.add_argument("--port", type=int, default=8001, help="Server port")
    health_parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    health_parser.set_defaults(func=health_check)
    
    # SAT solver command
    sat_parser = subparsers.add_parser("sat", help="Solve SAT problem")
    sat_parser.add_argument("--num-variables", type=int, help="Number of variables")
    sat_parser.add_argument("--clauses", nargs="*", help="Clauses as lists")
    sat_parser.add_argument("--input-file", help="Input DIMACS file")
    sat_parser.add_argument("--output-file", help="Output results file")
    sat_parser.add_argument("--ouroboros", action="store_true", help="Enable Ouroboros caching")
    sat_parser.add_argument("--max-depth", type=int, help="Maximum recursion depth")
    sat_parser.add_argument("--show-solution", action="store_true", help="Show solution")
    sat_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    sat_parser.set_defaults(func=solve_sat)
    
    # Model loading command
    model_parser = subparsers.add_parser("load-model", help="Load AI model")
    model_parser.add_argument("model_id", help="Model ID to load")
    model_parser.add_argument("--device", default="auto", help="Device (cpu/cuda/mps/auto)")
    model_parser.add_argument("--model-size", default="small", 
                            choices=["small", "medium", "large", "xlarge"])
    model_parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    model_parser.add_argument("--use-pmll", action="store_true", default=True,
                            help="Use PMLL memory optimization")
    model_parser.add_argument("--max-memory", type=int, default=4096, help="Max memory MB")
    model_parser.add_argument("--compression", default="balanced",
                            choices=["none", "low", "balanced", "aggressive", "max"])
    model_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    model_parser.set_defaults(func=load_model)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--create-default", action="store_true", 
                             help="Create default configuration file")
    config_parser.set_defaults(func=lambda args: create_default_config() if args.create_default else None)
    
    return parser


def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level.upper())
    
    if not hasattr(args, 'func') or args.func is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()