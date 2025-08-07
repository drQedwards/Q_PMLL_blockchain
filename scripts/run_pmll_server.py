#!/usr/bin/env python3
"""
PMLL Server Runner Script

Quick script to start the PMLL API server with optimal settings.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pmll.cli.main import start_server
from pmll.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run PMLL API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print("🚀 Starting PMLL AI Model Server")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔧 Workers: {args.workers}")
    print(f"🔄 Reload: {args.reload}")
    
    start_server(args)


if __name__ == "__main__":
    main()