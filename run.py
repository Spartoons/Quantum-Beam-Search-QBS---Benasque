#!/usr/bin/env python3
"""
Quantum Benasque Routing - Entry Point

This script starts the Flask web application for the quantum-classical
hybrid hiking route optimizer.

Usage:
    python run.py              # Start on default port 5000
    python run.py --port 8080  # Start on custom port
    python run.py --debug      # Start in debug mode
"""

import argparse
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import app, initialize_graph


def main():
    parser = argparse.ArgumentParser(
        description='Quantum Benasque Routing - Hiking Route Optimizer'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )
    parser.add_argument(
        '--host', '-H',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--no-reload',
        action='store_true',
        help='Disable auto-reloader (useful for some IDEs)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏔️  Quantum Benasque Routing")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 60)
    
    # Initialize the graph before starting
    print("\n[*] Initializing graph data...")
    initialize_graph()
    
    print("\n[*] Starting Flask server...")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug and not args.no_reload
        )
    except KeyboardInterrupt:
        print("\n\n[*] Server stopped. Goodbye!")
        sys.exit(0)


if __name__ == '__main__':
    main()
