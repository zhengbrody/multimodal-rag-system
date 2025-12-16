#!/usr/bin/env python3
"""
Run script for Personal RAG System

Starts both the API backend and Streamlit frontend
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path


def check_setup():
    """Verify basic setup is complete (env file)"""
    base_dir = Path(__file__).parent

    # Check .env
    env_path = base_dir / ".env"
    if not env_path.exists():
        print("Error: .env file not found")
        print("Please configure your environment variables")
        sys.exit(1)

    with open(env_path, 'r') as f:
        if "your_openai_api_key_here" in f.read():
            print("Error: OpenAI API key not configured")
            print("Please edit .env and add your API key")
            sys.exit(1)


def start_api():
    """Start the FastAPI backend"""
    print("Starting API server on http://localhost:8000...")
    api_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.personal_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return api_process


def start_frontend():
    """Start the Streamlit frontend"""
    print("Starting Streamlit frontend on http://localhost:8501...")
    frontend_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "frontend/personal_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return frontend_process


def monitor_processes(api_process, frontend_process):
    """Monitor and display output from both processes"""
    import select

    print("\n" + "=" * 50)
    print("Personal RAG System Running")
    print("=" * 50)
    print("API:      http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Frontend: http://localhost:8501")
    print("=" * 50)
    print("\nPress Ctrl+C to stop all services\n")

    try:
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                print("API server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("Frontend stopped unexpectedly")
                break

            # Read output from processes (non-blocking on Unix)
            if hasattr(select, 'select'):
                readable, _, _ = select.select(
                    [api_process.stdout, frontend_process.stdout],
                    [], [], 0.1
                )

                for stream in readable:
                    line = stream.readline()
                    if line:
                        if stream == api_process.stdout:
                            print(f"[API] {line.rstrip()}")
                        else:
                            print(f"[Frontend] {line.rstrip()}")
            else:
                # Windows fallback
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nShutting down services...")


def cleanup(api_process, frontend_process):
    """Clean up processes"""
    print("Stopping API server...")
    api_process.terminate()
    api_process.wait(timeout=5)

    print("Stopping Frontend...")
    frontend_process.terminate()
    frontend_process.wait(timeout=5)

    print("All services stopped")


def main():
    """Main entry point"""
    print("=" * 50)
    print("Personal RAG System Launcher")
    print("=" * 50)

    # Verify setup
    check_setup()
    print("✓ Setup verified\n")

    # Start services
    api_process = start_api()
    time.sleep(2)  # Give API time to start

    frontend_process = start_frontend()
    time.sleep(2)  # Give frontend time to start

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        cleanup(api_process, frontend_process)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Monitor processes
    try:
        monitor_processes(api_process, frontend_process)
    finally:
        cleanup(api_process, frontend_process)


if __name__ == "__main__":
    main()
