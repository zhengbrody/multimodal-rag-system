#!/usr/bin/env python3
"""
Setup script for Personal RAG System

This script:
1. Checks Python version
2. Installs dependencies
3. Verifies API key configuration
4. Builds the initial knowledge base index
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Ensure Python 3.9+ is being used"""
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print(f"Error: Python 3.9+ required. You have {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    requirements_file = Path(__file__).parent / "requirements_simple.txt"

    if not requirements_file.exists():
        print(f"Error: {requirements_file} not found")
        sys.exit(1)

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def check_env_file():
    """Check if .env file exists and has API key"""
    print("\nChecking environment configuration...")
    env_file = Path(__file__).parent / ".env"

    if not env_file.exists():
        print("Error: .env file not found")
        print("Please copy .env.example to .env and add your OpenAI API key")
        sys.exit(1)

    with open(env_file, 'r') as f:
        content = f.read()

    if "your_openai_api_key_here" in content or "OPENAI_API_KEY=" not in content:
        print("⚠️  Warning: OpenAI API key not configured in .env")
        print("Please edit .env and add your actual OpenAI API key")
        return False

    print("✓ Environment file configured")
    return True


def create_directories():
    """Ensure required directories exist"""
    print("\nCreating directories...")
    base_dir = Path(__file__).parent

    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "logs"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")


def check_knowledge_base():
    """Check if knowledge base file exists"""
    print("\nChecking knowledge base...")
    kb_file = Path(__file__).parent / "data" / "raw" / "knowledge_base.json"

    if not kb_file.exists():
        print("⚠️  Warning: knowledge_base.json not found")
        print(f"Please create your knowledge base at: {kb_file}")
        return False

    print(f"✓ Knowledge base found at {kb_file}")
    return True


def build_index():
    """Build the initial retriever index"""
    print("\nBuilding knowledge base index...")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from rag.knowledge_processor import build_knowledge_base
        from rag.retriever import PersonalRAGRetriever

        kb_path = Path(__file__).parent / "data" / "raw" / "knowledge_base.json"
        save_path = Path(__file__).parent / "data" / "processed" / "retriever.pkl"

        # Build knowledge base
        documents = build_knowledge_base(str(kb_path))

        # Initialize and populate retriever
        print("Generating embeddings (this may take a moment)...")
        retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")
        retriever.add_documents(documents)

        # Save retriever
        retriever.save(str(save_path))

        print(f"✓ Index built successfully")
        print(f"  - Documents: {len(retriever.documents)}")
        print(f"  - Categories: {retriever.get_category_stats()}")
        print(f"  - Saved to: {save_path}")

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure dependencies are installed correctly")
        sys.exit(1)
    except Exception as e:
        print(f"Error building index: {e}")
        if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
            print("Please ensure your OpenAI API key is correctly set in .env")
        sys.exit(1)


def main():
    """Main setup routine"""
    print("=" * 50)
    print("Personal RAG System Setup")
    print("=" * 50)

    # Step 1: Check Python version
    check_python_version()

    # Step 2: Create directories
    create_directories()

    # Step 3: Install dependencies
    install_dependencies()

    # Step 4: Check environment
    env_ok = check_env_file()

    # Step 5: Check knowledge base
    kb_ok = check_knowledge_base()

    # Step 6: Build index (if everything is ready)
    if env_ok and kb_ok:
        build_index()
    else:
        print("\n⚠️  Setup incomplete:")
        if not env_ok:
            print("  - Configure your OpenAI API key in .env")
        if not kb_ok:
            print("  - Add your knowledge_base.json file")
        print("\nRun this script again after fixing the above issues.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("=" * 50)
    print("\nTo start the system:")
    print("  1. Start API:      python -m uvicorn src.api.personal_api:app --reload")
    print("  2. Start Frontend: streamlit run frontend/personal_app.py")
    print("\nOr use the run script: python run.py")


if __name__ == "__main__":
    main()
