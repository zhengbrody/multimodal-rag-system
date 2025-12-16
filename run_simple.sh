#!/bin/bash
# Simplified run script - one-click start for Personal RAG system

echo "🚀 Personal RAG System - Quick Start"
echo "======================================"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "📋 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements_simple.txt
fi

# Check .env
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ Please edit .env file if needed (USE_MOCK=true for no API costs)"
fi

# Check index
if [ ! -f "data/processed/mock_retriever.pkl" ] && [ ! -f "data/processed/retriever.pkl" ]; then
    echo "🔨 Building knowledge base index..."
    python3 setup.py
fi

# Start system
echo ""
echo "✅ Starting Personal RAG System..."
echo "======================================"
echo "📱 Frontend: http://localhost:8501"
echo "🔧 API Docs: http://localhost:8000/docs"
echo "💡 Press Ctrl+C to stop"
echo ""

python3 run.py

