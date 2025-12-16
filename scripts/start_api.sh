#!/bin/bash
# Start API server

echo "🚀 Starting Personal RAG API Server..."
echo "======================================"

cd "$(dirname "$0")"

# Activate virtual environment (if exists)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check dependencies
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements_simple.txt
fi

# Start API
echo "🔧 Starting API on http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "💡 Press Ctrl+C to stop"
echo ""

uvicorn src.api.personal_api:app --reload --host 0.0.0.0 --port 8000

