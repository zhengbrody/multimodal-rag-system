n#!/bin/bash
# Quick restart script

echo "🔄 Restarting Personal RAG System..."
echo "======================================"

cd "$(dirname "$0")"

# Stop all processes
echo "🛑 Stopping old processes..."
pkill -f uvicorn 2>/dev/null
pkill -f streamlit 2>/dev/null
sleep 2

# Check and fix pyarrow (if needed)
echo "📦 Checking dependencies..."
if python3 -c "from pyarrow.lib import is_threading_enabled" 2>/dev/null; then
    echo "✅ PyArrow OK"
else
    echo "⚠️  Upgrading PyArrow..."
    pip install --upgrade pyarrow 2>/dev/null || pip install pyarrow==14.0.0
fi

# Start system
echo "🚀 Starting system..."
python3 run.py

