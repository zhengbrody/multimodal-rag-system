#!/bin/bash
# Quick fix script for common terminal issues

echo "🔧 Quick Fix for Terminal Issues"
echo "================================"

cd "$(dirname "$0")"

# 1. Stop hanging processes
echo "1️⃣  Stopping old processes..."
pkill -f uvicorn 2>/dev/null
pkill -f streamlit 2>/dev/null
sleep 2

# 2. Check if index needs rebuilding
echo "2️⃣  Checking knowledge base index..."
KB_TIME=$(stat -f %m data/raw/knowledge_base.json 2>/dev/null || stat -c %Y data/raw/knowledge_base.json 2>/dev/null || echo 0)
INDEX_TIME=$(stat -f %m data/processed/mock_retriever.pkl 2>/dev/null || stat -c %Y data/processed/mock_retriever.pkl 2>/dev/null || echo 0)

if [ "$KB_TIME" -gt "$INDEX_TIME" ]; then
    echo "   ⚠️  Index outdated - rebuilding..."
    python3 setup.py
else
    echo "   ✅ Index is up to date"
fi

# 3. Check dependencies
echo "3️⃣  Checking dependencies..."
if ! python3 -c "import fastapi, uvicorn, streamlit" 2>/dev/null; then
    echo "   ⚠️  Installing dependencies..."
    pip3 install -r requirements_simple.txt
else
    echo "   ✅ Dependencies OK"
fi

# 4. Check .env file
echo "4️⃣  Checking configuration..."
if [ ! -f ".env" ]; then
    echo "   ⚠️  Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "USE_MOCK=true" > .env
    echo "   ✅ Created .env file"
else
    echo "   ✅ .env file exists"
fi

# 5. Validate JSON
echo "5️⃣  Validating knowledge base..."
if python3 -c "import json; json.load(open('data/raw/knowledge_base.json'))" 2>/dev/null; then
    echo "   ✅ JSON is valid"
else
    echo "   ❌ JSON has errors - please check data/raw/knowledge_base.json"
    exit 1
fi

echo ""
echo "✅ Quick fix complete!"
echo ""
echo "Next steps:"
echo "  - Start system: python3 run.py"
echo "  - Or use: ./restart.sh"
echo ""

