#!/bin/bash
# Rebuild knowledge base index script

echo "🔨 Rebuilding Knowledge Base Index"
echo "===================================="

cd "$(dirname "$0")"

# Build knowledge base and save to mock retriever
python3 << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from rag.knowledge_processor import build_knowledge_base
from rag.mock_retriever import MockRetriever

# Build knowledge base
kb_path = Path('data/raw/knowledge_base.json')
print("📖 Loading knowledge base...")
documents = build_knowledge_base(str(kb_path))
print(f"✅ Built {len(documents)} documents")

# Create and populate mock retriever
print("🔧 Creating index...")
retriever = MockRetriever()
retriever.add_documents(documents)

# Save retriever
save_path = Path('data/processed/mock_retriever.pkl')
retriever.save(str(save_path))

print(f"✅ Index saved successfully!")
print(f"   Location: {save_path}")
print(f"   Documents: {len(retriever.documents)}")
print(f"   File size: {save_path.stat().st_size / 1024:.1f} KB")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Index rebuilt successfully!"
    echo ""
    echo "⚠️  Note: You need to restart the API to load the new index:"
    echo "   pkill -f uvicorn"
    echo "   python3 run.py"
    echo ""
    echo "Or use: ./restart.sh"
else
    echo ""
    echo "❌ Failed to rebuild index"
    exit 1
fi

