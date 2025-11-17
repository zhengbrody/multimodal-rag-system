#!/bin/bash
# Script to run the Streamlit frontend

echo "Starting Multimodal RAG Frontend..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Run Streamlit
streamlit run frontend/app.py --server.port 8501
