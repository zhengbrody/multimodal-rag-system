#!/bin/bash
# Script to run the FastAPI backend

echo "Starting Multimodal RAG API..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one from .env.example"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '#' | xargs)

# Run the API
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
