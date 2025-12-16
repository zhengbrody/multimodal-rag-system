# 🚀 Quick Start Guide

## 📋 Prerequisites

- Python 3.11 or 3.12
- pip package manager
- (Optional) OpenAI API Key (if using OpenAI mode)

## 🎯 Method 1: One-Click Run (Recommended)

### Step 1: Navigate to Project (if not already)

```bash
cd /Users/zhengdong/multimodal-rag-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install lightweight dependencies (recommended, uses Mock mode, no API Key needed)
pip install -r requirements_simple.txt

# Or install full dependencies (requires OpenAI API Key)
# pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy environment variable template
cp .env.example .env

# Edit .env file
# If using Mock mode (recommended, no API Key), keep USE_MOCK=true
# If using OpenAI mode, set:
# USE_MOCK=false
# OPENAI_API_KEY=sk-your-actual-key-here
```

**Mock Mode Configuration (Recommended for beginners):**
```bash
# .env file content
USE_MOCK=true
# OPENAI_API_KEY can be empty or placeholder
```

**OpenAI Mode Configuration:**
```bash
# .env file content
USE_MOCK=false
OPENAI_API_KEY=sk-your-actual-key-here
LLM_MODEL=gpt-3.5-turbo
```

### Step 5: Prepare Knowledge Base

The knowledge base file already exists: `data/raw/knowledge_base.json`

If you need to modify personal information, edit this file.

### Step 6: Build Index

```bash
# Run setup.py to build knowledge base index
python setup.py
```

This will:
- Check Python version
- Install dependencies (if not already installed)
- Verify environment configuration
- Build knowledge base index (generates `data/processed/retriever.pkl` or `mock_retriever.pkl`)

### Step 7: Run System

```bash
# One-click start (starts both API and frontend)
python run.py
```

Or start separately:

**Terminal 1 - Start API:**
```bash
uvicorn src.api.personal_api:app --reload --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend/personal_app.py
```

### Step 8: Access Application

- **Frontend Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎯 Method 2: Using Docker (Recommended for Production)

### Step 1: Build and Run

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Step 2: Access Application

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000

### Step 3: Stop Services

```bash
docker-compose down
```

## 🔧 Common Issues

### Issue 1: "Knowledge base index not found"

**Solution:**
```bash
# Run setup.py to build index
python setup.py
```

### Issue 2: ".env file not found"

**Solution:**
```bash
# Create .env file
cp .env.example .env
# Then edit .env file
```

### Issue 3: "OpenAI API key not configured"

**Solution:**

If using Mock mode (recommended):
```bash
# Set in .env
USE_MOCK=true
```

If using OpenAI mode:
```bash
# Add your API Key in .env
OPENAI_API_KEY=sk-your-actual-key-here
USE_MOCK=false
```

### Issue 4: Port Already in Use

**Solution:**

Change ports:
```bash
# API port
uvicorn src.api.personal_api:app --port 8001

# Frontend port
streamlit run frontend/personal_app.py --server.port 8502
```

### Issue 5: Dependency Installation Failed

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Use mirror (if network is slow)
pip install -r requirements_simple.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Issue 6: Python Version Incompatible

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.11 or 3.12

# If version is wrong, install correct version
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

## 📝 Verify Installation

Run the following commands to verify the system is working:

```bash
# 1. Check API health status
curl http://localhost:8000/health

# 2. Test asking a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who are you?",
    "k": 3
  }'

# 3. View metrics
curl http://localhost:8000/metrics
```

## 🎨 Using the Interface

1. **Open browser** and visit http://localhost:8501
2. **Enter a question** in the input box, for example:
   - "Who are you?"
   - "What technologies are you proficient in?"
   - "Tell me about your projects"
3. **Click Ask** button
4. **View answer** and source information
5. **Provide feedback** (👍😊😐😕👎) to help improve the system

## 🔄 Update Knowledge Base

If you modified `data/raw/knowledge_base.json`:

```bash
# Method 1: Re-run setup.py
python setup.py

# Method 2: Use API to rebuild index
curl -X POST "http://localhost:8000/rebuild-index"
```

## 📊 View Logs

```bash
# API logs
tail -f logs/api.log

# Or view console output
```

## 🛑 Stop Services

If using `python run.py`:
- Press `Ctrl+C` to stop all services

If started separately:
- Press `Ctrl+C` in each terminal

If using Docker:
```bash
docker-compose down
```

## 🎯 Next Steps

- View [README.md](README.md) for complete features
- View [PERSONAL_RAG_README.md](PERSONAL_RAG_README.md) for knowledge base structure
- View [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for project optimization details

## 💡 Tips

1. **First run recommended to use Mock mode** (`USE_MOCK=true`), no API Key needed, completely free
2. **Knowledge base already contains your personal information**, can be used directly
3. **If you encounter issues**, check log file `logs/api.log`
4. **Test the system**: Run `pytest tests/` to verify functionality

---

**Enjoy using it!** 🚀
