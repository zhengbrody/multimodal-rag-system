# 🔄 Project Workflow

## 📋 Standard Execution Order After Modifying the Project

### 1️⃣ Preparation Phase

```bash
# 1. Ensure you're in project root directory
cd /Users/zhengdong/multimodal-rag-system

# 2. Activate virtual environment (if using)
source venv/bin/activate  # macOS/Linux
# or
# venv\Scripts\activate  # Windows

# 3. Check Python version
python3 --version  # Should be 3.11 or 3.12
```

### 2️⃣ Dependency Check

```bash
# Check and install/update dependencies
pip install -r requirements_simple.txt

# If requirements were modified, reinstall
pip install --upgrade -r requirements_simple.txt
```

### 3️⃣ Configuration Check

```bash
# Check .env file
cat .env | grep -E "USE_MOCK|API_URL|OPENAI_API_KEY"

# If it doesn't exist, create it
cp .env.example .env
# Then edit .env file
```

### 4️⃣ Knowledge Base Update

If you modified `data/raw/knowledge_base.json`:

```bash
# Rebuild index
   ./scripts/rebuild_index.sh

# Or use API to rebuild (if API is running)
curl -X POST "http://localhost:8000/rebuild-index"
```

### 5️⃣ Stop Old Processes

```bash
# Stop all related processes
pkill -f uvicorn
pkill -f streamlit

# Or find and manually stop
lsof -ti:8000 | xargs kill -9  # API
lsof -ti:8501 | xargs kill -9  # Frontend
```

### 6️⃣ Start System

**Option A: One-click start (Recommended)**
```bash
python3 run.py
```

**Option B: Start separately**

Terminal 1 - API:
```bash
uvicorn src.api.personal_api:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Frontend:
```bash
streamlit run frontend/personal_app.py
```

### 7️⃣ Verification

```bash
# Check API health status
curl http://localhost:8000/health

# Check port usage
lsof -ti:8000 && echo "✅ API running"
lsof -ti:8501 && echo "✅ Frontend running"

# Access application
# Open browser: http://localhost:8501
```

## 🔧 Common Modification Scenarios

### Scenario 1: Modified Backend Code (API)

```bash
# 1. Stop API
pkill -f uvicorn

# 2. Restart API (will auto-reload)
uvicorn src.api.personal_api:app --reload --host 0.0.0.0 --port 8000

# Or use run.py (will auto-detect changes)
python3 run.py
```

### Scenario 2: Modified Frontend Code

```bash
# Streamlit will auto-reload, just refresh browser
# Or manually restart
pkill -f streamlit
streamlit run frontend/personal_app.py
```

### Scenario 3: Modified Knowledge Base

```bash
# 1. Stop system
pkill -f uvicorn
pkill -f streamlit

# 2. Rebuild index
   ./scripts/rebuild_index.sh

# 3. Restart
python3 run.py
```

### Scenario 4: Modified Dependencies

```bash
# 1. Stop system
pkill -f uvicorn
pkill -f streamlit

# 2. Update dependencies
pip install --upgrade -r requirements_simple.txt

# 3. Restart
python3 run.py
```

### Scenario 5: Modified Configuration

```bash
# 1. Edit .env file
nano .env  # or use other editor

# 2. Restart system (config changes require restart)
pkill -f uvicorn
pkill -f streamlit
python3 run.py
```

## 🐛 Error Handling Flow

### Error 1: Session State Error

**Error:**
```
st.session_state.question_input cannot be modified after widget instantiated
```

**Solution:**
- ✅ Fixed: Use `clear_input` flag instead of directly modifying session state
- If issue persists, clear browser cache or restart Streamlit

### Error 2: PyArrow Import Error

**Error:**
```
cannot import name 'is_threading_enabled' from 'pyarrow.lib'
```

**Solution:**
```bash
# Method 1: Upgrade pyarrow
pip install --upgrade pyarrow

# Method 2: Downgrade pyarrow (if upgrade fails)
pip install pyarrow==14.0.0

# Method 3: Use alternative (already implemented)
# Code has try-except, will show table instead of chart on failure
```

### Error 3: Port Already in Use

```bash
# Find process using port
lsof -ti:8000
lsof -ti:8501

# Kill process
kill -9 $(lsof -ti:8000)
kill -9 $(lsof -ti:8501)

# Or use different port
uvicorn src.api.personal_api:app --port 8001
streamlit run frontend/personal_app.py --server.port 8502
```

### Error 4: Module Import Error

```bash
# Ensure you're in project root
cd /Users/zhengdong/multimodal-rag-system

# Check Python path
python3 -c "import sys; sys.path.insert(0, 'src'); from api.personal_api import app; print('OK')"

# Reinstall dependencies
pip install --force-reinstall -r requirements_simple.txt
```

## 📝 Quick Checklist

After each modification, check in order:

- [ ] ✅ Python version correct (3.11+)
- [ ] ✅ Dependencies installed
- [ ] ✅ .env configuration correct
- [ ] ✅ Knowledge base index updated (if knowledge base was modified)
- [ ] ✅ Old processes stopped
- [ ] ✅ API started successfully
- [ ] ✅ Frontend started successfully
- [ ] ✅ Browser can access
- [ ] ✅ API health check passes
- [ ] ✅ Can ask questions normally

## 🚀 One-Click Restart Script

Create `restart.sh`:

```bash
#!/bin/bash
echo "🔄 Restarting Personal RAG System..."

# Stop all processes
pkill -f uvicorn
pkill -f streamlit
sleep 2

# Start system
python3 run.py
```

Usage:
```bash
chmod +x restart.sh
./restart.sh
```

## 💡 Best Practices

1. **Use `--reload` during development**: API will auto-reload code changes
2. **Use virtual environment**: Avoid dependency conflicts
3. **Check logs regularly**: `tail -f logs/api.log`
4. **Clean before testing**: Stop old processes, clear cache
5. **Version control**: Commit code to git before modifications

## 📊 Execution Order Summary

```
Modify code
    ↓
Check dependencies (pip install)
    ↓
Update configuration (if needed)
    ↓
Rebuild index (if knowledge base modified)
    ↓
Stop old processes (pkill)
    ↓
Start system (python3 run.py)
    ↓
Verify running (curl /health)
    ↓
Test functionality (browser access)
```

---

**Remember:** In most cases, just run `python3 run.py`!
