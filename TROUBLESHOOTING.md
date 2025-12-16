# 🔧 Troubleshooting Guide

## Issue: Frontend Shows "API Offline" ❌

### Cause
The API server is not running, so the frontend cannot connect to the backend.

### Solutions

#### Method 1: Use run.py (Recommended)
```bash
# This starts both API and frontend
python3 run.py
```

#### Method 2: Start Separately

**Terminal 1 - Start API:**
```bash
# Option A: Use script
./start_api.sh

# Option B: Direct command
uvicorn src.api.personal_api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend/personal_app.py
```

#### Method 3: Run API in Background
```bash
# Start API in background
nohup uvicorn src.api.personal_api:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# View logs
tail -f api.log
```

### Verify API is Running

```bash
# Check port
lsof -ti:8000

# Test health check
curl http://localhost:8000/health

# Should return JSON response
```

### Check Frontend Configuration

The frontend connects to `http://localhost:8000` by default. If the API is on a different port, set the environment variable:

```bash
# Set API URL when starting frontend
API_URL=http://localhost:8001 streamlit run frontend/personal_app.py
```

## Other Common Issues

### Issue 1: Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -ti:8000

# Kill process
kill -9 $(lsof -ti:8000)

# Or use different port
uvicorn src.api.personal_api:app --port 8001
```

### Issue 2: Module Import Error

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in project root directory
cd /Users/zhengdong/multimodal-rag-system

# Check Python path
python3 -c "import sys; print(sys.path)"

# Install dependencies
pip install -r requirements_simple.txt
```

### Issue 3: Index File Not Found

**Error:**
```
Knowledge base index not found
```

**Solution:**
```bash
# Rebuild index
python3 setup.py
```

### Issue 4: Environment Variables Not Configured

**Error:**
```
OpenAI API key not configured
```

**Solution:**
```bash
# Create .env file
cp .env.example .env

# Edit .env, set USE_MOCK=true (no API Key needed)
# Or set OPENAI_API_KEY=sk-...
```

### Issue 5: Frontend Cannot Connect to API (CORS Error)

**Solution:**
Check CORS configuration in `src/api/personal_api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should allow all origins
    ...
)
```

## Quick Diagnostic Commands

```bash
# 1. Check Python version
python3 --version

# 2. Check dependencies
python3 -c "import fastapi, uvicorn, streamlit; print('✅ Dependencies OK')"

# 3. Check if API is running
curl http://localhost:8000/health

# 4. Check port usage
lsof -ti:8000 && echo "Port 8000 in use" || echo "Port 8000 free"

# 5. Check index files
ls -lh data/processed/*.pkl

# 6. Check environment variables
cat .env | grep -E "USE_MOCK|API_URL"
```

## View Logs

```bash
# API logs
tail -f logs/api.log

# If using nohup
tail -f api.log

# Streamlit logs (in terminal output)
```

## Complete Reset

If nothing works, try a complete reset:

```bash
# 1. Stop all processes
pkill -f uvicorn
pkill -f streamlit

# 2. Reinstall dependencies
pip install --upgrade -r requirements_simple.txt

# 3. Rebuild index
python3 setup.py

# 4. Restart
python3 run.py
```

## Get Help

If the issue persists:
1. Check error messages in `logs/api.log`
2. View error stack traces in terminal output
3. Verify all dependencies are correctly installed
4. Confirm Python version (requires 3.11+)

---

**Current Status Check:**
- ✅ API module can be imported normally
- ✅ Index files exist
- ✅ Dependencies are installed

**Next Step:** Start the API server, then refresh the frontend page.
