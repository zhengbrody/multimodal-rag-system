# Terminal Troubleshooting Guide

## Common Terminal Issues and Solutions

### Issue 1: Command Not Found

**Symptoms:**
```
command not found: python3
command not found: uvicorn
```

**Solutions:**
```bash
# Check Python installation
which python3
python3 --version

# If Python not found, install it
# macOS: brew install python@3.12
# Or use system Python: /usr/bin/python3

# Install dependencies
pip3 install -r requirements_simple.txt
```

### Issue 2: Port Already in Use

**Symptoms:**
```
Address already in use
OSError: [Errno 48] Address already in use
```

**Solutions:**
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use different port
uvicorn src.api.personal_api:app --port 8001
```

### Issue 3: Module Import Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
ImportError: cannot import name 'MockRetriever'
```

**Solutions:**
```bash
# Ensure you're in project root
cd /Users/zhengdong/multimodal-rag-system

# Check Python path
python3 -c "import sys; print(sys.path)"

# Install dependencies
pip3 install -r requirements_simple.txt

# Verify imports
python3 -c "from src.rag.mock_retriever import MockRetriever; print('OK')"
```

### Issue 4: Permission Denied

**Symptoms:**
```
Permission denied
EACCES: permission denied
```

**Solutions:**
```bash
# Make scripts executable
chmod +x restart.sh
chmod +x start_api.sh
chmod +x run_simple.sh

# Check file permissions
ls -l *.sh
```

### Issue 5: Virtual Environment Issues

**Symptoms:**
```
venv/bin/activate: No such file or directory
```

**Solutions:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements_simple.txt
```

### Issue 6: JSON Syntax Error

**Symptoms:**
```
JSONDecodeError: Expecting ',' delimiter
json.decoder.JSONDecodeError
```

**Solutions:**
```bash
# Validate JSON
python3 -c "import json; json.load(open('data/raw/knowledge_base.json')); print('✅ Valid')"

# Check for syntax errors
python3 -m json.tool data/raw/knowledge_base.json > /dev/null
```

### Issue 7: Process Hanging

**Symptoms:**
- Terminal freezes
- Command doesn't respond
- Ctrl+C doesn't work

**Solutions:**
```bash
# Open new terminal and kill hanging processes
pkill -f uvicorn
pkill -f streamlit

# Or find and kill specific process
ps aux | grep uvicorn
kill -9 <PID>
```

### Issue 8: Index File Not Found

**Symptoms:**
```
Knowledge base index not found
FileNotFoundError: data/processed/mock_retriever.pkl
```

**Solutions:**
```bash
# Rebuild index
   ./scripts/rebuild_index.sh

# Or manually rebuild
python3 -m src.rag.knowledge_processor
```

### Issue 9: Environment Variables Not Set

**Symptoms:**
```
OpenAI API key not configured
KeyError: 'USE_MOCK'
```

**Solutions:**
```bash
# Create .env file
cp .env.example .env

# Edit .env file
nano .env
# Set: USE_MOCK=true

# Or export directly
export USE_MOCK=true
```

### Issue 10: Terminal Output Too Long

**Symptoms:**
- Too much output
- Can't see errors
- Terminal scrolling issues

**Solutions:**
```bash
# Redirect output to file
python3 run.py > output.log 2>&1

# View logs separately
tail -f logs/api.log

# Run in background
nohup python3 run.py > output.log 2>&1 &
```

## Quick Diagnostic Commands

```bash
# 1. Check system status
ps aux | grep -E "uvicorn|streamlit" | grep -v grep

# 2. Check ports
lsof -ti:8000 && echo "API running" || echo "API not running"
lsof -ti:8501 && echo "Frontend running" || echo "Frontend not running"

# 3. Test API
curl http://localhost:8000/health

# 4. Check Python
python3 --version
python3 -c "import fastapi, uvicorn, streamlit; print('✅ OK')"

# 5. Check files
ls -lh data/processed/*.pkl
ls -lh .env

# 6. Check logs
tail -20 logs/api.log
```

## Complete Reset

If nothing works:

```bash
# 1. Stop all processes
pkill -f uvicorn
pkill -f streamlit

# 2. Clean up
rm -rf __pycache__ */__pycache__ */*/__pycache__
rm -rf .pytest_cache

# 3. Reinstall dependencies
pip3 install --upgrade -r requirements_simple.txt

# 4. Rebuild index
   ./scripts/rebuild_index.sh

# 5. Restart
python3 run.py
```

## Getting Help

If you encounter a specific error:
1. Copy the full error message
2. Check which command you ran
3. Check the logs: `tail -50 logs/api.log`
4. Verify you're in the project root directory

---

**Note:** Most issues can be resolved by:
1. Ensuring you're in the project root: `cd /Users/zhengdong/multimodal-rag-system`
2. Stopping old processes: `pkill -f uvicorn && pkill -f streamlit`
3. Rebuilding index: `./scripts/rebuild_index.sh`
4. Restarting: `python3 run.py`

