# Codebase Cleanup Analysis

## 📊 Current Status

### ✅ What Works
- **Personal RAG API** (`src/api/personal_api.py`) - ✓ Imports successfully
- **Knowledge Base** - Data exists in `data/raw/knowledge_base.json`
- **Retrievers** - Both `retriever.pkl` and `mock_retriever.pkl` exist
- **Frontend dependencies** - Minimal requirements work for Streamlit Cloud

### ⚠️ Issues Found

1. **Frontend API URL Configuration**
   - `frontend/app.py` uses `st.secrets.get()` which requires `.streamlit/secrets.toml`
   - File doesn't exist locally, causes import errors

2. **Duplicate Documentation** (7 markdown files, much overlap)
   - README.md (431 lines) - Main docs
   - DEPLOYMENT.md (384 lines) - Deployment guide ✅ Keep
   - SETUP.md (349 lines) - Redundant with README
   - PERSONAL_RAG_README.md (291 lines) - Separate system docs
   - PROJECT_SUMMARY.md (273 lines) - Redundant with README
   - FILE_STRUCTURE.md (184 lines) - Redundant info
   - QUICKSTART.md (121 lines) - Redundant with README

3. **Two Separate Systems in One Repo**
   - **Multimodal RAG** (frontend/app.py + src/api/main.py) - Requires torch, CLIP, images
   - **Personal RAG** (frontend/personal_app.py + src/api/personal_api.py) - Lightweight, text-only

4. **Multiple Requirements Files**
   - requirements.txt - Full ML stack (torch, transformers, CLIP)
   - requirements_simple.txt - Simplified dependencies
   - frontend/requirements.txt - Frontend-only (new, ✅ correct)

## 🗂️ Recommended Cleanup

### Files to DELETE (Redundant Documentation)

```bash
# Redundant markdown files
FILE_STRUCTURE.md        # Info already in README
PROJECT_SUMMARY.md       # Summary already in README
QUICKSTART.md           # Instructions in README
SETUP.md                # Covered in README + DEPLOYMENT.md
```

### Files to KEEP

**Essential Documentation:**
- README.md - Main project documentation
- DEPLOYMENT.md - Streamlit Cloud deployment guide
- PERSONAL_RAG_README.md - Personal RAG system specific guide
- LICENSE
- .gitignore

**Code:**
- All src/ files - Both systems may be needed
- Both frontend files - User can choose which to use
- Configuration files (.env.example, configs/, etc.)

### Files to FIX

**1. frontend/app.py** - API URL configuration issue
```python
# Current (line 28):
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Should be:
API_URL = os.getenv("API_URL", st.secrets.get("API_URL", "http://localhost:8000") if hasattr(st, "secrets") else "http://localhost:8000")
```

**2. Create .streamlit/secrets.toml for local development**
```toml
API_URL = "http://localhost:8000"
```

## 🔍 System Comparison

| Feature | Multimodal RAG | Personal RAG |
|---------|---------------|--------------|
| **Frontend** | app.py | personal_app.py |
| **Backend** | main.py | personal_api.py |
| **Dependencies** | torch, CLIP, transformers | lightweight (OpenAI only) |
| **Size** | ~4GB | ~200MB |
| **Use Case** | Image + text search | Text Q&A only |
| **Deployed?** | ❌ Not yet (dependency issues) | ✅ Can deploy easily |

## 💡 Recommendations

### Option 1: Focus on Personal RAG (Recommended for Quick Deployment)
- **Keep:** personal_api.py, personal_app.py, mock implementations
- **Remove:** main.py (multimodal API), app.py (multimodal frontend)
- **Update:** README to focus on Personal RAG
- **Benefit:** Lighter, easier to deploy, works with current data

### Option 2: Keep Both Systems
- **Separate branches:** main (Personal RAG), multimodal (Full system)
- **Update:** README to explain both systems
- **Benefit:** Preserve both implementations

### Option 3: Fix Multimodal System
- **Add:** Sample image data and metadata.csv
- **Update:** Setup notebooks to process data
- **Create:** Full vector indexes for images
- **Benefit:** Complete showcase of ML skills

## 🎯 Immediate Actions Needed

1. **Fix frontend/app.py** - API URL fallback for local development
2. **Create .streamlit/secrets.toml** - For local testing
3. **Delete redundant docs** - Remove 4 duplicate markdown files
4. **Choose system** - Decide which system to focus on
5. **Update README** - Reflect chosen system clearly

## 📋 Testing Checklist

- [✅] personal_api.py imports successfully
- [❌] app.py imports (secrets issue)
- [⏳] personal_api.py runs (uvicorn)
- [⏳] personal_app.py runs (streamlit)
- [⏳] End-to-end flow works

---

**Created:** 2024-11-18
**Purpose:** Guide codebase cleanup and system selection
