# Personal RAG Q&A System - Quick Start Guide

## 3-Step Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements_simple.txt
```

Or manually install core packages:

```bash
pip install fastapi uvicorn openai numpy scikit-learn streamlit python-dotenv pyyaml requests tiktoken
```

### Step 2: Configure API Key

Edit `.env` file:

```bash
# Replace this line
OPENAI_API_KEY=your_openai_api_key_here

# With your actual key
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

**Get API key**: https://platform.openai.com/api-keys

### Step 3: Run System

```bash
python setup.py  # First run, builds index
python run.py    # Start services
```

After startup:
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## Customize Your Knowledge Base

Edit `data/raw/knowledge_base.json` and replace sample data with your personal information:

### Required Fields

```json
{
  "personal_info": {
    "name": "Your Real Name",
    "title": "Your Title",
    "email": "your@email.com",
    "location": "Your City",
    "bio": "Your bio..."
  }
}
```

### Optional Fields

- `skills`: Skills list
- `projects`: Project experience
- `blog_posts`: Blog articles
- `experience`: Work experience
- `education`: Education background
- `faq`: Frequently asked questions
- `interests`: Hobbies and interests
- `contact`: Contact information

### After Updating Knowledge Base

Re-run `python setup.py` or call API endpoint `/rebuild-index`

---

## Cost Estimation

| Use Case | Model Configuration | Index Cost | Per Query | Monthly (1000 queries) |
|----------|---------------------|-----------|-----------|------------------------|
| Economy | embedding-small + gpt-3.5 | ~$0.02 | ~$0.002 | ~$2 |
| Standard | embedding-large + gpt-3.5 | ~$0.20 | ~$0.003 | ~$3 |
| Premium | embedding-large + gpt-4 | ~$0.20 | ~$0.05 | ~$50 |

---

## FAQ

**Q: Installation failed?**
```bash
pip install --upgrade pip
pip install -r requirements_simple.txt
```

**Q: Where to get API key?**
Visit https://platform.openai.com/api-keys

**Q: How to change models?**
Edit `LLM_MODEL` in `.env` file

**Q: How to add more knowledge?**
Edit `knowledge_base.json` then run `python setup.py`

---

## File Overview

```
├── data/raw/knowledge_base.json  # ← Edit this file
├── .env                          # ← Add API key
├── requirements_simple.txt       # Dependencies
├── setup.py                      # Setup script
├── run.py                        # Launch script
└── PERSONAL_RAG_README.md        # Detailed documentation
```

---

**Having issues?** Check `PERSONAL_RAG_README.md` for detailed documentation.
