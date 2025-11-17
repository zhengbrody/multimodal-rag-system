# Personal RAG Q&A System

> Intelligent Q&A system for personal websites powered by RAG technology

## Features

- **Intelligent Q&A**: Visitors can ask questions about you in natural language
- **Anti-Hallucination Strategies**: Multiple mechanisms ensure AI doesn't fabricate information
- **Real-time Retrieval**: Fast retrieval based on semantic similarity
- **Confidence Assessment**: Display answer reliability level
- **Source Tracing**: View answer sources and evidence
- **Conversation Mode**: Multi-turn dialogue with context retention

## Architecture

```
User Question → Streamlit Frontend → FastAPI Backend → RAG Pipeline
                                                          ↓
                                              OpenAI Embeddings (Retrieval)
                                                          ↓
                                              Similarity Matching (NumPy)
                                                          ↓
                                              GPT-3.5/4 (Answer Generation)
                                                          ↓
                                              Anti-Hallucination Verification
```

## Quick Start

### 1. Configure API Key

Edit `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 2. Customize Knowledge Base

Edit `data/raw/knowledge_base.json` and replace with your personal information:

```json
{
  "personal_info": {
    "name": "Your Name",
    "title": "Your Title",
    ...
  },
  "projects": [...],
  "blog_posts": [...],
  ...
}
```

### 3. Run Setup Script

```bash
python setup.py
```

This will:
- Install dependencies
- Verify configuration
- Build vector index (generate embeddings)

### 4. Start System

```bash
python run.py
```

Or start separately:

```bash
# Terminal 1: Start API
python -m uvicorn src.api.personal_api:app --reload --port 8000

# Terminal 2: Start Frontend
streamlit run frontend/personal_app.py
```

### 5. Access System

- **Frontend UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Knowledge Base Structure

`data/raw/knowledge_base.json` contains these sections:

- **personal_info**: Basic personal information
- **skills**: Skills and expertise
- **projects**: Project experience
- **blog_posts**: Blog articles
- **experience**: Work experience
- **education**: Education background
- **faq**: Frequently asked questions
- **interests**: Hobbies and activities
- **contact**: Contact information

## Anti-Hallucination Strategies

1. **Low Temperature Generation** (temperature=0.3): Reduce randomness, increase determinism

2. **Strict System Prompt**:
   - Only use provided context
   - Explicitly state when unable to answer
   - Do not speculate or fabricate information

3. **Confidence Assessment**:
   - Evaluate answer reliability based on retrieval scores
   - Three-level confidence indicator: High/Medium/Low

4. **Source Tracing**:
   - Display source documents for answers
   - Show relevance scores

5. **Optional Answer Verification** (Experimental):
   - Second pass verification against source materials
   - Detect potential hallucinated content

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status |
| `/health` | GET | Health check |
| `/ask` | POST | Q&A endpoint |
| `/stats` | GET | System statistics |
| `/sample-questions` | GET | Sample questions |
| `/rebuild-index` | POST | Rebuild index |
| `/clear-conversation` | POST | Clear conversation history |

### Q&A Request Example

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What technologies are you proficient in?",
    "k": 5,
    "use_verification": false,
    "conversational": false
  }'
```

## Customization

### Change LLM Model

Edit `.env`:

```bash
# Cheaper but lower quality
LLM_MODEL=gpt-3.5-turbo

# Higher quality but more expensive
LLM_MODEL=gpt-4

# Balanced choice
LLM_MODEL=gpt-4-turbo-preview
```

### Adjust Retrieval Parameters

In `src/rag/retriever.py`:

```python
self.category_weights = {
    'faq': 1.2,      # Boost FAQ weight
    'about': 1.1,    # Boost personal info weight
    'projects': 1.0,
    ...
}
```

### Modify Generation Parameters

In `src/rag/pipeline.py`:

```python
temperature=0.3,      # Lower = more deterministic, Higher = more creative
max_tokens=1000,      # Maximum answer length
```

## Deployment

### Local Development

Use the provided `run.py` script.

### Production

1. **Use Gunicorn for API**:
```bash
gunicorn src.api.personal_api:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Use Docker**:
Modify existing `Dockerfile` and `docker-compose.yml`

3. **Cloud Deployment**:
- AWS EC2/ECS
- Google Cloud Run
- Heroku
- DigitalOcean App Platform

## Cost Estimation

Using `text-embedding-3-small` + `gpt-3.5-turbo`:
- Index building: ~$0.01-0.05 (depends on document count)
- Per query: ~$0.001-0.003
- Monthly cost (1000 queries): ~$1-3

Using `text-embedding-3-large` + `gpt-4`:
- Index building: ~$0.10-0.50
- Per query: ~$0.03-0.10
- Monthly cost (1000 queries): ~$30-100

## Extension Ideas

1. **Add More Data Sources**:
   - Import GitHub repository READMEs
   - Scrape blog articles
   - Parse resume PDFs

2. **Enhance Retrieval**:
   - Add BM25 hybrid retrieval
   - Implement reranking
   - Add query expansion

3. **Improve Frontend**:
   - Add dark mode
   - Implement typewriter effect
   - Add voice input

4. **Integrate with Existing Website**:
   - Embed as chat widget
   - Direct API integration

## Troubleshooting

### Common Issues

1. **API startup failure**:
   - Check if port 8000 is occupied
   - Verify OpenAI API key

2. **Poor retrieval results**:
   - Increase k value
   - Enrich knowledge base content
   - Adjust category weights

3. **Low answer quality**:
   - Try GPT-4 model
   - Enable answer verification
   - Optimize knowledge base structure

4. **High costs**:
   - Use text-embedding-3-small
   - Use gpt-3.5-turbo
   - Reduce number of retrieved documents

## File Structure

```
multimodal-rag-system/
├── data/
│   ├── raw/
│   │   └── knowledge_base.json    # Personal knowledge base
│   └── processed/
│       └── retriever.pkl          # Vector index
├── src/
│   ├── rag/
│   │   ├── knowledge_processor.py # Knowledge base processor
│   │   ├── retriever.py           # Retriever
│   │   └── pipeline.py            # RAG pipeline
│   └── api/
│       └── personal_api.py        # FastAPI backend
├── frontend/
│   └── personal_app.py            # Streamlit frontend
├── .env                           # Environment variables
├── requirements_simple.txt        # Dependencies
├── setup.py                       # Setup script
└── run.py                         # Launch script
```

## License

MIT License - Free to use and modify
