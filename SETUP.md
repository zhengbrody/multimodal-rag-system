# Setup Guide

This guide will walk you through setting up the Multimodal RAG System from scratch.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Running the Notebooks](#running-the-notebooks)
5. [Starting the Services](#starting-the-services)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Python 3.9+**: [Download](https://www.python.org/downloads/)
- **Git**: [Download](https://git-scm.com/downloads)
- **Docker** (optional, for containerized deployment): [Download](https://www.docker.com/get-started)

### API Keys
You'll need the following API keys:
- **OpenAI API Key**: For embeddings and LLM ([Get key](https://platform.openai.com/api-keys))
- **Pinecone API Key** (optional): For cloud vector database ([Get key](https://www.pinecone.io/))

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-system
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Using conda (alternative)
conda create -n multimodal-rag python=3.9
conda activate multimodal-rag
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Transformers and CLIP
- LangChain and OpenAI
- FastAPI and Uvicorn
- Streamlit
- FAISS and ChromaDB
- Scientific computing libraries (numpy, pandas, scikit-learn)

### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here  # Optional
```

## Data Preparation

### Option 1: Use Sample Data

The notebooks will automatically create sample data if no data is present.

### Option 2: Prepare Your Own Data

1. **Create the data structure:**
```bash
mkdir -p data/raw/images
```

2. **Add your images:**
   - Place product images in `data/raw/images/`
   - Supported formats: JPG, JPEG, PNG

3. **Create metadata CSV:**

Create `data/raw/metadata.csv` with the following structure:

```csv
id,title,description,image_path,category
prod_001,Red T-Shirt,Comfortable red cotton t-shirt with round neck,images/prod_001.jpg,clothing
prod_002,Blue Jeans,Classic blue denim jeans with regular fit,images/prod_002.jpg,clothing
prod_003,White Sneakers,White canvas sneakers with rubber sole,images/prod_003.jpg,footwear
```

### Option 3: Use Public Datasets

Download datasets from:
- [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- [Unsplash Dataset](https://unsplash.com/data)
- [COCO Dataset](https://cocodataset.org/)

## Running the Notebooks

The system uses Jupyter notebooks for the ML pipeline. Run them in order:

### 1. Start Jupyter
```bash
jupyter notebook notebooks/
```

### 2. Run Notebooks in Sequence

#### Notebook 01: Data Processing and Embeddings
- Loads your data
- Generates text embeddings using OpenAI
- Generates image embeddings using CLIP
- Saves processed embeddings

**Estimated time:** 5-10 minutes (depends on dataset size)

#### Notebook 02: Vector Database Setup
- Creates FAISS indexes
- Sets up ChromaDB collections
- Tests similarity search
- Benchmarks performance

**Estimated time:** 2-5 minutes

#### Notebook 03: RAG Pipeline
- Builds retrieval system
- Integrates with LLM
- Creates RAG chains
- Tests Q&A functionality

**Estimated time:** 3-5 minutes

#### Notebook 04: Evaluation
- Runs retrieval metrics
- Evaluates response quality
- Visualizes embeddings
- Generates evaluation report

**Estimated time:** 5-10 minutes

## Starting the Services

### Method 1: Local Development

**Terminal 1 - Start API:**
```bash
cd src
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend/app.py --server.port 8501
```

**Access:**
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

### Method 2: Docker Deployment

**Start all services:**
```bash
docker-compose up --build
```

**Run in background:**
```bash
docker-compose up -d
```

**Stop services:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

## Verification

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "All systems operational"
}
```

### 2. Test Search Endpoint
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "red clothing", "k": 3}'
```

### 3. Access Frontend
Navigate to http://localhost:8501 and try:
- Enter a search query
- Upload an image
- Ask a question in RAG mode

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

#### 2. API Key Errors
```
AuthenticationError: Invalid API key
```
**Solution:**
- Check your `.env` file
- Ensure API key is valid and has credits
- Restart the API server after changing `.env`

#### 3. CUDA/GPU Issues
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Use CPU instead: Set `device = "cpu"` in notebooks
- Or reduce batch size in config

#### 4. Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

#### 5. Docker Issues
```
docker: Cannot connect to the Docker daemon
```
**Solution:**
- Ensure Docker is running
- Check Docker Desktop is started
- Run: `docker info` to verify

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/yourusername/multimodal-rag-system/issues)
2. Review error logs in `logs/app.log`
3. Enable debug mode in `.env`: `DEBUG=True`
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - System information

## Next Steps

Once everything is running:

1. **Experiment with queries** in the frontend
2. **Review the evaluation metrics** in notebook 04
3. **Customize the RAG prompt** in `src/api/main.py`
4. **Add your own data** and re-run the notebooks
5. **Deploy to production** using the deployment guide

## Performance Optimization

### For Large Datasets

1. **Use IVF indexing in FAISS:**
```python
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
```

2. **Batch processing:**
   - Adjust `batch_size` in `configs/config.yaml`
   - Process embeddings in chunks

3. **Use GPU acceleration:**
```python
import faiss
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

### For Production

1. **Use Pinecone** instead of FAISS for scalability
2. **Add Redis** for caching embeddings
3. **Use Gunicorn** with multiple workers:
```bash
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Development Tips

### Code Organization
- Models: `src/models/`
- Utils: `src/utils/`
- API: `src/api/`
- Frontend: `frontend/`

### Adding New Features

1. Update configuration in `configs/config.yaml`
2. Add new endpoints in `src/api/main.py`
3. Update frontend in `frontend/app.py`
4. Add tests in `tests/`

### Testing

Run tests before committing:
```bash
pytest tests/ -v
```

---

**You're all set! Happy coding! 🚀**
