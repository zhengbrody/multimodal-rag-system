"""
FastAPI Backend for Personal RAG Q&A System

Professional RAG API with:
- Anti-hallucination strategies
- Structured logging
- Metrics and monitoring
- User feedback collection
- Conversation management
"""

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import PersonalRAGRetriever
from rag.pipeline import PersonalRAGPipeline, ConversationalRAGPipeline
from rag.knowledge_processor import build_knowledge_base
from rag.mock_retriever import MockRetriever
from rag.mock_pipeline import MockRAGPipeline, MockConversationalPipeline
from utils.logger import setup_logger

# Optional advanced components
try:
    from rag.clip_retriever import CLIPRetriever

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from rag.pinecone_retriever import PineconeRetriever

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from rag.langchain_pipeline import LangChainRAGPipeline, LangChainConversationalPipeline

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from utils.cache import SemanticCache
from utils.monitoring import CloudWatchMonitor, LocalMonitor
from rag.image_search_service import ImageSearchService

# Setup structured logging
logger = setup_logger("personal_rag_api", level="INFO", log_file="logs/api.log")


# Metrics tracking
class Metrics:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.questions_asked = 0
        self.feedback_count = 0
        self.start_time = datetime.now()

    def record_request(self, latency: float, success: bool = True):
        self.request_count += 1
        self.total_latency += latency
        if not success:
            self.error_count += 1

    def record_question(self):
        self.questions_asked += 1

    def record_feedback(self):
        self.feedback_count += 1

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_requests": self.request_count,
            "total_questions": self.questions_asked,
            "total_feedback": self.feedback_count,
            "error_count": self.error_count,
            "average_latency_ms": round(avg_latency * 1000, 2),
            "uptime_seconds": round(uptime, 2),
            "error_rate": (
                round(self.error_count / self.request_count * 100, 2)
                if self.request_count > 0
                else 0
            ),
        }


metrics = Metrics()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Global variables
retriever: Optional[PersonalRAGRetriever] = None
pipeline: Optional[PersonalRAGPipeline] = None
conversational_pipeline: Optional[ConversationalRAGPipeline] = None
cache = None
monitor = None
# Multimodal image search (lazy + gated): serves the Phase-3b OpenCLIP gallery when its
# cache is present, else the endpoints return 503 — keeps the zero-cost mock deployment intact.
image_search = ImageSearchService()


# Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=500)
    k: int = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    use_verification: bool = Field(False, description="Enable answer verification")
    conversational: bool = Field(False, description="Use conversation mode")


class SourceInfo(BaseModel):
    type: str
    category: str
    score: float
    preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    sources: List[SourceInfo]
    retrieval_scores: List[float]


class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int
    categories: Dict[str, int]
    retriever_mode: Optional[str] = None
    pipeline_mode: Optional[str] = None
    cache_enabled: Optional[bool] = None
    monitoring: Optional[str] = None


class StatsResponse(BaseModel):
    total_documents: int
    categories: Dict[str, int]
    embedding_model: str
    llm_model: str
    metrics: Dict[str, Any]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    feedback_text: Optional[str] = Field(None, description="Optional feedback text", max_length=500)
    helpful: Optional[bool] = Field(None, description="Was the answer helpful?")


class TextSearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Free-text query describing the image to find",
        min_length=1,
        max_length=500,
    )
    k: int = Field(5, description="Number of images to return", ge=1, le=50)


class ImageSearchResult(BaseModel):
    image_id: str
    score: float
    rep_caption: Optional[str] = None


class ImageSearchResponse(BaseModel):
    query: str
    modality: str
    backend: Optional[str] = None
    count: int
    results: List[ImageSearchResult]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global retriever, pipeline, conversational_pipeline, cache, monitor

    try:
        logger.info("Initializing Personal RAG System...")

        # Initialize semantic cache
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache = (
            SemanticCache(redis_url=redis_url)
            if os.getenv("ENABLE_CACHE", "false").lower() == "true"
            else None
        )

        # Initialize monitoring
        enable_cloudwatch = os.getenv("ENABLE_CLOUDWATCH", "false").lower() == "true"
        if enable_cloudwatch:
            monitor = CloudWatchMonitor(
                namespace="RAGSystem", region=os.getenv("AWS_REGION", "us-east-1")
            )
        else:
            monitor = LocalMonitor()

        # Check if we should use mock mode (no API needed)
        use_mock = os.getenv("USE_MOCK", "true").lower() == "true"
        retriever_mode = os.getenv("RETRIEVER_MODE", "auto")  # auto, mock, openai, pinecone, clip
        pipeline_mode = os.getenv("PIPELINE_MODE", "default")  # default, langchain

        mock_retriever_path = PROCESSED_DIR / "mock_retriever.pkl"
        retriever_path = PROCESSED_DIR / "retriever.pkl"
        knowledge_path = RAW_DIR / "knowledge_base.json"
        force_rebuild = os.getenv("FORCE_REBUILD_INDEX", "false").lower() == "true"

        def index_is_fresh(index_path: Path) -> bool:
            """Return whether a saved index is newer than the knowledge base."""
            if force_rebuild or not index_path.exists() or not knowledge_path.exists():
                return False
            return index_path.stat().st_mtime >= knowledge_path.stat().st_mtime

        if retriever_mode == "pinecone" and PINECONE_AVAILABLE:
            logger.info("Using Pinecone retriever mode")
            retriever = PineconeRetriever()

            if knowledge_path.exists():
                logger.info("Building knowledge base with Pinecone...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
            else:
                logger.info("No knowledge base file found; using existing Pinecone index")

        elif retriever_mode == "clip" and CLIP_AVAILABLE:
            logger.info("Using CLIP retriever mode")
            retriever = CLIPRetriever()

            clip_retriever_path = PROCESSED_DIR / "clip_retriever.pkl"
            if clip_retriever_path.exists():
                logger.info("Loading existing CLIP retriever...")
                retriever.load(str(clip_retriever_path))
            elif knowledge_path.exists():
                logger.info("Building knowledge base with CLIP...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(clip_retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

        elif use_mock or retriever_mode == "mock":
            logger.info("Using MOCK mode (no API costs)")
            # Use mock retriever
            retriever = MockRetriever()

            if index_is_fresh(mock_retriever_path):
                logger.info("Loading existing mock retriever...")
                retriever.load(str(mock_retriever_path))
            elif knowledge_path.exists():
                logger.info("Building new mock knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(mock_retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")
        else:
            logger.info("Using OpenAI API mode")
            # Initialize real retriever
            retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")

            # Try to load existing retriever, otherwise build new one
            if index_is_fresh(retriever_path):
                logger.info("Loading existing retriever...")
                retriever.load(str(retriever_path))
            elif knowledge_path.exists():
                logger.info("Building new knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

        # Initialize pipelines based on pipeline_mode
        if pipeline_mode == "langchain" and LANGCHAIN_AVAILABLE:
            logger.info("Using LangChain pipeline mode")
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            pipeline = LangChainRAGPipeline(retriever, llm_model=llm_model)
            conversational_pipeline = LangChainConversationalPipeline(
                retriever, llm_model=llm_model
            )
        elif use_mock or retriever_mode == "mock":
            # Use mock pipelines
            pipeline = MockRAGPipeline(retriever)
            conversational_pipeline = MockConversationalPipeline(retriever)
        else:
            # Initialize standard pipelines
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            pipeline = PersonalRAGPipeline(retriever, llm_model=llm_model)
            conversational_pipeline = ConversationalRAGPipeline(retriever, llm_model=llm_model)

        logger.info(
            "System initialized successfully",
            extra={
                "documents": len(retriever.documents),
                "categories": retriever.get_category_stats(),
                "retriever_mode": retriever_mode,
                "pipeline_mode": pipeline_mode,
            },
        )

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Personal RAG System...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Personal RAG API",
    description="Intelligent Q&A API - RAG System with anti-hallucination strategies",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend access
# allow_credentials must be False when allow_origins is ["*"] per CORS spec
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with structured format"""
    start_time = time.time()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
        latency = time.time() - start_time

        # Record metrics
        metrics.record_request(latency, success=response.status_code < 400)

        # Record monitoring metrics
        if monitor:
            monitor.record_latency(request.url.path, latency * 1000)
            if response.status_code >= 400:
                monitor.record_error(request.url.path, str(response.status_code))

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - {response.status_code}",
            extra={"status_code": response.status_code, "latency_ms": round(latency * 1000, 2)},
        )

        return response
    except Exception as e:
        latency = time.time() - start_time
        metrics.record_request(latency, success=False)
        logger.error(
            f"Error in {request.method} {request.url.path}: {str(e)}",
            extra={"error": str(e), "latency_ms": round(latency * 1000, 2)},
            exc_info=True,
        )
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - system status"""
    if retriever is None:
        return HealthResponse(
            status="error", message="System not initialized", documents_loaded=0, categories={}
        )

    return HealthResponse(
        status="healthy",
        message="Personal RAG System is running",
        documents_loaded=len(retriever.documents),
        categories=retriever.get_category_stats(),
        retriever_mode=os.getenv("RETRIEVER_MODE", "auto"),
        pipeline_mode=os.getenv("PIPELINE_MODE", "default"),
        cache_enabled=cache is not None,
        monitoring=(
            "cloudwatch" if isinstance(monitor, CloudWatchMonitor) and monitor.enabled else "local"
        ),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main Q&A endpoint with anti-hallucination strategies

    Ask a question about the person and get an AI-generated answer
    based on their knowledge base. The system uses:
    - Low temperature generation (0.3) for factual responses
    - Strict prompt engineering to prevent fabrication
    - Confidence assessment based on retrieval scores
    - Optional verification step for critical answers
    """
    if retriever is None or pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start_time = time.time()
    metrics.record_question()

    # Check cache first
    if cache:
        cached_result = cache.get(request.question)
        if cached_result:
            logger.info("Cache hit", extra={"question": request.question[:100]})
            if monitor:
                monitor.record_cache_hit(True)
            return AnswerResponse(**cached_result)
        if monitor:
            monitor.record_cache_hit(False)

    try:
        logger.info(
            f"Processing question: {request.question[:100]}...",
            extra={
                "question": request.question,
                "k": request.k,
                "conversational": request.conversational,
                "verification": request.use_verification,
            },
        )

        # Choose pipeline based on request
        if request.conversational:
            if conversational_pipeline is None:
                raise HTTPException(status_code=503, detail="Conversational pipeline not available")

            if request.use_verification:
                result = conversational_pipeline.query_with_verification(
                    request.question, k=request.k
                )
            else:
                result = conversational_pipeline.query(request.question, k=request.k)
        else:
            if request.use_verification:
                result = pipeline.query_with_verification(request.question, k=request.k)
            else:
                result = pipeline.query(request.question, k=request.k)

        # Convert sources to response model
        sources = []
        for source in result.get("sources", []):
            sources.append(
                SourceInfo(
                    type=source["type"],
                    category=source["category"],
                    score=source["score"],
                    preview=source["preview"],
                )
            )

        latency = time.time() - start_time
        logger.info(
            "Question answered successfully",
            extra={
                "question": request.question,
                "confidence": result["confidence"],
                "sources_count": len(sources),
                "latency_ms": round(latency * 1000, 2),
            },
        )

        response = AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=sources,
            retrieval_scores=result["retrieval_scores"],
        )

        # Cache the result
        if cache:
            cache.set(request.question, response.model_dump())

        return response

    except Exception as e:
        latency = time.time() - start_time
        logger.error(
            f"Error processing question: {str(e)}",
            extra={
                "question": request.question,
                "error": str(e),
                "latency_ms": round(latency * 1000, 2),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/search/text", response_model=ImageSearchResponse)
def search_text(request: TextSearchRequest):
    """
    Cross-modal **caption -> image** search (CLIP / OpenCLIP).

    Returns the gallery images whose CLIP embedding is most similar to the query text.
    Backed by the Phase-3b OpenCLIP ViT-H/14 gallery (R@5 = 0.942 on Flickr30k) when its
    cache is present; otherwise returns 503 — e.g. the lightweight mock deployment, which
    never loads the multi-GB model.
    """
    if not ImageSearchService.available():
        raise HTTPException(
            status_code=503,
            detail="Multimodal image search is not available in this deployment "
            "(no CLIP gallery cache). Build it with the eval prep + an openclip eval run.",
        )
    start = time.time()
    try:
        results = image_search.search_text(request.query, k=request.k)
    except (RuntimeError, ImportError) as e:
        raise HTTPException(status_code=503, detail=f"Image search backend unavailable: {e}")
    return ImageSearchResponse(
        query=request.query,
        modality="text->image",
        backend=image_search.backend_label,
        count=len(results),
        results=results,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/search/image", response_model=ImageSearchResponse)
def search_image(file: UploadFile = File(...), k: int = 5):
    """
    Cross-modal **image -> image** (reverse image) search.

    Upload an image; returns the most visually similar gallery images via CLIP / OpenCLIP
    image embeddings. Returns 503 when no gallery cache is present (mock deployment).
    """
    if not ImageSearchService.available():
        raise HTTPException(
            status_code=503,
            detail="Multimodal image search is not available in this deployment.",
        )
    k = max(1, min(int(k), 50))
    start = time.time()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = None
    try:
        contents = file.file.read()  # sync read; this handler runs in Starlette's threadpool
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        results = image_search.search_image(tmp_path, k=k)
    except (RuntimeError, ImportError) as e:
        raise HTTPException(status_code=503, detail=f"Image search backend unavailable: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return ImageSearchResponse(
        query=file.filename or "(uploaded image)",
        modality="image->image",
        backend=image_search.backend_label,
        count=len(results),
        results=results,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/clear-conversation")
async def clear_conversation():
    """Clear conversation history for conversational mode"""
    if conversational_pipeline is None:
        raise HTTPException(status_code=503, detail="Conversational pipeline not available")

    conversational_pipeline.clear_history()
    return {"message": "Conversation history cleared"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get comprehensive system statistics including metrics"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    return StatsResponse(
        total_documents=len(retriever.documents),
        categories=retriever.get_category_stats(),
        embedding_model=retriever.embedding_model,
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        metrics=metrics.get_stats(),
    )


@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics for monitoring (Prometheus-compatible format)"""
    stats = metrics.get_stats()
    return {
        "requests_total": stats["total_requests"],
        "questions_total": stats["total_questions"],
        "feedback_total": stats["total_feedback"],
        "errors_total": stats["error_count"],
        "average_latency_ms": stats["average_latency_ms"],
        "uptime_seconds": stats["uptime_seconds"],
        "error_rate_percent": stats["error_rate"],
    }


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback on answers

    Collects feedback to improve the system and track answer quality.
    This helps identify hallucination issues and improve prompt engineering.
    """
    try:
        # Save feedback to file
        feedback_dir = BASE_DIR / "logs"
        feedback_dir.mkdir(exist_ok=True)
        feedback_file = feedback_dir / "feedback.jsonl"

        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "question": feedback.question,
            "answer": feedback.answer,
            "rating": feedback.rating,
            "feedback_text": feedback.feedback_text,
            "helpful": feedback.helpful,
        }

        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

        metrics.record_feedback()

        logger.info(
            "Feedback submitted", extra={"rating": feedback.rating, "helpful": feedback.helpful}
        )

        return {"message": "Feedback submitted successfully", "rating": feedback.rating}

    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions users can ask"""
    return {
        "questions": [
            "Why are you a strong fit for Machine Learning Engineer roles?",
            "Tell me about your Allianz fraud detection work",
            "How should this RAG project be presented to recruiters?",
            "What is your strongest production ML project?",
            "What is your work authorization and availability?",
            "How do you build reliable ML systems?",
            "What evidence supports your RAG project claims?",
            "What is your work experience?",
            "How can I contact you?",
        ]
    }


@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild the knowledge base index"""
    global retriever, pipeline, conversational_pipeline

    try:
        knowledge_path = RAW_DIR / "knowledge_base.json"
        use_mock = os.getenv("USE_MOCK", "true").lower() == "true"

        if not knowledge_path.exists():
            raise HTTPException(status_code=404, detail="Knowledge base file not found")

        print("Rebuilding knowledge base...")

        # Invalidate cache on rebuild
        if cache:
            cache.invalidate_all()

        # Build new knowledge base
        documents = build_knowledge_base(str(knowledge_path))

        if use_mock:
            # Mock mode
            mock_retriever_path = PROCESSED_DIR / "mock_retriever.pkl"
            new_retriever = MockRetriever()
            new_retriever.add_documents(documents)
            new_retriever.save(str(mock_retriever_path))

            retriever = new_retriever
            pipeline = MockRAGPipeline(retriever)
            conversational_pipeline = MockConversationalPipeline(retriever)
        else:
            # OpenAI mode
            retriever_path = PROCESSED_DIR / "retriever.pkl"
            new_retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")
            new_retriever.add_documents(documents)
            new_retriever.save(str(retriever_path))

            retriever = new_retriever
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            pipeline = PersonalRAGPipeline(retriever, llm_model=llm_model)
            conversational_pipeline = ConversationalRAGPipeline(retriever, llm_model=llm_model)

        return {
            "message": "Knowledge base rebuilt successfully",
            "documents": len(retriever.documents),
            "categories": retriever.get_category_stats(),
            "mode": "mock" if use_mock else "openai",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
