"""
FastAPI Backend for Personal RAG Q&A System

Professional RAG API with:
- Anti-hallucination strategies
- Structured logging
- Metrics and monitoring
- User feedback collection
- Conversation management
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time
import json
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
            "error_rate": round(self.error_count / self.request_count * 100, 2) if self.request_count > 0 else 0
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global retriever, pipeline, conversational_pipeline

    try:
        logger.info("Initializing Personal RAG System...")

        # Check if we should use mock mode (no API needed)
        use_mock = os.getenv("USE_MOCK", "true").lower() == "true"

        mock_retriever_path = PROCESSED_DIR / "mock_retriever.pkl"
        retriever_path = PROCESSED_DIR / "retriever.pkl"
        knowledge_path = RAW_DIR / "knowledge_base.json"

        if use_mock:
            logger.info("Using MOCK mode (no API costs)")
            # Use mock retriever
            retriever = MockRetriever()

            if mock_retriever_path.exists():
                logger.info("Loading existing mock retriever...")
                retriever.load(str(mock_retriever_path))
            elif knowledge_path.exists():
                logger.info("Building new mock knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(mock_retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

            # Use mock pipelines
            pipeline = MockRAGPipeline(retriever)
            conversational_pipeline = MockConversationalPipeline(retriever)
        else:
            logger.info("Using OpenAI API mode")
            # Initialize real retriever
            retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")

            # Try to load existing retriever, otherwise build new one
            if retriever_path.exists():
                logger.info("Loading existing retriever...")
                retriever.load(str(retriever_path))
            elif knowledge_path.exists():
                logger.info("Building new knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

            # Initialize pipelines
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            pipeline = PersonalRAGPipeline(retriever, llm_model=llm_model)
            conversational_pipeline = ConversationalRAGPipeline(retriever, llm_model=llm_model)

        logger.info(
            "System initialized successfully",
            extra={
                "documents": len(retriever.documents),
                "categories": retriever.get_category_stats(),
                "mode": "mock" if use_mock else "openai"
            }
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
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
            "client": request.client.host if request.client else None
        }
    )
    
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        
        # Record metrics
        metrics.record_request(latency, success=response.status_code < 400)
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 2)
            }
        )
        
        return response
    except Exception as e:
        latency = time.time() - start_time
        metrics.record_request(latency, success=False)
        logger.error(
            f"Error in {request.method} {request.url.path}: {str(e)}",
            extra={"error": str(e), "latency_ms": round(latency * 1000, 2)},
            exc_info=True
        )
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - system status"""
    if retriever is None:
        return HealthResponse(
            status="error",
            message="System not initialized",
            documents_loaded=0,
            categories={}
        )

    return HealthResponse(
        status="healthy",
        message="Personal RAG System is running",
        documents_loaded=len(retriever.documents),
        categories=retriever.get_category_stats()
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

    try:
        logger.info(
            f"Processing question: {request.question[:100]}...",
            extra={
                "question": request.question,
                "k": request.k,
                "conversational": request.conversational,
                "verification": request.use_verification
            }
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
                result = conversational_pipeline.query(
                    request.question, k=request.k
                )
        else:
            if request.use_verification:
                result = pipeline.query_with_verification(
                    request.question, k=request.k
                )
            else:
                result = pipeline.query(
                    request.question, k=request.k
                )

        # Convert sources to response model
        sources = []
        for source in result.get('sources', []):
            sources.append(SourceInfo(
                type=source['type'],
                category=source['category'],
                score=source['score'],
                preview=source['preview']
            ))

        latency = time.time() - start_time
        logger.info(
            f"Question answered successfully",
            extra={
                "question": request.question,
                "confidence": result['confidence'],
                "sources_count": len(sources),
                "latency_ms": round(latency * 1000, 2)
            }
        )

        return AnswerResponse(
            question=result['question'],
            answer=result['answer'],
            confidence=result['confidence'],
            sources=sources,
            retrieval_scores=result['retrieval_scores']
        )

    except Exception as e:
        latency = time.time() - start_time
        logger.error(
            f"Error processing question: {str(e)}",
            extra={
                "question": request.question,
                "error": str(e),
                "latency_ms": round(latency * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


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
        metrics=metrics.get_stats()
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
        "error_rate_percent": stats["error_rate"]
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
            "helpful": feedback.helpful
        }
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        
        metrics.record_feedback()
        
        logger.info(
            "Feedback submitted",
            extra={
                "rating": feedback.rating,
                "helpful": feedback.helpful
            }
        )
        
        return {
            "message": "Feedback submitted successfully",
            "rating": feedback.rating
        }
    
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions users can ask"""
    return {
        "questions": [
            "Who are you? Give me a brief introduction",
            "What technologies are you proficient in?",
            "Tell me about your proudest project",
            "What is your work experience?",
            "What is your education background?",
            "What job opportunities interest you?",
            "What technical blogs have you written?",
            "How can I contact you?",
            "What do you know about RAG systems?",
            "Why did you start writing a technical blog?"
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
            "mode": "mock" if use_mock else "openai"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
