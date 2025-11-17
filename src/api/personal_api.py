"""
FastAPI Backend for Personal RAG Q&A System

Lightweight API for personal website Q&A functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import PersonalRAGRetriever
from rag.pipeline import PersonalRAGPipeline, ConversationalRAGPipeline
from rag.knowledge_processor import build_knowledge_base
from rag.mock_retriever import MockRetriever
from rag.mock_pipeline import MockRAGPipeline, MockConversationalPipeline


# Initialize FastAPI app
app = FastAPI(
    title="Personal RAG API",
    description="Intelligent Q&A API - RAG System based on personal knowledge base",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
retriever: Optional[PersonalRAGRetriever] = None
pipeline: Optional[PersonalRAGPipeline] = None
conversational_pipeline: Optional[ConversationalRAGPipeline] = None

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


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


@app.on_event("startup")
async def startup_event():
    """Load or build knowledge base on startup"""
    global retriever, pipeline, conversational_pipeline

    try:
        print("Initializing Personal RAG System...")

        # Check if we should use mock mode (no API needed)
        use_mock = os.getenv("USE_MOCK", "true").lower() == "true"

        mock_retriever_path = PROCESSED_DIR / "mock_retriever.pkl"
        retriever_path = PROCESSED_DIR / "retriever.pkl"
        knowledge_path = RAW_DIR / "knowledge_base.json"

        if use_mock:
            print("Using MOCK mode (no API costs)")
            # Use mock retriever
            retriever = MockRetriever()

            if mock_retriever_path.exists():
                print("Loading existing mock retriever...")
                retriever.load(str(mock_retriever_path))
            elif knowledge_path.exists():
                print("Building new mock knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(mock_retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

            # Use mock pipelines
            pipeline = MockRAGPipeline(retriever)
            conversational_pipeline = MockConversationalPipeline(retriever)
        else:
            print("Using OpenAI API mode")
            # Initialize real retriever
            retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")

            # Try to load existing retriever, otherwise build new one
            if retriever_path.exists():
                print("Loading existing retriever...")
                retriever.load(str(retriever_path))
            elif knowledge_path.exists():
                print("Building new knowledge base...")
                documents = build_knowledge_base(str(knowledge_path))
                retriever.add_documents(documents)
                retriever.save(str(retriever_path))
            else:
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")

            # Initialize pipelines
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            pipeline = PersonalRAGPipeline(retriever, llm_model=llm_model)
            conversational_pipeline = ConversationalRAGPipeline(retriever, llm_model=llm_model)

        print(f"System initialized successfully!")
        print(f"Documents: {len(retriever.documents)}")
        print(f"Categories: {retriever.get_category_stats()}")

    except Exception as e:
        print(f"Error during startup: {e}")
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
    Main Q&A endpoint

    Ask a question about the person and get an AI-generated answer
    based on their knowledge base.
    """
    if retriever is None or pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
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

        return AnswerResponse(
            question=result['question'],
            answer=result['answer'],
            confidence=result['confidence'],
            sources=sources,
            retrieval_scores=result['retrieval_scores']
        )

    except Exception as e:
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
    """Get system statistics"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    return StatsResponse(
        total_documents=len(retriever.documents),
        categories=retriever.get_category_stats(),
        embedding_model=retriever.embedding_model,
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    )


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
