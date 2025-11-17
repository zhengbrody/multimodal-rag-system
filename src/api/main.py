"""
FastAPI Backend for Multimodal RAG System

This API provides endpoints for:
- Text-based semantic search
- Image-based similarity search
- RAG-powered question answering
- Hybrid multimodal retrieval
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import os
from pathlib import Path
import pickle
import io
from PIL import Image
import torch
import clip

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="API for multimodal retrieval and RAG-powered question answering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
retriever = None
llm = None
rag_prompt = None
clip_model = None
clip_preprocess = None
device = None

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Pydantic models
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(5, description="Number of results to return", ge=1, le=50)
    use_rag: bool = Field(False, description="Whether to use RAG for response generation")


class SearchResult(BaseModel):
    id: str
    title: str
    description: str
    category: str
    score: float
    image_path: Optional[str] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    retrieved_products: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    message: str


@app.on_event("startup")
async def load_models():
    """Load models and indexes on startup"""
    global retriever, llm, rag_prompt, clip_model, clip_preprocess, device

    try:
        print("Loading models...")

        # Load RAG pipeline components
        rag_path = PROCESSED_DIR / "rag_pipeline.pkl"
        if rag_path.exists():
            with open(rag_path, 'rb') as f:
                components = pickle.load(f)
            retriever = components.get('retriever')
            llm = components.get('llm')
            rag_prompt = components.get('rag_prompt')
            print("RAG pipeline loaded successfully")
        else:
            print(f"Warning: RAG pipeline not found at {rag_path}")

        # Load CLIP model for image processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print(f"CLIP model loaded on {device}")

        print("All models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="Multimodal RAG API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if retriever is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Models not loaded"
            }
        )

    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """
    Text-based semantic search

    Returns top-k most relevant items based on semantic similarity
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    try:
        # Perform retrieval
        results = retriever.retrieve(query.query, k=query.k)

        # Convert to response model
        search_results = [
            SearchResult(
                id=r['id'],
                title=r['title'],
                description=r['description'],
                category=r['category'],
                score=r['score'],
                image_path=r.get('image_path')
            )
            for r in results
        ]

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/rag", response_model=RAGResponse)
async def rag_query(query: SearchQuery):
    """
    RAG-powered question answering

    Retrieves relevant context and generates natural language response using LLM
    """
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")

    try:
        # Retrieve relevant documents
        results = retriever.retrieve(query.query, k=query.k)

        # Format context
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(
                f"Product {i+1}:\n"
                f"- Title: {result['title']}\n"
                f"- Description: {result['description']}\n"
                f"- Category: {result['category']}\n"
                f"- Relevance Score: {result['score']:.2f}\n"
            )
        context = "\n".join(context_parts)

        # Generate response with LLM
        from langchain.schema.output_parser import StrOutputParser

        chain = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": query.query
        })

        # Convert results to response model
        search_results = [
            SearchResult(
                id=r['id'],
                title=r['title'],
                description=r['description'],
                category=r['category'],
                score=r['score'],
                image_path=r.get('image_path')
            )
            for r in results
        ]

        return RAGResponse(
            question=query.query,
            answer=answer,
            retrieved_products=search_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.post("/search/image", response_model=List[SearchResult])
async def search_by_image(
    file: UploadFile = File(...),
    k: int = Query(5, description="Number of results to return", ge=1, le=50)
):
    """
    Image-based similarity search

    Upload an image and find visually similar items
    """
    if clip_model is None or retriever is None:
        raise HTTPException(status_code=503, detail="CLIP model not loaded")

    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Generate image embedding using CLIP
        image_input = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        image_embedding = image_features.cpu().numpy()[0]

        # Perform retrieval using image embedding
        results = retriever.retrieve_with_image(image_embedding, k=k)

        # Convert to response model
        search_results = [
            SearchResult(
                id=r['id'],
                title=r['title'],
                description=r['description'],
                category=r['category'],
                score=r['score'],
                image_path=r.get('image_path')
            )
            for r in results
        ]

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")


@app.post("/search/hybrid")
async def hybrid_search(
    query: str = Query(..., description="Text query"),
    file: Optional[UploadFile] = File(None),
    k: int = Query(5, description="Number of results to return", ge=1, le=50),
    text_weight: float = Query(0.5, description="Weight for text vs image (0-1)", ge=0, le=1)
):
    """
    Hybrid multimodal search combining text and image

    Provide both text query and image for combined semantic + visual search
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    try:
        # Text retrieval
        text_results = retriever.retrieve(query, k=k*2)

        if file:
            # Image retrieval if file provided
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            image_embedding = image_features.cpu().numpy()[0]
            image_results = retriever.retrieve_with_image(image_embedding, k=k*2)

            # Combine scores
            combined_scores = {}

            for result in text_results:
                doc_id = result['id']
                combined_scores[doc_id] = text_weight * result['score']

            for result in image_results:
                doc_id = result['id']
                if doc_id in combined_scores:
                    combined_scores[doc_id] += (1 - text_weight) * result['score']
                else:
                    combined_scores[doc_id] = (1 - text_weight) * result['score']

            # Sort by combined score
            sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]

            # Get full result objects
            all_results = {r['id']: r for r in text_results + image_results}
            final_results = [all_results[doc_id] for doc_id in sorted_ids if doc_id in all_results]

            # Update scores
            for result in final_results:
                result['score'] = combined_scores[result['id']]
        else:
            # Text-only if no image
            final_results = text_results[:k]

        # Convert to response model
        search_results = [
            SearchResult(
                id=r['id'],
                title=r['title'],
                description=r['description'],
                category=r['category'],
                score=r['score'],
                image_path=r.get('image_path')
            )
            for r in final_results
        ]

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Load metadata to get stats
        import pandas as pd
        metadata_df = pd.read_csv(PROCESSED_DIR / "metadata_processed.csv")

        return {
            "total_items": len(metadata_df),
            "categories": metadata_df['category'].value_counts().to_dict(),
            "embedding_dimensions": {
                "text": np.load(PROCESSED_DIR / "text_embeddings.npy").shape[1],
                "clip": np.load(PROCESSED_DIR / "clip_text_embeddings.npy").shape[1]
            },
            "models_loaded": {
                "retriever": retriever is not None,
                "llm": llm is not None,
                "clip": clip_model is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
