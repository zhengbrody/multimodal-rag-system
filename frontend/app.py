"""
Streamlit Frontend for Multimodal RAG System

Features:
- Text-based semantic search
- Image upload and visual search
- RAG-powered Q&A
- Hybrid multimodal search
- Interactive result visualization
"""

import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    .rag-response {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Multimodal RAG System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Product Search with Text & Image Understanding</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1E88E5/FFFFFF?text=RAG+System", use_container_width=True)

    st.markdown("### Search Mode")
    search_mode = st.radio(
        "Choose search type:",
        ["Text Search", "RAG Q&A", "Image Search", "Hybrid Search"],
        help="Select the type of search you want to perform"
    )

    st.markdown("### Settings")
    k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if search_mode == "Hybrid Search":
        text_weight = st.slider(
            "Text vs Image Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = Image only, 1 = Text only"
        )

    st.markdown("---")
    st.markdown("### About")
    st.info(
        """
        This system uses:
        - **CLIP** for multimodal embeddings
        - **OpenAI** for text embeddings
        - **FAISS** for vector search
        - **LangChain** for RAG pipeline
        - **GPT-4** for response generation
        """
    )

    # System status
    st.markdown("### System Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")


# Main content area
def display_results(results, show_images=False):
    """Display search results in a nice format"""
    if not results:
        st.warning("No results found")
        return

    for i, result in enumerate(results):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {i+1}. {result['title']}")
                st.markdown(f"**Category:** {result['category']}")
                st.markdown(f"**Description:** {result['description']}")

            with col2:
                score = result['score']
                color = "#4CAF50" if score > 0.8 else "#FF9800" if score > 0.6 else "#F44336"
                st.markdown(
                    f'<div style="background-color: {color}; color: white; padding: 0.5rem; '
                    f'border-radius: 10px; text-align: center; font-weight: bold;">'
                    f'Score: {score:.3f}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")


# Text Search Mode
if search_mode == "Text Search":
    st.markdown('<div class="sub-header">📝 Text-Based Semantic Search</div>', unsafe_allow_html=True)
    st.write("Enter a text query to find semantically similar products")

    query = st.text_input(
        "Search Query",
        placeholder="e.g., comfortable red clothing for casual wear",
        help="Enter natural language description"
    )

    if st.button("Search", type="primary", use_container_width=True):
        if query:
            with st.spinner("Searching..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={"query": query, "k": k}
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"Found {len(results)} results")
                        display_results(results)
                    else:
                        st.error(f"Search failed: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")


# RAG Q&A Mode
elif search_mode == "RAG Q&A":
    st.markdown('<div class="sub-header">🤖 RAG-Powered Question Answering</div>', unsafe_allow_html=True)
    st.write("Ask questions and get AI-powered answers with product recommendations")

    question = st.text_area(
        "Your Question",
        placeholder="e.g., What products would you recommend for a casual weekend outfit?",
        help="Ask any question about the products"
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if question:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/rag",
                        json={"query": question, "k": k}
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Display AI answer
                        st.markdown('<div class="sub-header">AI Assistant Response:</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="rag-response">{data["answer"]}</div>',
                            unsafe_allow_html=True
                        )

                        # Display retrieved products
                        st.markdown('<div class="sub-header">Referenced Products:</div>', unsafe_allow_html=True)
                        display_results(data["retrieved_products"])
                    else:
                        st.error(f"RAG query failed: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question")


# Image Search Mode
elif search_mode == "Image Search":
    st.markdown('<div class="sub-header">🖼️ Image-Based Visual Search</div>', unsafe_allow_html=True)
    st.write("Upload an image to find visually similar products")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a product image"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("Search Similar Images", type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)

                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(
                            f"{API_URL}/search/image",
                            files=files,
                            params={"k": k}
                        )

                        if response.status_code == 200:
                            results = response.json()
                            st.success(f"Found {len(results)} similar items")

                            # Display results below
                            st.markdown("---")
                            display_results(results, show_images=True)
                        else:
                            st.error(f"Image search failed: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# Hybrid Search Mode
elif search_mode == "Hybrid Search":
    st.markdown('<div class="sub-header">🔀 Hybrid Multimodal Search</div>', unsafe_allow_html=True)
    st.write("Combine text and image for powerful multimodal search")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Text Query**")
        query = st.text_input(
            "Search Query",
            placeholder="e.g., casual clothing",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**Image Upload (Optional)**")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption="Reference Image", width=200)

    if st.button("Hybrid Search", type="primary", use_container_width=True):
        if query or uploaded_file:
            with st.spinner("Performing hybrid search..."):
                try:
                    files = None
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                    response = requests.post(
                        f"{API_URL}/search/hybrid",
                        params={
                            "query": query or "general products",
                            "k": k,
                            "text_weight": text_weight
                        },
                        files=files
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"Found {len(results)} results")
                        display_results(results)
                    else:
                        st.error(f"Hybrid search failed: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please provide either a text query or an image")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with Streamlit, FastAPI, CLIP, OpenAI, and FAISS</p>
        <p>Multimodal RAG System | Version 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
