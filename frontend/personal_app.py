"""
Streamlit Frontend for Personal RAG Q&A System

Clean, professional interface for personal website Q&A
"""

import streamlit as st
import requests
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Intelligent Q&A - Personal Website",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .confidence-high {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .confidence-medium {
        background-color: #ffc107;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .confidence-low {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .sample-question {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        cursor: pointer;
        display: inline-block;
        font-size: 0.9rem;
    }
    .sample-question:hover {
        background-color: #bbdefb;
    }
    .stats-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""


def get_api_health() -> Optional[dict]:
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def ask_question(question: str, k: int = 5, use_verification: bool = False, conversational: bool = False) -> Optional[dict]:
    """Send question to API"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "k": k,
                "use_verification": use_verification,
                "conversational": conversational
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
    return None


def get_sample_questions() -> list:
    """Get sample questions from API"""
    try:
        response = requests.get(f"{API_URL}/sample-questions", timeout=5)
        if response.status_code == 200:
            return response.json().get("questions", [])
    except:
        pass
    return [
        "Who are you?",
        "What technologies are you proficient in?",
        "Tell me about your project experience",
        "How can I contact you?"
    ]


def clear_conversation():
    """Clear conversation history"""
    try:
        requests.post(f"{API_URL}/clear-conversation", timeout=5)
    except:
        pass


def display_confidence_badge(confidence: str):
    """Display confidence level badge"""
    if confidence == "high":
        st.markdown('<span class="confidence-high">High Confidence</span>', unsafe_allow_html=True)
    elif confidence == "medium":
        st.markdown('<span class="confidence-medium">Medium Confidence</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="confidence-low">Low Confidence</span>', unsafe_allow_html=True)


# Header
st.markdown('<h1 class="main-header">💬 Intelligent Q&A System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personal Knowledge Base Q&A powered by RAG - Ask me anything about myself!</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Number of documents to retrieve
    k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More documents may provide more comprehensive answers but may also introduce noise"
    )

    # Verification mode
    use_verification = st.checkbox(
        "Enable answer verification",
        value=False,
        help="Double-check if answer accurately reflects source materials (increases response time)"
    )

    # Conversation mode
    conversation_mode = st.checkbox(
        "Conversation mode",
        value=st.session_state.conversation_mode,
        help="Maintain conversation context for multi-turn dialogue"
    )
    st.session_state.conversation_mode = conversation_mode

    if conversation_mode:
        if st.button("Clear conversation history"):
            clear_conversation()
            st.success("Conversation history cleared")

    st.markdown("---")

    # System status
    st.markdown("### 📊 System Status")
    health = get_api_health()
    if health:
        st.success(f"✅ System Online")
        st.info(f"📚 Loaded {health['documents_loaded']} documents")

        # Show category breakdown
        with st.expander("View document categories"):
            for cat, count in health['categories'].items():
                st.write(f"- {cat}: {count}")
    else:
        st.error("❌ API Offline")
        st.warning("Please ensure API service is running")

    st.markdown("---")

    # About section
    st.markdown("### ℹ️ About")
    st.info("""
    **Tech Stack:**
    - OpenAI Embeddings
    - GPT-3.5/4 LLM
    - FastAPI Backend
    - Streamlit Frontend

    **Anti-Hallucination Strategies:**
    - Low temperature generation
    - Strict prompt engineering
    - Confidence assessment
    - Optional answer verification
    """)


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Question input
    st.markdown("### 💭 Ask Your Question")

    question = st.text_area(
        "Question",
        value=st.session_state.question_input,
        placeholder="e.g., What technologies are you most proficient in? What projects have you worked on?",
        height=100,
        label_visibility="collapsed"
    )

    # Ask button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Process question
    if ask_button and question:
        with st.spinner("Thinking..."):
            result = ask_question(
                question,
                k=k,
                use_verification=use_verification,
                conversational=conversation_mode
            )

        if result:
            # Display answer
            st.markdown("### 🤖 Answer")

            # Confidence badge
            display_confidence_badge(result['confidence'])

            # Answer text
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)

            # Sources
            if result['sources']:
                with st.expander("📚 View Information Sources", expanded=False):
                    for i, source in enumerate(result['sources'], 1):
                        score_color = "#28a745" if source['score'] > 0.7 else "#ffc107" if source['score'] > 0.5 else "#dc3545"
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}</strong> | Type: {source['type']} | Category: {source['category']}
                            <br>
                            <span style="color: {score_color}; font-weight: bold;">Relevance: {source['score']:.2%}</span>
                            <br><br>
                            <em>{source['preview']}</em>
                        </div>
                        """, unsafe_allow_html=True)

            # Retrieval stats
            with st.expander("📈 Retrieval Statistics"):
                avg_score = sum(result['retrieval_scores']) / len(result['retrieval_scores'])
                st.write(f"Average Relevance: {avg_score:.2%}")
                st.write(f"Highest Relevance: {max(result['retrieval_scores']):.2%}")
                st.write(f"Lowest Relevance: {min(result['retrieval_scores']):.2%}")

    elif ask_button and not question:
        st.warning("Please enter a question")

with col2:
    # Sample questions
    st.markdown("### 💡 Suggested Questions")
    st.markdown("Click on questions below for quick queries:")

    sample_questions = get_sample_questions()

    for sq in sample_questions[:8]:  # Show first 8 questions
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state.question_input = sq
            st.rerun()

    st.markdown("---")

    # Quick stats
    st.markdown("### 📊 Quick Stats")

    if health:
        st.metric("Total Documents", health['documents_loaded'])
        st.metric("Document Categories", len(health['categories']))


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🚀 Built with RAG Technology | Using OpenAI Embeddings and GPT Models</p>
    <p>Personal RAG Q&A System v1.0</p>
</div>
""", unsafe_allow_html=True)

# Tips section
with st.expander("💡 Usage Tips"):
    st.markdown("""
    **How to Get the Best Answers:**

    1. **Ask specific questions** - More specific questions yield more accurate answers
    2. **Check confidence level** - High confidence indicates more reliable answers
    3. **View sources** - Understand where the answer information comes from
    4. **Adjust retrieval count** - Modify the number of retrieved documents as needed

    **System Features:**

    - All answers are generated based on personal knowledge base
    - System clearly indicates when it cannot answer a question
    - Multiple strategies employed to prevent AI fabrication
    - Supports conversation mode for multi-turn Q&A
    """)
