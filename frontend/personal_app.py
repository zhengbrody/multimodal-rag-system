"""
Streamlit Frontend for Personal RAG Q&A System

Modern, interactive interface with:
- Conversation-style UI
- Real-time feedback
- Visual analytics
- Theme customization
"""

import streamlit as st
import requests
import time
from typing import Optional, List, Dict
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Intelligent Q&A - Personal Website",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
import os
# Detect if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None or os.getenv("STREAMLIT_SHARING_MODE") == "true"

try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
except (FileNotFoundError, KeyError):
    API_URL = os.getenv("API_URL", "http://localhost:8000")

# For Streamlit Cloud, if API_URL is localhost, show configuration message
if IS_STREAMLIT_CLOUD and API_URL.startswith("http://localhost"):
    USE_MOCK_MODE = True
    MOCK_MODE_MESSAGE = "⚠️ Running in demo mode (no backend API). Configure API_URL in Streamlit Cloud secrets to connect to your backend."
else:
    USE_MOCK_MODE = False
    MOCK_MODE_MESSAGE = None

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Custom CSS with theme support
def get_css(theme='light'):
    if theme == 'dark':
        return """
        <style>
            .stApp {
                background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            }
            .main-header {
                font-size: 2.8rem;
                font-weight: bold;
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .message-user {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 20px 20px 5px 20px;
                margin: 0.5rem 0;
                margin-left: auto;
                max-width: 70%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .message-assistant {
                background: #2d2d44;
                color: #e0e0e0;
                padding: 1rem 1.5rem;
                border-radius: 20px 20px 20px 5px;
                margin: 0.5rem 0;
                max-width: 70%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                border-left: 4px solid #667eea;
            }
        </style>
        """
    else:
        return """
<style>
    .main-header {
                font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
            .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
                padding: 1rem 1.5rem;
                border-radius: 20px 20px 5px 20px;
                margin: 0.5rem 0;
                margin-left: auto;
                max-width: 70%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
            .message-assistant {
                background: #f8f9fa;
                color: #333;
                padding: 1rem 1.5rem;
                border-radius: 20px 20px 20px 5px;
                margin: 0.5rem 0;
                max-width: 70%;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #667eea;
            }
            .confidence-badge {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
        margin: 0.5rem 0;
    }
    .confidence-high {
        background-color: #28a745;
        color: white;
    }
    .confidence-medium {
        background-color: #ffc107;
                color: #333;
    }
    .confidence-low {
        background-color: #dc3545;
        color: white;
            }
            .source-card {
                background: #f8f9fa;
                padding: 0.8rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
                transition: transform 0.2s;
            }
            .source-card:hover {
                transform: translateX(5px);
            }
            .feedback-buttons {
                display: flex;
                gap: 0.5rem;
                margin-top: 1rem;
            }
            .typewriter {
                overflow: hidden;
                border-right: 0.15em solid #667eea;
                white-space: nowrap;
                margin: 0 auto;
                letter-spacing: 0.05em;
                animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
            }
            @keyframes typing {
                from { width: 0; }
                to { width: 100%; }
            }
            @keyframes blink-caret {
                from, to { border-color: transparent; }
                50% { border-color: #667eea; }
            }
            .stats-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
        padding: 1rem;
                border-radius: 15px;
                text-align: center;
                margin: 0.5rem 0;
    }
</style>
        """

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)


def get_api_health() -> Optional[dict]:
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_metrics() -> Optional[dict]:
    """Get system metrics"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def ask_question(question: str, k: int = 5, use_verification: bool = False, conversational: bool = False) -> Optional[dict]:
    """Send question to API"""
    try:
        start_time = time.time()
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
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result['latency'] = latency
            return result
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
    return None


def submit_feedback(question: str, answer: str, rating: int, feedback_text: Optional[str] = None, helpful: Optional[bool] = None):
    """Submit user feedback"""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "question": question,
                "answer": answer,
                "rating": rating,
                "feedback_text": feedback_text,
                "helpful": helpful
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def get_sample_questions() -> list:
    """Get sample questions from API"""
    try:
        response = requests.get(f"{API_URL}/sample-questions", timeout=5)
        if response.status_code == 200:
            return response.json().get("questions", [])
    except:
        pass
    return [
        "Who are you? Give me a brief introduction",
        "What technologies are you proficient in?",
        "Tell me about your proudest project",
        "What is your work experience?",
        "What is your education background?",
        "How can I contact you?",
        "What do you know about RAG systems?",
        "Why did you start writing a technical blog?"
    ]


def clear_conversation():
    """Clear conversation history"""
    try:
        requests.post(f"{API_URL}/clear-conversation", timeout=5)
        st.session_state.conversation_history = []
        st.session_state.current_result = None
        st.success("Conversation cleared!")
    except:
        pass


def display_message(message: str, role: str = "assistant"):
    """Display a message in conversation style"""
    if role == "user":
        st.markdown(f'<div class="message-user">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-assistant">{message}</div>', unsafe_allow_html=True)


def display_confidence(confidence: str):
    """Display confidence badge"""
    conf_class = f"confidence-{confidence}"
    conf_text = confidence.title()
    st.markdown(f'<span class="confidence-badge {conf_class}">🎯 {conf_text} Confidence</span>', unsafe_allow_html=True)


# Header
st.markdown('<h1 class="main-header">💬 Intelligent Q&A System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personal Knowledge Base Q&A powered by RAG with Anti-Hallucination Strategies</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Theme toggle
    theme = st.selectbox("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
    st.session_state.theme = theme
    
    # Number of documents
    k = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More documents = more context, but may introduce noise"
    )

    # Advanced options
    use_verification = st.checkbox(
        "🔍 Enable verification",
        value=False,
        help="Double-check answer accuracy (slower but more reliable)"
    )

    conversation_mode = st.checkbox(
        "💬 Conversation mode",
        value=len(st.session_state.conversation_history) > 0,
        help="Maintain context across questions"
    )
    
    if st.button("🗑️ Clear History", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.markdown("---")

    # System status
    st.markdown("### 📊 System Status")
    health = get_api_health()
    if health:
        st.success("✅ Online")
        st.metric("Documents", health['documents_loaded'])
        
        # Category breakdown
        if health.get('categories'):
            st.markdown("**Categories:**")
            for cat, count in list(health['categories'].items())[:5]:
                st.caption(f"• {cat}: {count}")
    else:
        if USE_MOCK_MODE:
            st.warning("⚠️ Demo Mode")
            st.caption("Backend API not configured. Running in demo mode.")
        else:
            st.error("❌ Offline")
            st.caption("Backend API is not reachable.")
    
    # Metrics
    metrics = get_metrics()
    if metrics:
        st.markdown("---")
        st.markdown("### 📈 Performance")
        st.metric("Avg Latency", f"{metrics.get('average_latency_ms', 0):.0f}ms")
        st.metric("Total Questions", metrics.get('questions_total', 0))
        st.metric("Error Rate", f"{metrics.get('error_rate_percent', 0):.1f}%")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Technologies:**
    - OpenAI Embeddings
    - GPT-3.5/4 LLM
    - FastAPI Backend
    - Streamlit Frontend

    **Anti-Hallucination:**
    - Low temperature (0.3)
    - Strict prompts
    - Confidence scoring
    - Optional verification
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])

with tab1:
    # Conversation display
    if st.session_state.conversation_history:
        st.markdown("### 💭 Conversation")
        for msg in st.session_state.conversation_history[-10:]:  # Show last 10 messages
            display_message(msg['content'], msg['role'])
    
    # Question input
    st.markdown("### ✍️ Ask a Question")
    
    # Sample questions
    sample_questions = get_sample_questions()
    cols = st.columns(4)
    for i, sq in enumerate(sample_questions[:4]):
        with cols[i % 4]:
            if st.button(sq[:30] + "..." if len(sq) > 30 else sq, key=f"sample_{i}", use_container_width=True):
                st.session_state.question_input = sq
                st.rerun()
    
    # Handle input clearing
    if st.session_state.get('clear_input', False):
        question_value = ''
        st.session_state.clear_input = False
        st.session_state.question_input = ''
    else:
        question_value = st.session_state.get('question_input', '')

    question = st.text_area(
        "Your question",
        value=question_value,
        placeholder="e.g., What technologies are you most proficient in?",
        height=100,
        label_visibility="collapsed",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🚀 Ask", type="primary", use_container_width=True)

    # Process question
    if ask_btn and question:
        # Add user message to history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now()
        })
        
        if USE_MOCK_MODE:
            # Demo mode: show a friendly message
            demo_answer = f"""I'm currently running in demo mode without a backend API connection. 

To enable full functionality:
1. Deploy the FastAPI backend (see README.md)
2. Configure the API_URL secret in Streamlit Cloud settings
3. Restart the app

For now, here's a quick answer based on your question: "{question}"

This is a demo of the Personal RAG Q&A System. The full system would retrieve relevant information from a knowledge base and generate accurate answers using RAG (Retrieval-Augmented Generation) technology."""
            
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': demo_answer,
                'timestamp': datetime.now(),
                'confidence': 'medium',
                'sources': []
            })
            display_message(demo_answer, "assistant")
            if MOCK_MODE_MESSAGE:
                st.info(MOCK_MODE_MESSAGE)
        else:
            with st.spinner("🤔 Thinking..."):
                result = ask_question(
                    question,
                    k=k,
                    use_verification=use_verification,
                    conversational=conversation_mode
                )

            if result:
                st.session_state.current_result = result
                
                # Add assistant response to history
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'timestamp': datetime.now(),
                    'confidence': result['confidence'],
                    'sources': result.get('sources', [])
                })
                
                # Display answer
                display_message(result['answer'], "assistant")
                
                # Confidence and metadata
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    display_confidence(result['confidence'])
                with col2:
                    st.caption(f"⚡ {result.get('latency', 0)*1000:.0f}ms")
                with col3:
                    st.caption(f"📚 {len(result.get('sources', []))} sources")
                
                # Sources
                sources = result.get('sources', [])
                if sources:
                with st.expander(f"📖 View Sources ({len(sources)})"):
                    for i, source in enumerate(sources, 1):
                        # Handle both dict and object formats
                        if isinstance(source, dict):
                            score = source.get('score', 0)
                            category = source.get('category', 'unknown')
                            preview = source.get('preview', '')
                        else:
                            # Handle Pydantic model objects
                            score = getattr(source, 'score', 0)
                            category = getattr(source, 'category', 'unknown')
                            preview = getattr(source, 'preview', '')
                        
                        score_pct = score * 100 if isinstance(score, (int, float)) else 0
                        # Clean preview text: remove newlines and limit length
                        preview_text = preview.replace('\n', ' ').strip()
                        preview_text = preview_text[:200] + '...' if len(preview_text) > 200 else preview_text
                        
                        # Escape HTML special characters to prevent XSS
                        preview_text = preview_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        category = str(category).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            st.markdown(f"""
                        <div class="source-card" style="margin-bottom: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                            <strong>Source {i}</strong> ({category}) - {score_pct:.0f}% relevant<br>
                            <small style="color: #666;">{preview_text}</small>
            </div>
            """, unsafe_allow_html=True)

                # Feedback section
                st.markdown("---")
                st.markdown("### 💭 Was this helpful?")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                feedback_rating = None
                with col1:
                    if st.button("👍", use_container_width=True, key="helpful"):
                        feedback_rating = 5
                with col2:
                    if st.button("😊", use_container_width=True, key="good"):
                        feedback_rating = 4
                with col3:
                    if st.button("😐", use_container_width=True, key="ok"):
                        feedback_rating = 3
                with col4:
                    if st.button("😕", use_container_width=True, key="bad"):
                        feedback_rating = 2
                with col5:
                    if st.button("👎", use_container_width=True, key="poor"):
                        feedback_rating = 1
                
                if feedback_rating:
                    if submit_feedback(question, result['answer'], feedback_rating):
                        st.success("Thank you for your feedback! 🙏")
                    else:
                        st.error("Failed to submit feedback")
                
                # Clear input by using a flag instead of modifying session state directly
                if 'clear_input' not in st.session_state:
                    st.session_state.clear_input = False
                st.session_state.clear_input = True
            else:
                if not USE_MOCK_MODE:
                    st.error("Failed to get answer from API. Please check your backend connection.")
    
    elif ask_btn and not question:
        st.warning("Please enter a question")

with tab2:
    st.markdown("### 📊 System Analytics")
    
    health = get_api_health()
    metrics_data = get_metrics()

    if not (health and metrics_data):
        st.warning("Analytics are temporarily unavailable. Make sure the backend API is running.")
    else:
        # Top summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", health.get("documents_loaded", 0))
        with col2:
            st.metric("Categories", len(health.get("categories", {})))
        with col3:
            st.metric("Avg Latency", f"{metrics_data.get('average_latency_ms', 0):.0f}ms")
        with col4:
            st.metric("Error Rate", f"{metrics_data.get('error_rate_percent', 0):.1f}%")

        st.markdown("---")

        # Document categories summary
        categories = health.get("categories", {})
        if categories:
            st.markdown("#### Document Categories")
            sorted_items = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            total_docs = sum(count for _, count in sorted_items) or 1

            table_lines = [
                "| Category | Count | Percentage |",
                "| --- | ---: | ---: |",
            ]
            for cat, count in sorted_items:
                pct = round(count / total_docs * 100, 1)
                table_lines.append(f"| {cat} | {count} | {pct}% |")

            st.markdown("\n".join(table_lines))

        # Conversation + confidence statistics from current session
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("#### Conversation Statistics (Current Session)")
            st.metric("Messages", len(st.session_state.conversation_history))

            confidences = [
                msg.get("confidence")
                for msg in st.session_state.conversation_history
                if msg.get("confidence")
            ]
            if confidences:
                from collections import Counter

                conf_counts = Counter(confidences)
                st.markdown("##### Confidence Distribution")

                levels = ["high", "medium", "low"]
                total_conf = sum(conf_counts.get(lvl, 0) for lvl in levels) or 1

                table_lines = [
                    "| Confidence | Count | Percentage |",
                    "| --- | ---: | ---: |",
                ]
                for lvl in levels:
                    count = conf_counts.get(lvl, 0)
                    if count == 0:
                        continue
                    pct = round(count / total_conf * 100, 1)
                    table_lines.append(f"| {lvl.title()} | {count} | {pct}% |")

                st.markdown("\n".join(table_lines))

with tab3:
    st.markdown("### 🎯 About This System")
    
    st.markdown("""
    #### 🧠 RAG Pipeline Architecture
    
    This system implements a **Retrieval-Augmented Generation (RAG)** pipeline specifically designed for personal Q&A:
    
    1. **Knowledge Base Construction** 📚
       - Structured personal data (projects, experience, skills, etc.)
       - Document chunking and metadata tagging
       - Category-based organization
    
    2. **Semantic Retrieval** 🔍
       - OpenAI embeddings (text-embedding-3-small)
       - Cosine similarity search
       - Category-weighted scoring
    
    3. **Prompt Engineering** ✍️
       - Low temperature (0.3) for factual responses
       - Strict system prompts preventing fabrication
       - Context-aware question answering
    
    4. **Anti-Hallucination Strategies** 🛡️
       - **Low Temperature**: Reduces randomness, increases determinism
       - **Strict Prompts**: Explicit instructions to only use provided context
       - **Confidence Assessment**: Based on retrieval scores
       - **Source Tracing**: Show where information comes from
       - **Optional Verification**: Second-pass fact-checking
    
    #### 🚀 Key Features
    
    - **Conversation Mode**: Multi-turn dialogue with context retention
    - **Confidence Scoring**: High/Medium/Low based on retrieval quality
    - **Source Attribution**: See exactly where answers come from
    - **Feedback Collection**: Help improve the system
    - **Real-time Metrics**: Monitor system performance
    
    #### 💡 Best Practices
    
    - Ask specific questions for better results
    - Check confidence levels for reliability
    - Review sources to verify information
    - Use conversation mode for follow-up questions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🚀 Built with RAG Technology | Anti-Hallucination Strategies | FastAPI + Streamlit</p>
    <p>Personal RAG Q&A System v2.0</p>
</div>
""", unsafe_allow_html=True)
    