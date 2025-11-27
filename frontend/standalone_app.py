"""
Standalone Streamlit Frontend - Demo Mode
Works without backend API - perfect for Streamlit Cloud deployment
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Intelligent Q&A - Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .demo-badge {
        background-color: #ffc107;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load knowledge base
@st.cache_data
def load_knowledge_base() -> List[Dict]:
    """Load knowledge base from JSON file"""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent / "data" / "raw" / "knowledge_base.json",
        Path("data/raw/knowledge_base.json"),
        Path("../data/raw/knowledge_base.json"),
    ]

    for kb_path in possible_paths:
        if kb_path.exists():
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    st.success(f"✓ Loaded knowledge base from {kb_path}")
                    return data
            except Exception as e:
                st.error(f"Error loading {kb_path}: {e}")

    st.error("⚠️ Knowledge base file not found. Please ensure data/raw/knowledge_base.json exists.")
    return []

# Parse knowledge base into searchable format
def parse_knowledge_base(kb: Dict) -> Dict[str, any]:
    """Parse the knowledge base into searchable format"""
    parsed = {
        'personal_info': kb.get('personal_info', {}),
        'keyword_mappings': kb.get('keyword_mappings', {}),
        'qa_pairs': []
    }

    # Flatten comprehensive_qa into a list
    comprehensive_qa = kb.get('comprehensive_qa', {})
    for category, items in comprehensive_qa.items():
        if isinstance(items, list):
            for item in items:
                parsed['qa_pairs'].append({
                    'question': item.get('q', ''),
                    'answer': item.get('a', ''),
                    'keywords': item.get('keywords', []),
                    'category': category
                })

    return parsed

# Simple search function
def simple_search(query: str, parsed_kb: Dict, k: int = 3) -> List[Dict]:
    """Simple keyword-based search"""
    query_lower = query.lower()
    results = []

    # 1. Check keyword mappings first
    for keyword, answer in parsed_kb['keyword_mappings'].items():
        if keyword.lower() in query_lower:
            results.append({
                'answer': answer,
                'source': 'keyword_mapping',
                'relevance': 10  # High relevance for exact keyword match
            })

    # 2. Search Q&A pairs
    for qa in parsed_kb['qa_pairs']:
        score = 0
        # Check question match
        if any(kw in query_lower for kw in qa['keywords']):
            score += 5
        # Check word overlap
        for word in query_lower.split():
            if len(word) > 2:
                if word in qa['question'].lower():
                    score += 2
                if word in qa['answer'].lower():
                    score += 1

        if score > 0:
            results.append({
                'answer': qa['answer'],
                'source': qa['category'],
                'relevance': score
            })

    # Sort by relevance and return top k
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results[:k]

def generate_answer(query: str, search_results: List[Dict]) -> str:
    """Generate answer from search results"""
    if not search_results:
        return "I don't have enough information to answer that question. Try asking about my background, skills, education, projects, or experience!"

    # Combine top answers
    answers = []
    for result in search_results[:3]:
        answer = result['answer']
        if answer and answer not in answers:
            answers.append(answer)

    # Join answers with proper formatting
    if len(answers) == 1:
        return answers[0]
    else:
        return "\n\n".join(answers)

# Header
st.markdown('<h1 class="main-header">💬 Intelligent Q&A System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personal Knowledge Base Q&A - Demo Mode</p>', unsafe_allow_html=True)
st.markdown('<div class="demo-badge">🎭 Demo Mode - No API Required</div>', unsafe_allow_html=True)

# Load and parse knowledge base
knowledge_base = load_knowledge_base()
parsed_kb = parse_knowledge_base(knowledge_base) if knowledge_base else {}

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant answers to retrieve"
    )

    st.markdown("---")

    st.markdown("### 📊 System Status")
    st.success(f"✅ Demo Mode Active")
    if parsed_kb:
        total_qa = len(parsed_kb.get('qa_pairs', []))
        total_keywords = len(parsed_kb.get('keyword_mappings', {}))
        st.info(f"📚 {total_qa} Q&A pairs + {total_keywords} keywords")

    st.markdown("---")

    st.markdown("### ℹ️ About Demo Mode")
    st.info("""
    **This is a standalone demo version:**
    - Works without backend API
    - Uses simple keyword matching
    - Perfect for Streamlit Cloud
    - Shows knowledge base content directly

    **For full AI-powered Q&A:**
    - Deploy backend API separately
    - Use the full version with OpenAI
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💭 Ask Your Question")

    question = st.text_area(
        "Question",
        placeholder="e.g., What technologies are you proficient in? What projects have you worked on?",
        height=100,
        label_visibility="collapsed"
    )

    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    if ask_button and question:
        with st.spinner("Searching knowledge base..."):
            if not parsed_kb:
                st.error("Knowledge base not loaded. Please check the data file.")
            else:
                # Search for relevant information
                search_results = simple_search(question, parsed_kb, k=k)

                if search_results:
                    # Generate answer
                    answer = generate_answer(question, search_results)

                    # Display answer
                    st.markdown("### 💬 Answer")

                    # Format answer with proper line breaks
                    answer_formatted = answer.replace('\n', '<br>')

                    st.markdown(f"""
                    <div class="answer-box">
                        {answer_formatted}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show sources (optional, collapsed)
                    sources = list(set([r['source'] for r in search_results]))
                    if len(sources) <= 3:
                        st.caption(f"📂 Source: {', '.join(sources)}")

                else:
                    st.warning("No relevant information found. Try asking about skills, projects, education, or experience!")

    elif ask_button and not question:
        st.warning("Please enter a question")

with col2:
    st.markdown("### 💡 Suggested Questions")
    st.markdown("Click on questions below:")

    sample_questions = [
        "Who are you?",
        "What technologies are you proficient in?",
        "Tell me about your education",
        "What projects have you worked on?",
        "What are your skills?",
        "How can I contact you?",
        "Tell me about your experience",
        "What is your background?"
    ]

    for sq in sample_questions:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state.question = sq
            st.rerun()

    st.markdown("---")

    st.markdown("### 📊 Quick Stats")
    if parsed_kb:
        # Count categories from Q&A pairs
        categories = {}
        for qa in parsed_kb.get('qa_pairs', []):
            cat = qa.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        total_qa = len(parsed_kb.get('qa_pairs', []))
        st.metric("Q&A Pairs", total_qa)
        st.metric("Categories", len(categories))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🎭 Demo Mode - Standalone Version</p>
    <p>No backend API required | Perfect for Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)

# Tips
with st.expander("💡 About This Demo"):
    st.markdown("""
    **How This Demo Works:**

    - Uses simple keyword matching to find relevant information
    - Retrieves content directly from the knowledge base JSON file
    - No AI model or API calls required
    - Perfect for deployment on Streamlit Cloud (free)

    **Limitations:**

    - Answers are direct excerpts from the knowledge base
    - No AI-generated summaries or reasoning
    - Limited natural language understanding

    **For Full AI-Powered Version:**

    Deploy the backend API and use the full frontend with:
    - OpenAI embeddings for semantic search
    - GPT models for intelligent answer generation
    - Conversation memory
    - Answer verification
    """)
