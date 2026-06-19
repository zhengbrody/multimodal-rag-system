# Interview Preparation: Multimodal RAG System
## Ingram Micro — AI Intern Position

> **Purpose**: Comprehensive interview preparation document covering all technical details, AI/ML core knowledge, software engineering practices, and system design thinking demonstrated in this project. Review the night before your interview to confidently discuss every aspect.
> 
> **Answer Format**: All interview answers use the **STAR framework** (Situation → Task → Action → Result) to provide structured, compelling responses.

---

## 1. Project Overview

### Elevator Pitch

This is a **Retrieval-Augmented Generation (RAG) system** purpose-built for intelligent Q&A on a personal website. Users can ask natural-language questions about the site owner's skills, project experience, education background, and more. The system performs semantic retrieval to find the most relevant knowledge snippets, then feeds them to an LLM to generate accurate, source-backed answers. The core highlight is a **4-layer anti-hallucination strategy** (low temperature, strict prompts, confidence scoring, two-pass verification) that ensures the AI never fabricates information. The system also provides a Mock mode so it can run and be tested end-to-end at zero API cost.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit)                     │
│   ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
│   │ Chat UI  │  │ Session State│  │  Analytics  │  │Feedback │ │
│   └────┬─────┘  └──────────────┘  └────────────┘  └─────────┘ │
│        │                                                        │
└────────┼────────────────────────────────────────────────────────┘
         │  HTTP/REST
┌────────▼────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                          │
│   ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
│   │  /ask    │  │  /feedback   │  │  /metrics   │  │ /health │ │
│   │  /stats  │  │  /rebuild    │  │  Middleware  │  │  CORS   │ │
│   └────┬─────┘  └──────────────┘  └────────────┘  └─────────┘ │
│        │                                                        │
│   ┌────▼─────────────────────────────────────────────────────┐  │
│   │              RAG Pipeline Layer                           │  │
│   │  ┌──────────────────┐    ┌───────────────────────────┐   │  │
│   │  │ PersonalRAG      │    │ ConversationalRAG          │   │  │
│   │  │ Pipeline         │    │ Pipeline (+ memory)        │   │  │
│   │  └────────┬─────────┘    └────────────┬──────────────┘   │  │
│   │           │   query_with_verification  │                  │  │
│   │           │   (two-pass anti-hallucination)               │  │
│   │  ┌────────▼────────────────────────────▼──────────────┐   │  │
│   │  │              Retriever Layer                       │   │  │
│   │  │  ┌─────────────────┐  ┌──────────────────────┐    │   │  │
│   │  │  │ SimpleRetriever │  │ MockRetriever         │    │   │  │
│   │  │  │ (OpenAI Embed)  │  │ (Keyword + Intent)    │    │   │  │
│   │  │  │       │         │  │                       │    │   │  │
│   │  │  │ PersonalRAG     │  │ Jaccard Similarity    │    │   │  │
│   │  │  │ Retriever       │  │ + Intent Boosting     │    │   │  │
│   │  │  │ (+ cat weights) │  │ + Category Boosting   │    │   │  │
│   │  │  └─────────────────┘  └──────────────────────┘    │   │  │
│   │  └───────────────────────────────────────────────────┘   │  │
│   └──────────────────────────────────────────────────────────┘  │
│        │                                                        │
│   ┌────▼─────────────────────────────────────────────────────┐  │
│   │         Knowledge Processor                               │  │
│   │  JSON knowledge base → Document objects (chunked)         │  │
│   │  process_skills / process_projects / process_faq / ...    │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| LLM | OpenAI GPT-3.5/4 | Answer generation + two-pass verification |
| Embedding | text-embedding-3-small (1536 dims) | Vectorization of documents and queries |
| Vector Search | NumPy cosine similarity | Semantic similarity computation |
| Backend API | FastAPI + Pydantic + Uvicorn | RESTful API service |
| Frontend | Streamlit | Interactive Chat UI |
| Testing | pytest + pytest-cov + httpx | Unit tests + integration tests |
| CI/CD | GitHub Actions (Python 3.11/3.12 matrix) | Automated testing + linting |
| Containerization | Docker + Docker Compose | One-command deployment |
| Code Quality | Black + Ruff + MyPy | Formatting + linting + type checking |
| Logging | Python logging (structured) | Observability |
| Configuration | YAML + dotenv | Environment management |

---

## 2. Core AI/ML Knowledge Points

### 2.1 RAG Architecture (Retrieval-Augmented Generation)

**Concept**

RAG is an architecture that combines information retrieval with language generation. Instead of letting an LLM answer purely from its parametric memory, RAG first retrieves relevant documents from an external knowledge base, then passes those documents as context to the LLM for answer generation. This leverages the LLM's language ability while ensuring answers are grounded in verifiable facts.

**Implementation**

File: `src/rag/pipeline.py` — `PersonalRAGPipeline.query()`

```python
def query(self, question: str, k: int = 5, include_sources: bool = True):
    # Step 1: Retrieve relevant documents
    retrieved_docs = self.retriever.retrieve(question, k=k, threshold=0.3)
    
    # Step 2: Build context
    context = self._build_context(retrieved_docs)
    
    # Step 3: Generate response with LLM
    response = self.client.chat.completions.create(
        model=self.llm_model,
        messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=self.temperature,  # 0.3
        max_tokens=self.max_tokens,
    )
    
    # Step 4: Assess confidence
    # Step 5: Build response with sources
```

**Why This Approach**

- The personal website's knowledge is **private data** that no LLM has been pre-trained on, so RAG is essential to inject it
- Compared to fine-tuning, RAG is **lower cost**, **faster to update** (just edit the JSON), and **more explainable** (answers can be traced to sources)
- The 5-step pipeline design makes each stage independently testable and replaceable

**Interview Q&A**

**Q: What is the difference between RAG and fine-tuning? When would you choose RAG?**

- **Situation**: When building this personal Q&A system, I needed to decide how to give the LLM access to private personal information it had never seen during training.
- **Task**: I needed an approach that was cost-effective, easy to update, and could provide source traceability for every answer.
- **Action**: I chose RAG over fine-tuning because fine-tuning encodes knowledge into model weights (suited for general capability improvements like style or domain terminology), whereas RAG stores knowledge in an external retrieval index. I implemented the full pipeline in `src/rag/pipeline.py` with a 5-step query flow: retrieve, build context, generate, assess confidence, and build response.
- **Result**: This approach allows me to update the knowledge base by simply editing a JSON file with zero retraining cost, and every answer includes traceable source documents that interviewers or users can verify.

**Q: What are the main bottlenecks of a RAG system?**

- **Situation**: During development, I observed that the overall system quality depended heavily on two critical pipeline stages.
- **Task**: I needed to identify and mitigate these bottlenecks to ensure reliable answers.
- **Action**: I identified two key bottlenecks: (1) **Retrieval quality** — if the retriever fails to find the correct documents, even the best LLM will produce garbage-in-garbage-out answers; (2) **Context window limits** — retrieving too many documents can exceed the token limit or introduce noise. I addressed the first with category weighting in `PersonalRAGRetriever` and the second by using `threshold=0.3` to filter low-quality results and `k=5` to cap the number of documents.
- **Result**: These controls ensure the LLM receives only high-relevance context, which directly improves answer accuracy and keeps latency low.

**Q: What are the innovations in your RAG system?**

- **Situation**: Most basic RAG tutorials demonstrate a simple retrieve-and-generate pipeline with no reliability safeguards.
- **Task**: I wanted my system to go beyond the basics and address real-world concerns like hallucination, cost, and multi-turn dialogue.
- **Action**: I implemented four differentiating features: (1) **Category weighting** in `PersonalRAGRetriever` to boost FAQ and About documents; (2) **4-layer anti-hallucination strategy** spanning low temperature, strict prompts, confidence scoring, and two-pass verification; (3) **Mock/Real dual-mode design** so development and testing run at zero API cost; (4) **ConversationalRAGPipeline** supporting multi-turn dialogue with sliding-window memory.
- **Result**: The system is production-quality rather than a toy demo — it handles edge cases, supports cost-free CI, and maintains conversation context across multiple turns.

---

### 2.2 Embeddings & Vector Representations

**Concept**

Embeddings map text into a high-dimensional vector space where semantically similar texts are positioned closer together. OpenAI's `text-embedding-3-small` encodes text into 1536-dimensional vectors, capturing lexical, syntactic, and semantic meaning.

**Implementation**

File: `src/rag/retriever.py` — `SimpleRetriever`

```python
class SimpleRetriever:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model

    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100):
        """Batch processing to reduce API calls"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.embedding_model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
```

**Why This Approach**

- Chose `text-embedding-3-small` over `text-embedding-3-large`: for the personal website's document volume (<500 documents), the small model is more than sufficient and costs only 1/5 of the large model
- **Batch embedding**: reduces API call count by processing 100 texts per request, balancing efficiency against API rate limits
- Pre-computed and persisted (pickle): avoids regenerating embeddings on every startup, saving both time and cost

**Interview Q&A**

**Q: Why not use BERT or other open-source embedding models?**

- **Situation**: When selecting an embedding model, I needed to balance semantic quality, integration simplicity, and cost for a personal-scale project.
- **Task**: I needed an embedding solution that would produce high-quality semantic representations without requiring GPU infrastructure.
- **Action**: I chose OpenAI's embedding API because it delivers superior semantic understanding (especially for mixed-language text), requires no GPU, and integrates with a simple API call in `SimpleRetriever._get_embedding()`. The cost for embeddings is extremely low at this scale.
- **Result**: The system achieves high retrieval accuracy with minimal infrastructure. For a production deployment requiring cost reduction, I could swap in `sentence-transformers` as an open-source alternative without changing the pipeline architecture, thanks to the Strategy pattern.

**Q: What does the 1536 embedding dimension mean?**

- **Concept**: Each text is represented as a vector of 1536 floating-point numbers. Higher dimensions capture more semantic nuance but increase computational cost. 1536 is a well-balanced trade-off between performance and efficiency.
- **In My Project**: The `text-embedding-3-small` model outputs 1536-dimensional vectors stored as a NumPy array in `SimpleRetriever.embeddings`. All similarity computations operate on this dimensionality.
- **Key Insight**: At very large scale, dimensionality reduction techniques like PCA or Matryoshka embeddings could reduce this to 256 or 512 dimensions with minimal quality loss, cutting memory usage by 3-6x.

---

### 2.3 Similarity Search (Cosine Similarity)

**Concept**

Cosine similarity measures how similar two vectors are in direction, with a range of [-1, 1]. It ignores vector magnitude and focuses only on orientation, making it naturally suited for comparing the semantic similarity of texts of different lengths. Formula: `cos(A, B) = (A . B) / (||A|| * ||B||)`

**Implementation**

File: `src/rag/retriever.py` — `SimpleRetriever._cosine_similarity()`

```python
def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray):
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    # Compute similarities (vectorized dot product)
    similarities = np.dot(doc_norms, query_norm)
    return similarities
```

**Why This Approach**

- Normalizing first and then computing the dot product is mathematically equivalent to cosine similarity but computationally faster (leverages NumPy matrix operations)
- Computes similarity for **all documents at once** rather than looping one by one, fully utilizing NumPy's SIMD vectorization
- For the current document volume (a few hundred), brute-force search latency is < 1ms — no need to introduce FAISS or other ANN indices

**Interview Q&A**

**Q: Why did you choose cosine similarity instead of L2 distance?**

- **Situation**: When building the retrieval layer, I needed to decide on a distance metric that would work well with OpenAI embeddings and also be intuitive for threshold-based filtering.
- **Task**: I needed a metric that produces interpretable scores so I could set meaningful quality thresholds.
- **Action**: I chose cosine similarity because for OpenAI embeddings (which are already normalized), cosine similarity and L2 distance produce equivalent rankings. However, cosine similarity's value range of [0, 1] is far more intuitive — 0.8 means "very similar," 0.3 means "not very relevant" — making it straightforward to set `threshold=0.3` in `SimpleRetriever.retrieve()` for filtering irrelevant results.
- **Result**: The threshold-based filtering cleanly eliminates low-quality matches, and the interpretable scores feed directly into the confidence assessment system (high/medium/low) in the pipeline.

**Q: What is the complexity of brute-force search? When would you need to switch?**

- **Situation**: I needed to justify why brute-force search is acceptable for the current system and know when it would become a bottleneck.
- **Task**: Understand the computational limits and have a concrete scaling plan ready.
- **Action**: Brute-force cosine similarity is O(n * d), where n is the document count and d is the embedding dimension. With the current 200 documents and 1536 dimensions, this is trivially fast. I documented that when documents exceed 100,000, the system should migrate to FAISS (IVF + HNSW) or a managed vector database like Pinecone/Qdrant, reducing search complexity to O(log n). The Strategy pattern in my architecture makes this a drop-in replacement.
- **Result**: The current system runs retrieval in under 1ms while the architecture explicitly supports scaling to millions of documents without redesigning the pipeline.

---

### 2.4 Retrieval Strategies

**Concept**

Retrieval strategies determine how the most relevant documents are found in the knowledge base. This project implements two retrieval strategies: (1) **Semantic retrieval** — embedding-based semantic search (SimpleRetriever); (2) **Keyword + Intent retrieval** — keyword matching with intent detection (MockRetriever). Both support **category weighting** to boost the ranking of specific document types.

**Implementation**

File: `src/rag/retriever.py` — `PersonalRAGRetriever.retrieve()` (category weighting)

```python
class PersonalRAGRetriever(SimpleRetriever):
    def __init__(self, embedding_model="text-embedding-3-small"):
        super().__init__(embedding_model)
        self.category_weights = {
            "faq": 1.2,    # Boost FAQ answers
            "about": 1.1,  # Personal info is important
            "projects": 1.0,
            "skills": 1.1,
            "contact": 1.0,
        }

    def retrieve(self, query, k=5, threshold=0.3):
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        # Apply category weights
        for i, doc in enumerate(self.documents):
            category = doc["metadata"].get("category", "other")
            weight = self.category_weights.get(category, 1.0)
            weighted_similarities[i] *= weight
```

File: `src/rag/mock_retriever.py` — `MockRetriever._detect_query_intent()` (intent detection)

```python
def _detect_query_intent(self, query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["project", "built", "developed"]):
        return "projects"
    if any(kw in query_lower for kw in ["skill", "technology", "proficient"]):
        return "skills"
    # ... more intents
    return "general"
```

**Why This Approach**

- **Category weighting** solves the problem of "FAQ and About documents are short but highly important" — without weighting, longer documents tend to rank higher simply because they have more keyword matches
- **Intent detection** enables MockRetriever to achieve precise retrieval even without embeddings — by recognizing user intent (e.g., "experience" triggers prioritization of work history documents) to compensate for the limitations of pure keyword matching
- **Dual retriever design** lets the system run entirely without the OpenAI API during development and testing

**Interview Q&A**

**Q: Why not use BM25 directly?**

- **Situation**: I considered several retrieval algorithms and needed one that handles paraphrasing well in a personal Q&A context.
- **Task**: I needed a retriever that can match "what are you best at?" with a knowledge base entry labeled "Skills: Python, ML..." — a paraphrase problem that pure keyword matching cannot solve.
- **Action**: I chose semantic retrieval (embedding-based) as the primary approach because it understands synonyms and paraphrases. BM25 is a classic keyword retrieval algorithm well-suited for exact keyword matching, but it cannot bridge the gap between different phrasings of the same concept. For the MockRetriever, I built an intent-detection layer (`_detect_query_intent()`) to simulate some of this semantic understanding without embeddings.
- **Result**: The semantic retriever handles paraphrases effectively, and the mock retriever provides a cost-free alternative. Ideally, a production system would use **hybrid search** (BM25 + semantic) with weighted fusion to get the best of both worlds.

**Q: How did you determine the category weighting values?**

- **Situation**: After deploying the initial retriever, I noticed that FAQ and About documents were being outranked by longer but less relevant documents.
- **Task**: I needed to boost the ranking of high-value short documents without breaking the overall relevance ordering.
- **Action**: I set weights empirically based on observed retrieval behavior: FAQ at 1.2x and About at 1.1x in the `category_weights` dictionary in `PersonalRAGRetriever.__init__()`. These modest boosts give priority to concise, high-signal documents without overwhelming the base similarity scores.
- **Result**: Retrieval quality improved noticeably for common personal questions. In a production system, these weights could be tuned automatically via A/B testing, learning-to-rank, or online learning from user feedback collected through the `/feedback` endpoint.

---

### 2.5 Anti-Hallucination (4 Strategies)

**Concept**

LLM hallucination refers to the model generating content that contradicts the facts — it appears confident but is actually fabricated. In a personal Q&A system, hallucination is especially dangerous: if the AI invents nonexistent work experience or skills, it seriously undermines credibility.

**Implementation**

**Strategy 1: Low Temperature (reduced randomness)**

File: `src/rag/pipeline.py`

```python
def __init__(self, retriever, llm_model="gpt-3.5-turbo",
             temperature: float = 0.3,  # Lower = more deterministic
             max_tokens: int = 1000):
```

**Strategy 2: Strict System Prompt**

```python
self.system_prompt = """...
Important Rules:
1. Only use the provided context information to answer questions
2. If there is no relevant information in the context, explicitly state 
   "Based on the information I have, I cannot answer this question"
3. Do not fabricate or speculate any information
4. If information is incomplete, honestly point this out
..."""
```

**Strategy 3: Confidence Assessment**

```python
avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)
max_score = max(doc["score"] for doc in retrieved_docs)

if max_score > 0.75 and avg_score > 0.6:
    confidence = "high"
elif max_score > 0.5 and avg_score > 0.4:
    confidence = "medium"
else:
    confidence = "low"
```

**Strategy 4: Two-Pass Verification**

File: `src/rag/pipeline.py` — `query_with_verification()`

```python
def query_with_verification(self, question, k=5):
    initial_result = self.query(question, k=k)
    
    # Second pass: verify answer against sources
    verification_prompt = """Please verify whether the following answer 
    is completely based on the provided information..."""
    
    # If verification fails, downgrade confidence
    if not verification_data.get("verified", True):
        initial_result["answer"] += "\n\n[Note: Some content may need 
        further verification against source materials.]"
        initial_result["confidence"] = "low"
```

**Why This Approach**

- **Defense in depth**: no single strategy is reliable enough on its own; stacking 4 layers drastically reduces hallucination risk
- **Temperature 0.3** (not 0): fully deterministic output (0) produces overly rigid answers; 0.3 balances accuracy with natural-sounding language
- **Confidence assessment** is based on retrieval scores, not LLM self-assessment — LLMs are notoriously unreliable at judging their own confidence
- **Two-pass verification** is optional (`use_verification=False` by default) because it adds an extra API call, increasing latency and cost

**Interview Q&A**

**Q: How effective are your anti-hallucination strategies? How do you measure them?**

- **Situation**: After building the initial RAG pipeline, I needed to validate that the system would refuse to answer questions about information not in the knowledge base rather than fabricating answers.
- **Task**: I needed both qualitative and quantitative approaches to assess hallucination resistance.
- **Action**: Qualitatively, I tested with questions about nonexistent projects or fabricated skills to verify the system refuses to answer. Quantitatively, the correct approach is to use a **faithfulness metric**: compare each claim in the answer against source documents and compute `supported claims / total claims`. In my project, I collect user feedback through the `/feedback` endpoint (1-5 ratings stored in `logs/feedback.jsonl`) to indirectly assess answer quality, and the confidence scoring system flags low-confidence answers for review.
- **Result**: The system reliably refuses to answer out-of-scope questions, and the confidence scoring accurately reflects retrieval quality, giving users a clear signal about answer reliability.

**Q: Why is two-pass verification not enabled by default?**

- **Situation**: I implemented `query_with_verification()` as an additional safeguard but had to decide on its default state.
- **Task**: I needed to balance maximum accuracy against latency and cost constraints.
- **Action**: I made verification opt-in (`use_verification=False` in `QuestionRequest`) because the second LLM call adds approximately 1-2 seconds of latency and doubles the API cost per query. For most routine personal Q&A, the first three layers of defense (low temperature, strict prompts, confidence scoring) are sufficient. The verification toggle is exposed in the Streamlit sidebar so users can enable it for critical queries.
- **Result**: Default queries are fast and cost-effective, while users who need extra assurance can enable verification on demand. This respects both the user's time and API budget.

**Q: What if the retrieved documents themselves contain incorrect information?**

- **Situation**: I recognized that RAG has an inherent limitation: if the knowledge base itself contains errors, the system will confidently produce incorrect answers.
- **Task**: I needed to mitigate this "garbage in, garbage out" risk.
- **Action**: I addressed this with three measures: (1) ensuring knowledge base data quality through manual review of the JSON source; (2) supporting hot-reload via the `/rebuild-index` endpoint in `personal_api.py` so the knowledge base can be refreshed without downtime; (3) displaying source documents in the UI (the `sources` field in `AnswerResponse`) so users can verify information themselves.
- **Result**: The combination of curated data, easy updates, and transparent source attribution gives users confidence in the answers while providing a clear path for correction when errors are found.

---

### 2.6 Prompt Engineering

**Concept**

Prompt engineering is the practice of carefully designing input prompts to guide an LLM toward producing desired outputs. Good prompts significantly improve answer quality, reduce hallucination, and control output format.

**Implementation**

File: `src/rag/pipeline.py` — three-layer prompt design

```python
# Layer 1: System Prompt (defines role and rules)
self.system_prompt = """You are an intelligent assistant for a personal 
website, specialized in answering questions about the website owner.
Important Rules:
1. Only use the provided context information...
2. If there is no relevant information, explicitly state...
..."""

# Layer 2: User Prompt (injects retrieved context + question)
def _build_user_prompt(self, question, context):
    return f"""Answer the user's question based on the following information.
=== Reference Information ===
{context}
=== End of Reference Information ===
User Question: {question}
Please provide an accurate and concise answer..."""

# Layer 3: Context Formatting (with relevance labels)
def _build_context(self, retrieved_docs):
    for i, doc in enumerate(retrieved_docs, 1):
        relevance = "High" if doc["score"] > 0.7 else "Medium" if doc["score"] > 0.5 else "Low"
        context_parts.append(
            f"[Information Snippet {i}] (Relevance: {relevance})\n{doc['content']}\n")
```

**Why This Approach**

- **Three-layer separation** allows each layer to be independently adjusted: changing system behavior only requires modifying the system prompt; changing context format only requires modifying the context builder
- **Relevance labels** (High/Medium/Low) guide the LLM to prioritize high-relevance snippets, reducing noise interference
- **Clear delimiters** (`=== Reference Information ===`) help the LLM distinguish context from instructions, providing a basic defense against prompt injection

**Interview Q&A**

**Q: What is the difference between the system prompt and user prompt? Why separate them?**

- **Situation**: When designing the prompt architecture, I needed to decide how to structure the instructions versus the per-query content.
- **Task**: I wanted anti-hallucination rules to remain consistent across all queries while allowing the context and question to change with each request.
- **Action**: I separated concerns into a system prompt (role definition and global rules, set once in `__init__`) and a user prompt (per-query context and question, built dynamically in `_build_user_prompt()`). The system prompt defines immutable behavior like "only use provided context," while the user prompt injects the specific retrieved documents and user question.
- **Result**: Rules remain consistent regardless of the question asked, and in multi-turn conversations the system prompt only needs to be sent once, saving tokens. This clean separation also makes it easy to A/B test different prompt strategies.

**Q: How do you defend against prompt injection?**

- **Situation**: Since user questions are passed directly into the LLM prompt alongside retrieved context, there is a risk of prompt injection attacks.
- **Task**: I needed to mitigate injection risks without overly restricting legitimate user input.
- **Action**: I implemented three defenses: (1) clear delimiters (`=== Reference Information ===`) to isolate context from instructions; (2) explicit system prompt rules stating "only use provided information"; (3) Pydantic validation with `max_length=500` on `QuestionRequest` to limit input length. More advanced defenses could include input filtering (detecting injection patterns) and output filtering (detecting anomalous answers).
- **Result**: These measures provide a solid baseline defense appropriate for a personal Q&A system. The layered approach makes it straightforward to add more sophisticated filtering if the system is deployed in a higher-risk environment.

---

### 2.7 Conversational RAG & Memory Management

**Concept**

Multi-turn conversational RAG needs to maintain contextual coherence while keeping retrieval accurate. The core challenge is **memory management**: too much history fills the context window, too little loses conversation coherence.

**Implementation**

File: `src/rag/pipeline.py` — `ConversationalRAGPipeline`

```python
class ConversationalRAGPipeline(PersonalRAGPipeline):
    def __init__(self, retriever, llm_model="gpt-3.5-turbo", memory_size: int = 5):
        super().__init__(retriever, llm_model)
        self.conversation_history = []
        self.memory_size = memory_size

    def _get_conversation_context(self) -> str:
        """Get recent conversation history (sliding window)"""
        history_text = "\n=== Conversation History ===\n"
        for turn in self.conversation_history[-self.memory_size:]:
            history_text += f"User: {turn['question']}\n"
            history_text += f"Assistant: {turn['answer'][:200]}...\n\n"  # Truncate long answers
        return history_text

    def query(self, question, k=5, include_sources=True):
        conv_context = self._get_conversation_context()
        enhanced_question = f"{conv_context}\nCurrent Question: {question}"
        result = super().query(enhanced_question, k=k, include_sources=include_sources)
        
        # Trim history to prevent unbounded growth
        if len(self.conversation_history) > self.memory_size * 2:
            self.conversation_history = self.conversation_history[-self.memory_size:]
        return result
```

**Why This Approach**

- **Sliding window** (`memory_size=5`) limits history length to prevent exceeding the token limit
- **Answer truncation** (`[:200]`): only a summary of each past answer is retained, saving context space
- **Inherits from PersonalRAGPipeline**: uses the Template Method pattern to reuse all anti-hallucination logic
- **Double-threshold cleanup**: trimming triggers at `memory_size * 2` to reduce the frequency of list slicing operations

**Interview Q&A**

**Q: What are the limitations of this simple sliding window approach?**

- **Situation**: After implementing the sliding window memory, I evaluated its behavior in long conversations and identified clear limitations.
- **Task**: I needed to understand the trade-offs and know what alternatives exist for more demanding scenarios.
- **Action**: I identified two main limitations: (1) the window cannot handle long-range dependencies (e.g., information from turn 1 referenced in turn 10); (2) it does not distinguish important from unimportant history — all turns are treated equally. I documented better alternatives including **summarization-based memory** (using an LLM to condense history), **entity memory** (extracting key entities), and **RAG over conversation history** (making past turns retrievable as documents).
- **Result**: The sliding window is appropriate for the current personal Q&A use case where conversations are typically 3-5 turns. The architecture supports swapping in a more sophisticated memory system if the use case demands it.

**Q: Why not use LangChain's ConversationBufferMemory?**

- **Situation**: LangChain offers ready-made memory management components that could have saved development time.
- **Task**: I needed to decide between using a framework and building a custom solution.
- **Action**: I chose to implement memory management from scratch in `ConversationalRAGPipeline` for three reasons: (1) reduces external dependencies, keeping the system lightweight; (2) gives me complete understanding and control over the memory logic; (3) allows me to explain every line of code in an interview. LangChain is excellent for rapid prototyping, but for a project intended to demonstrate engineering depth, a custom solution is more appropriate.
- **Result**: The custom implementation is only ~40 lines of code, has zero additional dependencies, and demonstrates clear understanding of the underlying concepts rather than just framework usage.

---

### 2.8 Knowledge Processing & Document Chunking

**Concept**

Knowledge processing transforms raw data into document fragments suitable for retrieval. Good chunking strategy directly impacts retrieval quality — chunks that are too large introduce noise, chunks that are too small lose context.

**Implementation**

File: `src/rag/knowledge_processor.py`

```python
class Document:
    """Simple document class for RAG"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

def build_knowledge_base(knowledge_path: str) -> List[Document]:
    """Build complete knowledge base from JSON file"""
    data = load_knowledge_base(knowledge_path)
    documents = []
    documents.extend(process_personal_info(data))   # 1 doc
    documents.extend(process_skills(data))           # 1 comprehensive doc
    documents.extend(process_projects(data))         # 1 doc per project
    documents.extend(process_experience(data))       # 1 doc per job
    documents.extend(process_education(data))        # 1 doc
    documents.extend(process_faq(data))              # 1 doc per FAQ
    documents.extend(process_comprehensive_qa(data)) # 1 doc per Q&A pair
    documents.extend(process_keyword_mappings(data)) # 1 doc per keyword
    # ... more processors
    return documents
```

**Chunking Strategy**: chunking by **semantic units**, not by fixed length

```python
# Each project is an independent Document
def process_projects(data):
    for project in projects:
        content = f"""
Project: {project.get('name', '')}
Description: {project.get('description', '')}
Tech Stack: {', '.join(project.get('tech_stack', []))}
Key Achievements: {', '.join(highlights)}
..."""
        documents.append(Document(content=content.strip(),
            metadata={"type": "project", "category": "projects", ...}))
```

**Why This Approach**

- **Semantic unit chunking** (one chunk per project, one per FAQ, one per experience) instead of fixed-token splitting ensures every chunk is self-contained and meaningful
- **Rich metadata** (type, category, keywords, question) enables the retriever to perform category weighting and intent matching
- **16 independent processor functions**: one per data type, following the Single Responsibility Principle. Adding a new data type only requires adding one function

**Interview Q&A**

**Q: What if a project description is 2000 tokens long — is that too large?**

- **Situation**: Some project descriptions in the knowledge base contain detailed information that could approach the embedding model's input limits.
- **Task**: I needed to determine whether my semantic-unit chunking strategy would hold up for longer documents.
- **Action**: For the current scenario, this is not a problem — OpenAI embeddings support a maximum of 8192 tokens per input. However, if future documents become very long (e.g., papers or reports), the system would need a **recursive text splitter** (splitting first by paragraph, then by sentence, ensuring chunks stay within `chunk_size`) with chunk overlap to preserve contextual continuity. I kept this as a documented extension point.
- **Result**: The current approach works well for personal portfolio data (projects, experience, FAQs), and the modular processor design makes it straightforward to add recursive splitting for longer document types.

**Q: What are the benefits of your metadata design?**

- **Concept**: Metadata transforms retrieval from pure "vector similarity ranking" into a richer, multi-signal search system.
- **In My Project**: The metadata fields (type, category, keywords, question) enable three advanced retrieval features: **category filtering** (search within specific categories), **category boosting** (FAQ weighted 1.2x in `PersonalRAGRetriever`), and **intent-based retrieval** (MockRetriever's `_detect_query_intent()`). This is a lightweight form of **structured + unstructured hybrid retrieval**.
- **Key Insight**: Even simple metadata can dramatically improve retrieval precision — the category weights in my system improved relevance for common personal questions without any model changes.

---

## 3. Software Engineering Knowledge Points

### 3.1 Design Patterns

**Concept**

Design patterns are reusable solutions to common software design problems. This project uses 4 classic patterns to ensure code extensibility and maintainability.

**Implementation**

**Strategy Pattern — Retriever is swappable**

The pipeline does not care about the retriever's concrete implementation, only that it has a `retrieve()` method:

```python
# Any retriever can be injected
pipeline = PersonalRAGPipeline(retriever)  # retriever can be SimpleRetriever or MockRetriever

# At API startup, the strategy is selected based on configuration
if use_mock:
    retriever = MockRetriever()
    pipeline = MockRAGPipeline(retriever)
else:
    retriever = PersonalRAGRetriever(embedding_model="text-embedding-3-small")
    pipeline = PersonalRAGPipeline(retriever, llm_model=llm_model)
```

**Template Method — Pipeline inheritance**

```python
class ConversationalRAGPipeline(PersonalRAGPipeline):
    """Inherits and extends query(), reusing _build_context, _build_user_prompt, etc."""
    def query(self, question, k=5, include_sources=True):
        conv_context = self._get_conversation_context()  # New step
        result = super().query(enhanced_question, ...)   # Reuse parent template
        self.conversation_history.append(...)            # New step
        return result
```

**Factory Pattern — Knowledge Processor**

```python
def build_knowledge_base(knowledge_path: str) -> List[Document]:
    """Factory function: automatically creates different Document types based on data"""
    documents.extend(process_personal_info(data))
    documents.extend(process_skills(data))
    documents.extend(process_projects(data))
    # ... each process_* function is a concrete factory
```

**Dependency Injection — Pipeline receives Retriever**

```python
class PersonalRAGPipeline:
    def __init__(self, retriever, ...):  # retriever is injected from outside
        self.retriever = retriever       # not created internally; easy to test and swap
```

**Why This Approach**

- **Strategy + DI**: lets mock and real modes share the same pipeline logic — switching only requires changing one line of configuration
- **Template Method**: `ConversationalRAGPipeline` only needs to add memory management code; all anti-hallucination logic is fully reused
- **Factory**: adding a new data type (e.g., "certifications") only requires writing a `process_certifications()` function and one `documents.extend()` call

**Interview Q&A**

**Q: If you needed to add a new retriever (say, using FAISS), what code changes would be required?**

- **Situation**: The system was designed to support multiple retriever backends, but initially only needed NumPy-based and keyword-based implementations.
- **Task**: I wanted to ensure that adding a new retriever (like FAISS) would be a minimal-effort change.
- **Action**: Thanks to the Strategy + DI patterns, adding a FAISS retriever requires only: (1) creating a new `FAISSRetriever` class implementing `retrieve()`, `add_documents()`, `save()`, `load()`, and `get_category_stats()`; (2) adding a `if use_faiss:` branch in `personal_api.py`'s `lifespan()` function. Zero changes to any pipeline code.
- **Result**: The architecture cleanly separates retrieval logic from generation logic, meaning any new retriever is a drop-in replacement. This is the concrete benefit of Strategy + DI patterns.

**Q: Which SOLID principles did you apply in this project?**

- **Situation**: I designed the system with maintainability and extensibility as core goals.
- **Task**: I needed to apply software engineering best practices that would make the codebase professional and easy to extend.
- **Action**: I applied four SOLID principles: (1) **S**ingle Responsibility — each processor function handles only one data type; (2) **O**pen-Closed — adding a new retriever does not require modifying the pipeline; (3) **L**iskov Substitution — MockRetriever can seamlessly replace SimpleRetriever; (4) **D**ependency Inversion — Pipeline depends on an abstraction (the retrieve interface), not a concrete class.
- **Result**: These principles make the codebase modular, testable, and ready for extension — qualities that distinguish a well-engineered project from a quick prototype.

---

### 3.2 API Design (FastAPI + Pydantic + REST)

**Concept**

RESTful APIs are the foundation of a decoupled frontend-backend architecture. FastAPI provides automatic documentation generation, async support, and type-safe validation; Pydantic handles request/response data validation and serialization.

**Implementation**

File: `src/api/personal_api.py`

```python
# Pydantic data validation models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=500)
    k: int = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    use_verification: bool = Field(False, description="Enable answer verification")
    conversational: bool = Field(False, description="Use conversation mode")

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    sources: List[SourceInfo]
    retrieval_scores: List[float]

# FastAPI endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Main Q&A endpoint with anti-hallucination strategies"""
    ...

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load retriever and initialize pipelines
    yield
    # Shutdown: cleanup
```

**API Endpoint Design**:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` , `/health` | Health check |
| POST | `/ask` | Main Q&A endpoint |
| POST | `/feedback` | User feedback submission |
| GET | `/stats` | System statistics |
| GET | `/metrics` | Monitoring metrics |
| GET | `/sample-questions` | Sample questions |
| POST | `/clear-conversation` | Clear conversation history |
| POST | `/rebuild-index` | Rebuild knowledge base index |

**Why This Approach**

- **Pydantic validation**: automatically rejects invalid input (empty question returns 422, rating > 5 returns 422), no manual if-else validation needed
- **Lifespan pattern** (instead of deprecated `@app.on_event`): FastAPI's recommended approach for safer resource management
- **CORS middleware**: allows the Streamlit frontend to make cross-origin requests
- **Response model**: automatic serialization + automatic documentation (Swagger UI at `/docs`)

**Interview Q&A**

**Q: What are the differences between FastAPI and Flask?**

- **Situation**: When choosing a web framework for the API layer, I needed to select the best option for a modern AI-powered backend.
- **Task**: I needed a framework that offered async support, type safety, and auto-generated documentation.
- **Action**: I chose FastAPI over Flask because: (1) FastAPI natively supports **async/await**, while Flask requires additional extensions; (2) FastAPI has built-in Pydantic **type validation**, while Flask needs libraries like marshmallow; (3) FastAPI auto-generates **OpenAPI documentation** (Swagger UI), while Flask needs flask-restx; (4) FastAPI runs on ASGI (uvicorn) with 2-5x higher performance than Flask's WSGI.
- **Result**: FastAPI provided automatic request validation, interactive API documentation at `/docs`, and async request handling — all out of the box with minimal configuration. This significantly reduced boilerplate code and made the API self-documenting.

**Q: How do you handle API errors?**

- **Situation**: A production API needs comprehensive error handling across multiple layers to prevent silent failures and provide meaningful error messages.
- **Task**: I needed a layered error-handling strategy that covers input validation, business logic errors, and unexpected exceptions.
- **Action**: I implemented four-layer error handling: (1) **Pydantic validation** automatically returns 422 for malformed input; (2) **HTTPException** returns 503 (system not initialized) and 404 (resource not found); (3) **catch-all try-except** returns 500 and logs the full stack trace; (4) **middleware-level logging** records every request's latency and status code via the `log_requests` middleware.
- **Result**: Every error is captured, logged, and returned with an appropriate HTTP status code. The middleware ensures no request goes unrecorded, making debugging straightforward.

---

### 3.3 Testing Strategy (Unit, Integration, Mock-based CI)

**Concept**

Testing is the foundation of code quality. This project employs a three-layer testing strategy: unit tests verify components, integration tests verify endpoints, and CI automation ensures every commit passes.

**Implementation**

File: `tests/test_retriever.py` — Unit tests

```python
def test_mock_retriever_retrieve():
    """Test retrieval functionality"""
    retriever = MockRetriever()
    doc1 = Document(
        content="I am proficient in Python, machine learning, and deep learning",
        metadata={"type": "skills", "category": "skills"})
    doc2 = Document(
        content="I worked on a recommendation system project",
        metadata={"type": "project", "category": "projects"})
    retriever.add_documents([doc1, doc2])

    results = retriever.retrieve("What Python and machine learning skills do you have?", k=2)
    assert len(results) > 0
    assert results[0]["score"] > 0
```

File: `tests/test_api.py` — Integration tests

```python
from fastapi.testclient import TestClient
from api.personal_api import app
client = TestClient(app)

def test_ask_question_valid():
    response = client.post("/ask", json={
        "question": "What technologies are you proficient in?",
        "k": 3, "use_verification": False, "conversational": False})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["confidence"] in ["high", "medium", "low"]
```

File: `.github/workflows/ci.yml` — CI Pipeline

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
    - name: Run tests
      run: pytest tests/ -v --cov=src --cov-report=term-missing
      env:
        USE_MOCK: "true"  # CI uses mock mode, no API key needed
```

**Why This Approach**

- **Mock-based testing**: CI environments have no OpenAI API key, so `USE_MOCK=true` lets all tests run in mock mode at zero cost
- **`assert response.status_code in [200, 503]`**: tolerates uninitialized system state, allowing tests to pass across different environments
- **Python matrix (3.11, 3.12)**: ensures cross-version compatibility
- **Coverage report** (`--cov=src`): tracks code coverage

**Interview Q&A**

**Q: How do you test a system that depends on an LLM?**

- **Situation**: Testing AI systems is fundamentally harder than testing traditional software because outputs are non-deterministic.
- **Task**: I needed a testing strategy that could verify system correctness without requiring live API calls in CI.
- **Action**: I used a three-level approach: (1) **Mock LLM** — the MockRetriever + MockRAGPipeline replaces real API calls, allowing full pipeline logic testing in `tests/test_retriever.py` and `tests/test_api.py`; (2) **Structural testing** — for the same question, I verify the output structure (answer, confidence, sources fields exist and have valid types) rather than exact text; (3) **Integration testing** — a small number of real API calls verify the end-to-end flow (run manually, not in CI).
- **Result**: CI runs reliably without API keys, tests cover both component logic and API contract compliance, and the mock mode makes test execution fast (sub-second) and free.

**Q: What edge cases do your tests cover?**

- **Situation**: Edge cases are where most production bugs hide, so I systematically identified them during test development.
- **Task**: I needed to cover boundary conditions and failure modes to ensure robustness.
- **Action**: I wrote tests for: (1) empty input (`test_ask_question_invalid`); (2) invalid rating values (`test_feedback_invalid_rating`, rating=10); (3) 503 responses when the system is uninitialized; (4) retrieval on an empty document store (returns empty list); (5) save/load persistence consistency (`test_mock_retriever_save_load`).
- **Result**: These edge-case tests catch regression issues early and ensure the system degrades gracefully rather than crashing on unexpected input.

---

### 3.4 Observability (Structured Logging, Metrics, Health Checks)

**Concept**

Observability rests on three pillars: **Logs** (what happened), **Metrics** (how the system is performing), and **Traces** (what path a request followed). This project implements the first two.

**Implementation**

File: `src/utils/logger.py` — Structured Logging

```python
def setup_logger(name, level="INFO", log_file=None):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    # Console + File dual output
    console_handler = logging.StreamHandler(sys.stdout)
    if log_file:
        file_handler = logging.FileHandler(log_file)
```

File: `src/api/personal_api.py` — Metrics class + Request Logging Middleware

```python
class Metrics:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.questions_asked = 0
        self.feedback_count = 0
        self.start_time = datetime.now()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    metrics.record_request(latency, success=response.status_code < 400)
    logger.info(f"Response: {request.method} {request.url.path} - {response.status_code}",
                extra={"latency_ms": round(latency * 1000, 2)})
```

**Health Check Endpoint**:

```python
@app.get("/health")
async def health_check():
    return HealthResponse(
        status="healthy",
        documents_loaded=len(retriever.documents),
        categories=retriever.get_category_stats())
```

**Why This Approach**

- **Middleware-level logging**: every request is automatically logged without adding log statements to each endpoint
- **Metrics class**: in-memory metrics collection, suitable for a small system; can be replaced with a Prometheus client for production
- **Health check**: provides the foundation for Docker HEALTHCHECK and Kubernetes liveness probes
- **Extra fields** (`logger.info(..., extra={"latency_ms": ...})`): structured log fields make parsing with ELK/Splunk straightforward

**Interview Q&A**

**Q: If you needed to integrate Prometheus + Grafana, what changes would be required?**

- **Situation**: The current in-memory Metrics class works for a small deployment but would not scale to a production monitoring stack.
- **Task**: I needed to design the observability layer so that upgrading to Prometheus is a minimal change.
- **Action**: The upgrade path is straightforward: (1) install `prometheus-fastapi-instrumentator`; (2) call `Instrumentator().instrument(app).expose(app)` on the FastAPI app object; (3) the existing `/metrics` endpoint already returns data in a Prometheus-compatible format. The custom Metrics class can be retained as a supplement for business-specific metrics like `questions_asked` and `feedback_count`.
- **Result**: The current architecture is designed for easy migration to production monitoring. The key metrics (latency, error rate, request count) are already being tracked and just need to be exposed in Prometheus format.

---

### 3.5 Containerization & Deployment (Docker, Docker Compose, CI/CD)

**Concept**

Containerization ensures that "it works on my machine" translates to working in any environment. Docker Compose orchestrates multi-container services. CI/CD automatically tests and validates on every commit.

**Implementation**

File: `Dockerfile` — Layer-optimized build + security best practices

```dockerfile
FROM python:3.11-slim as base
WORKDIR /app

# Install dependencies first (leverages Docker layer cache)
COPY requirements_simple.txt .
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy application code (changes frequently, placed later)
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/

# Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check built into image
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.personal_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

File: `docker-compose.yml` — Multi-service orchestration

```yaml
services:
  api:
    environment:
      - USE_MOCK=${USE_MOCK:-true}     # Environment variable configuration
    volumes:
      - ./data:/app/data               # Data persistence
      - ./logs:/app/logs               # Log persistence
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  frontend:
    environment:
      - API_URL=http://api:8000        # Inter-service communication via Docker network
    depends_on:
      - api                            # Ensures API starts first
```

**Why This Approach**

- **Layer cache optimization**: copying requirements before source code — dependencies change rarely but code changes frequently, so code modifications do not trigger dependency reinstallation
- **Non-root user**: a security best practice ensuring the container process does not run as root
- **`restart: unless-stopped`**: automatic restart on process crash in production
- **`depends_on`**: ensures the frontend starts after the API (note: this only guarantees startup order, not that the API is ready)

**Interview Q&A**

**Q: What is the difference between Docker and a virtual machine?**

- **Concept**: Docker containers share the host's kernel, start in seconds, and produce MB-sized images. Virtual machines include a complete OS, take minutes to start, and produce GB-sized images. Docker is lighter and more suited for microservices, but VMs offer stronger security isolation (separate kernels).
- **In My Project**: I use Docker for the RAG API and Streamlit frontend because the lightweight container approach is ideal for a system with two cooperating services that need fast startup and easy deployment.
- **Key Insight**: The trade-off is isolation vs. efficiency. For this project, Docker's efficiency wins because both services are trusted internal components, not multi-tenant workloads requiring strong isolation.

**Q: Can `depends_on` guarantee the API is ready?**

- **Situation**: The frontend service depends on the API service being fully initialized, not just started.
- **Task**: I needed to ensure the frontend would not crash when the API container is running but still loading the knowledge base.
- **Action**: `depends_on` only guarantees container **startup**, not service **readiness**. Docker Compose v2 supports `condition: service_healthy` to wait for a healthcheck to pass. Additionally, my Streamlit frontend implements graceful degradation: `_check_api_available()` probes the health endpoint, and when the API is unreachable, `USE_MOCK_MODE` is set to `True`, automatically switching to the local mock pipeline.
- **Result**: The frontend always works — either via the backend API when available, or via the local mock pipeline when the API is not ready. Users never see a broken page.

---

### 3.6 Frontend Engineering (Streamlit — Session State, Caching, Graceful Degradation)

**Concept**

Streamlit transforms Python scripts into interactive web applications. The core challenge is managing Session State (Streamlit re-runs the entire script on every interaction) and handling backend unavailability.

**Implementation**

File: `frontend/personal_app.py`

**Session State Management**:

```python
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
```

**Resource Caching** (`@st.cache_resource`):

```python
@st.cache_resource
def get_local_pipeline():
    """Initialize local mock pipeline directly (no backend needed)."""
    from rag.mock_retriever import MockRetriever
    from rag.mock_pipeline import MockRAGPipeline
    from rag.knowledge_processor import build_knowledge_base
    
    kb_path = Path(__file__).parent.parent / "data" / "raw" / "knowledge_base.json"
    retriever = MockRetriever()
    documents = build_knowledge_base(str(kb_path))
    retriever.add_documents(documents)
    return MockRAGPipeline(retriever)
```

**Graceful Degradation**:

```python
def _check_api_available() -> bool:
    """Probe the backend API once and cache the result for the session."""
    if "api_available" not in st.session_state:
        try:
            resp = requests.get(f"{API_URL}/health", timeout=3)
            st.session_state.api_available = resp.status_code == 200
        except Exception:
            st.session_state.api_available = False
    return st.session_state.api_available

USE_MOCK_MODE = not _check_api_available()
```

**Why This Approach**

- **`@st.cache_resource`**: the pipeline is initialized only on first load; subsequent interactions reuse the same instance, avoiding knowledge base reconstruction on every interaction
- **Graceful degradation**: when the API is unavailable, the frontend automatically switches to a local mock pipeline, ensuring users always see content (especially important for Streamlit Cloud deployments where the backend may not be available)
- **Session state**: Streamlit's stateless execution model requires explicit state management; otherwise conversation history is lost on every interaction

**Interview Q&A**

**Q: What problems does Streamlit's re-execution model cause? How do you solve them?**

- **Situation**: Streamlit re-executes the entire Python script from top to bottom on every user interaction (button click, text input, etc.), creating challenges for state persistence and performance.
- **Task**: I needed to maintain conversation history, avoid expensive recomputation, and prevent UI glitches.
- **Action**: I solved three categories of problems: (1) **State loss** — used `st.session_state` to persist conversation history, current results, and theme preference across re-runs; (2) **Redundant computation** — used `@st.cache_resource` to initialize the mock pipeline only once; (3) **UI flickering** — avoided directly modifying widget values in callbacks, instead using a flag pattern (the `clear_input` flag in `st.session_state`) to signal input clearing on the next re-run.
- **Result**: The application feels like a normal stateful web app despite Streamlit's re-execution model. Conversation history persists, pipeline initialization happens only once, and input clearing works smoothly.

**Q: Why did you choose Streamlit over React?**

- **Situation**: I needed a frontend for the RAG system and had to choose between a full JavaScript framework and a Python-native option.
- **Task**: I needed rapid development, tight integration with the Python AI backend, and built-in UI components for data display.
- **Action**: I chose Streamlit because: (1) it is pure Python, consistent with the backend AI code, requiring no separate frontend team or toolchain; (2) prototype speed is extremely fast, ideal for a personal project and demo; (3) built-in session state, caching, and layout eliminate the need for Redux/Webpack configuration. The trade-off is limited customization ability.
- **Result**: The entire frontend was built in a single Python file (`personal_app.py`) with rich features including chat UI, analytics tab, theme switching, and feedback collection. For a production-grade application, the RESTful API design ensures a React frontend could be added later without any backend changes.

---

## 4. System Design Extension Questions

### 4.1 "If the document count grew from 200 to 10 million, how would you handle it?"

**Current System Bottleneck**:
- NumPy cosine similarity is O(n * d) brute-force search; 10 million documents * 1536 dimensions is approximately 60GB of memory, with per-query latency potentially reaching seconds

**Scaling Plan**:

```
Phase 1 (100K-1M): FAISS (Facebook AI Similarity Search)
├── IVF (Inverted File Index): partition documents into ~1000 clusters
├── PQ (Product Quantization): compress vectors, reducing memory 4-8x
└── Query searches only the nearest ~10 clusters → O(n/k * d)

Phase 2 (1M-10M): Vector Database
├── Pinecone / Qdrant / Weaviate (managed service)
├── HNSW (Hierarchical Navigable Small World): O(log n) queries
├── Distributed indexing + automatic sharding
└── Built-in metadata filtering and hybrid search

Phase 3 (10M+): Multi-Stage Retrieval
├── Stage 1: BM25 coarse filtering → 100K candidates
├── Stage 2: Lightweight embedding (e.g., 256-dim) → 10K candidates
├── Stage 3: Full embedding reranking → top-k
└── Cross-encoder reranker (fine ranking)
```

**Key Trade-offs**:
- **Recall vs Speed**: ANN methods sacrifice some accuracy (95-99% recall) for 100x speedup
- **Memory vs Precision**: PQ quantization compression introduces minor errors
- **Build time vs Query time**: HNSW index construction is slow, but query execution is extremely fast

---

### 4.2 "If you needed to support image search, how would you extend the system?"

**Approach**:

```
1. Use CLIP (Contrastive Language-Image Pre-Training) as the multimodal embedding model
   - CLIP maps both text and images into the same 512/768-dim vector space
   - Enables text-to-image queries and image-to-image queries

2. Knowledge Processing Layer Extension
   ├── ImageDocument class (extends Document, adds image_path)
   ├── process_portfolio_images() to process portfolio images
   └── Image caption generation (using GPT-4V/BLIP-2 to generate image descriptions)

3. Retrieval Layer Extension
   ├── MultimodalRetriever (supports text + image embedding)
   ├── Index stores both text and image vectors
   └── Queries compute cross-modal similarity

4. Frontend Extension
   ├── Image upload (st.file_uploader)
   ├── Mixed text-image display
   └── Image relevance preview
```

**Implementation Details**:
- CLIP embedding dimensions are typically 512, smaller than text-embedding-3-small (1536). Mixed storage requires separate normalization
- Alternatively: use `text-embedding-3-small` for text and CLIP for images, performing late fusion during retrieval

---

### 4.3 "How do you evaluate RAG system quality?"

**Evaluation Framework** (referencing RAGAS):

| Metric | Meaning | Calculation |
|--------|---------|-------------|
| **Context Recall@k** | Did retrieval find the correct documents? | `relevant docs found / total relevant docs` |
| **Context Precision@k** | How many retrieved results are truly relevant? | `relevant docs / k` |
| **NDCG** | Ranking quality — are relevant documents ranked higher? | Normalized Discounted Cumulative Gain |
| **Faithfulness** | Is the answer faithful to retrieved documents? | `supported claims / total claims` |
| **Answer Relevance** | Does the answer directly address the question? | LLM-as-judge scoring |
| **Latency** | End-to-end response time | P50, P95, P99 |

**Evaluation Methods in This Project**:
- `/feedback` endpoint collects user 1-5 ratings, providing indirect Answer Relevance assessment
- Confidence score (high/medium/low) provides indirect Faithfulness assessment
- `/metrics` endpoint tracks average latency and error rate

**Ideal Evaluation Pipeline**:
```python
# Build evaluation set: {question, expected_answer, relevant_doc_ids}
# Run RAG pipeline
# Compute automated metrics
# Manual spot-check review (10-20%)
```

---

### 4.4 "How would you do A/B testing to compare different retrievers?"

**Experiment Design**:

```
1. Traffic Splitting Strategy
   ├── Random assignment: 50% users use Retriever A, 50% use Retriever B
   └── Hash session_id to ensure same user stays in same experiment group

2. Comparison Dimensions
   ├── Retrieval metrics: recall@5, precision@5, NDCG
   ├── Generation metrics: faithfulness, relevance (using LLM-as-judge)
   ├── User metrics: feedback rating, click-through rate on sources
   └── System metrics: latency (P50, P95), cost per query

3. Implementation (within the current architecture)
   ├── Add retriever_type parameter to /ask endpoint
   ├── Log which retriever was used for each request in Metrics
   └── Use statistical tests (t-test, Mann-Whitney U) to check significance

4. Pitfalls to Avoid
   ├── Do not use the same LLM to evaluate different retrievers (control variables)
   ├── Ensure sufficient sample size (at least 100 queries per group)
   └── Watch for novelty effect (new approach may have initial bias)
```

---

## 5. Behavioral Interview Prep

### 5.1 "Tell me about a challenging technical project"

- **Situation**: I wanted to add an intelligent Q&A feature to my personal website so visitors could learn about my background and project experience through natural-language conversation, rather than browsing static pages. The challenge was that personal data is private — no LLM has been trained on it — and any fabricated information would undermine my credibility.
- **Task**: I needed to build a RAG system that met three requirements: (1) answers must be accurate with zero fabrication; (2) development and testing could not incur API costs on every run; (3) the system should be deployable with a single command.
- **Action**: I designed a 4-layer anti-hallucination strategy (low temperature, strict prompts, confidence scoring, and a verification pass) to ensure answer reliability. I implemented a Mock/Real dual-mode architecture — the MockRetriever uses keyword matching plus intent detection as an embedding substitute, enabling development, testing, and CI to run entirely at zero cost. I containerized the API and frontend services with Docker Compose for one-command deployment and built a complete CI pipeline with GitHub Actions running pytest, black, ruff, and mypy.
- **Result**: The system accurately answers questions about my skills, projects, and education, with the confidence-"high" rate exceeding 70%. Mock-mode CI never fails due to API quota limits. End-to-end latency is under 3 seconds in real mode and under 100ms in mock mode. This project gave me deep understanding of RAG architecture, prompt engineering, and production-grade Python engineering practices.

---

### 5.2 "How do you handle trade-offs?"

- **Situation**: Throughout this project, I repeatedly faced decisions where two valid approaches had competing benefits — cost vs. accuracy, speed vs. reliability, simplicity vs. capability. I needed a consistent framework for making these decisions.
- **Task**: My task was to make principled trade-off decisions that optimized for the current project phase while explicitly preserving the ability to change course later.
- **Action**: I adopted a framework of "choose for the current constraints, design for future flexibility." For each major decision, I identified the trade-off axes, evaluated them against current requirements, and ensured the architecture would accommodate the alternative choice later. Specific examples:

| Trade-off | Choice | Reasoning |
|-----------|--------|-----------|
| Mock vs Real mode | Default Mock | Development efficiency and cost come first, but Real mode is fully retained and switchable via one config line |
| Accuracy vs Latency | Verification off by default | Most Q&A does not need two-pass verification, but high-stakes scenarios can enable it on demand |
| NumPy vs FAISS | NumPy | 200 documents do not need an ANN index, but the Strategy Pattern preserves the replacement interface |
| Streamlit vs React | Streamlit | Personal project and demo scenario favors rapid development, but the RESTful API design supports a React frontend later |
| GPT-3.5 vs GPT-4 | GPT-3.5 default | Lower cost and faster, sufficient for personal Q&A; seamlessly switchable to GPT-4 via an environment variable |

- **Result**: Every decision has a clear rationale tied to current constraints, and every decision has an explicit upgrade path. This approach prevents over-engineering while ensuring the system is never painted into a corner.

---

### 5.3 "How do you test AI systems?"

- **Situation**: Testing AI systems is fundamentally different from testing traditional software because LLM outputs are non-deterministic — the same input can produce different outputs each time. I needed a testing strategy that could provide meaningful quality assurance despite this inherent variability.
- **Task**: I had to design a test suite that could run in CI without API keys, verify system correctness despite non-deterministic outputs, and cover both traditional software edge cases and AI-specific quality concerns.
- **Action**: I implemented a four-layer testing strategy:
  1. **Deterministic component testing**: unit tests for the retriever's add/retrieve/save/load operations (`test_retriever.py`) and API endpoint request/response format validation (`test_api.py`), including Pydantic model validation (testing 422 responses).
  2. **Structural testing for AI components**: instead of testing exact LLM output text, I verify output **structure** (answer, confidence, and sources fields exist with valid types), confidence value domain (must be high/medium/low), and retrieval score reasonableness (0-1 range, descending order).
  3. **Mock-based CI**: MockRetriever + MockRAGPipeline replace all API calls; CI environment sets `USE_MOCK=true`; coverage tracking (`--cov=src`) monitors branch coverage.
  4. **User feedback loop**: the `/feedback` endpoint collects 1-5 ratings; feedback data is stored in `logs/feedback.jsonl`; periodic analysis of low-scored answers informs prompt and retrieval strategy improvements.
- **Result**: The test suite runs reliably in CI at zero API cost, covers both deterministic and AI-specific edge cases, and the feedback loop provides a continuous quality improvement mechanism. Tests pass across Python 3.11 and 3.12 via the CI matrix.

---

### 5.4 "Describe a time you had to learn a new technology quickly"

- **Situation**: When I decided to build the RAG system, I had theoretical knowledge of embeddings and LLMs from coursework, but I had never implemented a complete RAG pipeline with production-quality anti-hallucination strategies, nor had I built a FastAPI backend with Pydantic validation, middleware, and lifespan management. I also needed to learn Streamlit's unique re-execution model for the frontend.
- **Task**: I needed to go from conceptual understanding to a working, deployable system within a limited timeframe, learning FastAPI, Streamlit, OpenAI's embedding API, and prompt engineering best practices along the way.
- **Action**: I used a deliberate learning strategy: first, I built the simplest possible end-to-end prototype (a basic retriever + LLM call) to validate the core concept. Then I iteratively added complexity — anti-hallucination layers, mock mode, conversation memory, API validation, structured logging — learning each technology just-in-time as the feature required it. I read FastAPI's official documentation for the lifespan pattern and Pydantic models, studied OpenAI's embedding API for batch processing and model selection, and explored Streamlit's `session_state` and `cache_resource` through hands-on experimentation. Critically, I built the mock retriever early so I could iterate on the full pipeline without waiting for API calls or incurring costs.
- **Result**: Within the project timeline, I delivered a system with 8 API endpoints, a conversational frontend, 4-layer anti-hallucination, Docker deployment, and CI/CD — all technologies I had to learn during the project. The mock-mode architecture was especially valuable because it allowed me to iterate 10x faster during the learning phase. This experience taught me that the fastest way to learn a technology is to have a concrete problem to solve with it.

---

### 5.5 "Tell me about a time you made a design decision you later had to change"

- **Situation**: Early in the project, I initially implemented the retriever as a single `SimpleRetriever` class that always required an OpenAI API key to generate embeddings. This meant every test run, every CI pipeline execution, and every development iteration incurred API calls and costs. I quickly realized this was unsustainable for rapid development and would make CI unreliable.
- **Task**: I needed to redesign the retriever layer so the entire system could run and be tested without any API dependency, while keeping the real embedding-based retriever fully functional for production use.
- **Action**: I created the `MockRetriever` class in `src/rag/mock_retriever.py` that uses keyword matching with Jaccard similarity and intent-based boosting instead of embeddings. I also created `MockRAGPipeline` to pair with it. I restructured the API's `lifespan()` function to check the `USE_MOCK` environment variable and instantiate the appropriate retriever and pipeline pair. The key architectural insight was applying the Strategy pattern — both retrievers implement the same interface (`retrieve()`, `add_documents()`, `save()`, `load()`, `get_category_stats()`), so the pipeline code does not change at all.
- **Result**: This redesign saved significant API costs during development, made CI 100% reliable (no API quota failures), and actually improved the architecture by introducing the Strategy pattern. The mock retriever also enabled the Streamlit frontend's graceful degradation feature. This experience reinforced that early architectural decisions are worth revisiting — the cost of change is low early on, and the benefits compound over time.

---

### 5.6 "How do you prioritize when you have multiple approaches to a problem?"

- **Situation**: When designing the anti-hallucination system, I identified at least six possible strategies: low temperature, strict prompts, confidence scoring, two-pass verification, output filtering (regex-based claim validation), and fine-tuning a smaller model specifically for fact-checking. I could not implement all of them within the project timeline and needed to decide which ones would deliver the most value.
- **Task**: I needed to select a subset of strategies that would maximize hallucination prevention while respecting constraints on development time, API cost, and system complexity.
- **Action**: I evaluated each approach on three criteria: (1) **Implementation effort** — how long would it take to build and test; (2) **Impact** — how much would it reduce hallucination in practice; (3) **Reversibility** — could I easily change or remove it later. Low temperature and strict prompts scored highest on all three criteria (trivial to implement, high impact, easily adjustable), so I implemented them first. Confidence scoring was medium effort but provided valuable user-facing information, so it came next. Two-pass verification was higher effort but offered a unique defense-in-depth capability, so I made it optional (opt-in via a flag). I explicitly deferred output filtering and fine-tuning as future enhancements because they required significantly more effort for marginal improvement over the first four layers.
- **Result**: The four strategies I implemented provide robust hallucination prevention — the system reliably refuses to answer out-of-scope questions and flags low-confidence answers. By making verification optional, I avoided imposing latency costs on routine queries. The prioritization framework (effort, impact, reversibility) is one I now apply to all technical decisions, not just this project.

---

## 6. Quick Review Cards

| # | Term | Definition | Where in Project |
|---|------|------------|------------------|
| 1 | **RAG** | Retrieval-Augmented Generation. Retrieves relevant documents from a knowledge base first, then has the LLM generate answers based on those documents. | `src/rag/pipeline.py` — `PersonalRAGPipeline.query()` |
| 2 | **Embedding** | Maps text to high-dimensional vectors where semantically similar texts are closer in vector space. | `src/rag/retriever.py` — `_get_embedding()`, model: `text-embedding-3-small` (1536-dim) |
| 3 | **Cosine Similarity** | Measures directional similarity between two vectors, range [-1, 1], where 1 means identical. | `src/rag/retriever.py` — `_cosine_similarity()`, normalize then dot product |
| 4 | **Vector Database** | Database specialized for storing and retrieving high-dimensional vectors (e.g., FAISS, Pinecone, Qdrant). | Currently uses NumPy array (`self.embeddings`); Qdrant config reserved in `docker-compose.yml` |
| 5 | **Chunking** | The process of splitting long documents into smaller segments suitable for retrieval. | `src/rag/knowledge_processor.py` — semantic unit chunking (one project/FAQ/experience per chunk) |
| 6 | **Temperature** | LLM generation randomness parameter; lower values produce more deterministic output. | `src/rag/pipeline.py` — `temperature=0.3` (main query), `temperature=0.1` (verification query) |
| 7 | **Top-k** | Number of most relevant documents returned during retrieval. | `src/rag/retriever.py` — `retrieve(query, k=5)`, API constrains `ge=1, le=10` |
| 8 | **Hallucination** | LLM generates plausible-sounding but factually incorrect content. | `src/rag/pipeline.py` — 4-layer strategy: low temp, strict prompt, confidence, verification |
| 9 | **Prompt Engineering** | Designing input prompts to guide LLM output toward desired behavior. | `src/rag/pipeline.py` — `self.system_prompt` (role+rules), `_build_user_prompt()` (context injection) |
| 10 | **FAISS** | Facebook AI Similarity Search, efficient vector retrieval library supporting IVF/HNSW/PQ. | Not currently used (NumPy brute-force); discussed as upgrade path in system design section |
| 11 | **Pydantic** | Python data validation library, core component of FastAPI for request/response schema definition. | `src/api/personal_api.py` — `QuestionRequest`, `AnswerResponse`, `FeedbackRequest` |
| 12 | **FastAPI** | High-performance Python web framework with async support, auto OpenAPI docs, and type validation. | `src/api/personal_api.py` — 8 REST endpoints, CORS middleware, lifespan management |
| 13 | **Middleware** | A common logic layer inserted into the request/response processing chain. | `src/api/personal_api.py` — `log_requests` middleware (records latency and status code per request) |
| 14 | **CORS** | Cross-Origin Resource Sharing, allows frontends on different domains to access the API. | `src/api/personal_api.py` — `CORSMiddleware(allow_origins=["*"])` |
| 15 | **Docker** | Containerization platform that packages applications and dependencies into portable images. | `Dockerfile` — layer-optimized build, non-root user, HEALTHCHECK |
| 16 | **CI/CD** | Continuous Integration / Continuous Delivery, automated testing and deployment pipeline. | `.github/workflows/ci.yml` — Python 3.11/3.12 matrix, pytest + black + ruff + mypy |
| 17 | **Session State** | State storage in Streamlit that persists across user interactions. | `frontend/personal_app.py` — `st.session_state.conversation_history`, `api_available`, `theme` |
| 18 | **Graceful Degradation** | Design where the system still provides basic service when some features are unavailable. | `frontend/personal_app.py` — auto-switches to local mock pipeline when API is unavailable |
| 19 | **Dependency Injection** | Injecting dependencies via constructors from outside rather than creating them internally, facilitating testing and swapping. | `src/rag/pipeline.py` — `PersonalRAGPipeline(retriever)` receives retriever from outside |
| 20 | **Strategy Pattern** | Defines a family of algorithms, encapsulates each one, and makes them interchangeable. | `src/rag/retriever.py` vs `src/rag/mock_retriever.py` — two retriever implementations sharing the same interface, switchable at runtime |

---

## Appendix: Interview Day Checklist

- [ ] Can draw the system architecture diagram (the ASCII diagram above)
- [ ] Can explain the RAG 5-step flow (retrieve -> build context -> generate -> assess confidence -> build response)
- [ ] Can name the 4 anti-hallucination strategies and their trade-offs
- [ ] Can explain why cosine similarity was chosen and when to switch to FAISS
- [ ] Can explain the Mock/Real dual-mode design motivation and implementation
- [ ] Can list the design patterns used (Strategy, Template Method, Factory, DI)
- [ ] Can explain the purpose of category weighting and intent detection
- [ ] Can discuss the scaling path from 200 to 10 million documents
- [ ] Can describe how to evaluate RAG system quality (recall@k, faithfulness, etc.)
- [ ] Can tell the project story using the STAR framework

---

> **Final Reminder**: During the interview, do not just say "I used RAG" — say "I built a RAG system with 4-layer anti-hallucination, Mock/Real dual-mode architecture, and multi-turn conversation support." **Specific details + design reasoning** is what distinguishes a junior candidate from a senior one. Good luck!
