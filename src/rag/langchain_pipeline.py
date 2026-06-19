"""
LangChain-based RAG pipeline with chain composition, memory management,
and structured output parsing. Wraps retriever components with LangChain
abstractions for modularity and extensibility.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferWindowMemory

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LangChainRAGPipeline:
    """
    RAG Pipeline built on LangChain with anti-hallucination prompt design,
    confidence scoring, and optional two-pass verification.

    Drop-in replacement for PersonalRAGPipeline -- the return format of
    ``query()`` matches exactly so the API layer works without changes.
    """

    # The system prompt is identical to PersonalRAGPipeline for consistency.
    SYSTEM_PROMPT = (
        "You are an intelligent assistant for a personal website, specialized "
        "in answering questions about the website owner.\n\n"
        "Important Rules:\n"
        "1. Only use the provided context information to answer questions\n"
        "2. If there is no relevant information in the context, explicitly state "
        '"Based on the information I have, I cannot answer this question"\n'
        "3. Do not fabricate or speculate any information\n"
        "4. If information is incomplete, honestly point this out\n"
        "5. Use a friendly but professional tone\n"
        "6. Keep answers concise and focused\n"
        "7. When referencing specific projects or experiences, provide concrete "
        "details when available\n\n"
        "Your answers should:\n"
        "- Accurately reflect information in the context\n"
        "- Not add any content not mentioned in the context\n"
        "- Appropriately acknowledge limitations of knowledge\n"
        "- Maintain consistency and credibility"
    )

    HUMAN_TEMPLATE = (
        "Answer the user's question based on the following information.\n\n"
        "=== Reference Information ===\n"
        "{context}\n"
        "=== End of Reference Information ===\n\n"
        "User Question: {question}\n\n"
        "Please provide an accurate and concise answer based on the above "
        "information. If the information is insufficient to fully answer the "
        "question, please indicate so."
    )

    VERIFICATION_SYSTEM_PROMPT = (
        "You are a fact-checking assistant specialized in verifying whether "
        "answers are consistent with source materials."
    )

    VERIFICATION_HUMAN_TEMPLATE = (
        "Please verify whether the following answer is completely based on the "
        "provided information without adding extra content.\n\n"
        "Original Question: {question}\n\n"
        "Generated Answer: {answer}\n\n"
        "Reference Information Summary:\n{sources_summary}\n\n"
        "Verification Requirements:\n"
        "1. Can all facts in the answer be found in the reference information?\n"
        "2. Does the answer add content not present in the reference information?\n"
        "3. If there are issues, please point them out and provide correction "
        "suggestions.\n\n"
        "Please reply in JSON format:\n"
        '{{"verified": true/false, "issues": ["list of issues"], '
        '"suggestion": "correction suggestion or empty string"}}'
    )

    def __init__(
        self,
        retriever,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
    ):
        """
        Initialize the LangChain RAG pipeline.

        Args:
            retriever: Any retriever instance that exposes a
                       ``retrieve(query, k, threshold)`` method returning
                       ``[{"content", "metadata", "score"}, ...]``.
            llm_model: OpenAI chat model identifier.
            temperature: Sampling temperature (lower = more deterministic).
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages are required. Install with: "
                "pip install langchain langchain-openai"
            )

        self.retriever = retriever
        self.llm_model = llm_model
        self.temperature = temperature

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it in your .env file or export it in your shell."
            )

        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            openai_api_key=openai_api_key,
        )

        # Build main generation chain and verification chain
        self.chain = self._build_retrieval_chain()
        self.verification_chain = self._build_verification_chain()

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_retrieval_chain(self) -> LLMChain:
        """Create the retrieval + generation LLMChain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(self.HUMAN_TEMPLATE),
            ]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def _build_verification_chain(self) -> LLMChain:
        """Create a separate chain for answer verification."""
        verification_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.VERIFICATION_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(self.VERIFICATION_HUMAN_TEMPLATE),
            ]
        )
        return LLMChain(llm=verification_llm, prompt=prompt)

    # ------------------------------------------------------------------
    # Context & confidence helpers
    # ------------------------------------------------------------------

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build a context string from retrieved documents."""
        context_parts: List[str] = []
        for i, doc in enumerate(retrieved_docs, 1):
            relevance = "High" if doc["score"] > 0.7 else "Medium" if doc["score"] > 0.5 else "Low"
            context_parts.append(
                f"[Information Snippet {i}] (Relevance: {relevance})\n" f"{doc['content']}\n"
            )
        return "\n".join(context_parts)

    def _assess_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Assess answer confidence based on retrieval scores."""
        if not retrieved_docs:
            return "low"

        avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)
        max_score = max(doc["score"] for doc in retrieved_docs)

        if max_score > 0.75 and avg_score > 0.6:
            return "high"
        elif max_score > 0.5 and avg_score > 0.4:
            return "medium"
        else:
            return "low"

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a question through the LangChain RAG pipeline.

        Args:
            question: User question.
            k: Number of documents to retrieve.
            include_sources: Whether to attach source details.

        Returns:
            Dict with keys: question, answer, confidence, sources,
            retrieval_scores.
        """
        # Step 1: Retrieve
        retrieved_docs = self.retriever.retrieve(question, k=k, threshold=0.3)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": (
                    "Sorry, I could not find relevant information in the "
                    "knowledge base for your question. You can try asking "
                    "about my skills, project experience, education "
                    "background, or contact information."
                ),
                "sources": [],
                "confidence": "low",
                "retrieval_scores": [],
            }

        # Step 2: Build context
        context = self._build_context(retrieved_docs)

        # Step 3: Generate answer via LangChain chain
        try:
            chain_output = self.chain.invoke({"context": context, "question": question})
            answer = chain_output.get("text", "")
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": "error",
                "retrieval_scores": [],
            }

        # Step 4: Confidence
        confidence = self._assess_confidence(retrieved_docs)

        # Step 5: Build response
        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "retrieval_scores": [doc["score"] for doc in retrieved_docs],
        }

        if include_sources:
            result["sources"] = [
                {
                    "type": doc["metadata"].get("type", "unknown"),
                    "category": doc["metadata"].get("category", "unknown"),
                    "score": round(doc["score"], 3),
                    "preview": (
                        doc["content"][:200] + "..."
                        if len(doc["content"]) > 200
                        else doc["content"]
                    ),
                }
                for doc in retrieved_docs
            ]

        return result

    def query_with_verification(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Two-pass query: generate an answer, then verify it against sources.

        Returns the same dict as ``query()`` with an additional
        ``verification`` key containing the LLM's verification output.
        """
        # First pass
        initial_result = self.query(question, k=k, include_sources=True)

        if initial_result["confidence"] == "error" or not initial_result["sources"]:
            return initial_result

        # Second pass: verify
        sources_summary = "\n".join(s["preview"] for s in initial_result["sources"][:3])

        try:
            verification_output = self.verification_chain.invoke(
                {
                    "question": question,
                    "answer": initial_result["answer"],
                    "sources_summary": sources_summary,
                }
            )
            verification_text = verification_output.get("text", "")
            initial_result["verification"] = verification_text

            # Parse verification result
            try:
                verification_data = json.loads(verification_text)
                if not verification_data.get("verified", True):
                    initial_result["answer"] += (
                        "\n\n[Note: Some content in this answer may need "
                        "further verification against source materials.]"
                    )
                    initial_result["confidence"] = "low"
            except (json.JSONDecodeError, TypeError):
                if (
                    '"verified": false' in verification_text.lower()
                    or '"verified":false' in verification_text.lower()
                ):
                    initial_result["answer"] += (
                        "\n\n[Note: Some content in this answer may need "
                        "further verification against source materials.]"
                    )
                    initial_result["confidence"] = "low"

        except Exception as e:
            initial_result["verification_error"] = str(e)

        return initial_result


class LangChainConversationalPipeline(LangChainRAGPipeline):
    """
    LangChain RAG Pipeline with conversation memory for multi-turn
    interactions.  Uses ``ConversationBufferWindowMemory`` to keep a
    sliding window of recent exchanges.
    """

    def __init__(
        self,
        retriever,
        llm_model: str = "gpt-3.5-turbo",
        memory_size: int = 5,
    ):
        """
        Args:
            retriever: Retriever instance with ``retrieve()`` method.
            llm_model: OpenAI chat model identifier.
            memory_size: Number of recent conversation turns to retain.
        """
        super().__init__(retriever, llm_model=llm_model)
        self.memory_size = memory_size

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages are required. Install with: "
                "pip install langchain langchain-openai"
            )

        self.memory = ConversationBufferWindowMemory(k=self.memory_size)

    def _get_conversation_context(self) -> str:
        """Retrieve recent conversation history from LangChain memory."""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        if not history:
            return ""
        return f"\n=== Conversation History ===\n{history}\n"

    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Query with conversation context injected.

        The conversation history is prepended to the question so the
        retriever and LLM can use prior context.
        """
        conv_context = self._get_conversation_context()

        if conv_context:
            enhanced_question = f"{conv_context}\nCurrent Question: {question}"
        else:
            enhanced_question = question

        # Use parent class query with enhanced question
        result = super().query(enhanced_question, k=k, include_sources=include_sources)
        # Restore original question in output
        result["question"] = question

        # Save turn to memory
        self.memory.save_context({"input": question}, {"output": result["answer"]})

        return result

    def clear_history(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
