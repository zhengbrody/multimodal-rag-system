"""
Personal RAG Pipeline with Anti-Hallucination Strategies

Core RAG pipeline that:
1. Retrieves relevant personal information
2. Uses carefully designed prompts
3. Implements strategies to prevent hallucination
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class PersonalRAGPipeline:
    """
    RAG Pipeline specialized for personal Q&A with anti-hallucination measures
    """

    def __init__(
        self,
        retriever,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,  # Lower temperature for more factual responses
        max_tokens: int = 1000
    ):
        """
        Initialize RAG pipeline

        Args:
            retriever: Document retriever instance
            llm_model: OpenAI model to use (gpt-3.5-turbo or gpt-4)
            temperature: Response randomness (lower = more deterministic)
            max_tokens: Maximum response length
        """
        self.retriever = retriever
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # System prompt with anti-hallucination instructions
        self.system_prompt = """You are an intelligent assistant for a personal website, specialized in answering questions about the website owner.

Important Rules:
1. Only use the provided context information to answer questions
2. If there is no relevant information in the context, explicitly state "Based on the information I have, I cannot answer this question"
3. Do not fabricate or speculate any information
4. If information is incomplete, honestly point this out
5. Use a friendly but professional tone
6. Keep answers concise and focused
7. When referencing specific projects or experiences, provide concrete details when available

Your answers should:
- Accurately reflect information in the context
- Not add any content not mentioned in the context
- Appropriately acknowledge limitations of knowledge
- Maintain consistency and credibility"""

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            relevance = "High" if doc['score'] > 0.7 else "Medium" if doc['score'] > 0.5 else "Low"
            context_parts.append(
                f"[Information Snippet {i}] (Relevance: {relevance})\n"
                f"{doc['content']}\n"
            )

        return "\n".join(context_parts)

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build the user prompt with context and question"""
        return f"""Answer the user's question based on the following information.

=== Reference Information ===
{context}
=== End of Reference Information ===

User Question: {question}

Please provide an accurate and concise answer based on the above information. If the information is insufficient to fully answer the question, please indicate so."""

    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline

        Args:
            question: User's question
            k: Number of documents to retrieve
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary with answer and metadata
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k=k, threshold=0.3)

        if not retrieved_docs:
            return {
                'question': question,
                'answer': 'Sorry, I could not find relevant information in the knowledge base for your question. You can try asking about my skills, project experience, education background, or contact information.',
                'sources': [],
                'confidence': 'low',
                'retrieval_scores': []
            }

        # Step 2: Build context
        context = self._build_context(retrieved_docs)

        # Step 3: Generate response with LLM
        user_prompt = self._build_user_prompt(question, context)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content

        except Exception as e:
            return {
                'question': question,
                'answer': f'Error generating answer: {str(e)}',
                'sources': [],
                'confidence': 'error',
                'retrieval_scores': []
            }

        # Step 4: Assess confidence based on retrieval scores
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        max_score = max(doc['score'] for doc in retrieved_docs)

        if max_score > 0.75 and avg_score > 0.6:
            confidence = 'high'
        elif max_score > 0.5 and avg_score > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Step 5: Build response
        result = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'retrieval_scores': [doc['score'] for doc in retrieved_docs]
        }

        if include_sources:
            result['sources'] = [
                {
                    'type': doc['metadata'].get('type', 'unknown'),
                    'category': doc['metadata'].get('category', 'unknown'),
                    'score': round(doc['score'], 3),
                    'preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
                }
                for doc in retrieved_docs
            ]

        return result

    def query_with_verification(
        self,
        question: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Enhanced query with additional verification step to reduce hallucination

        Performs a second pass to verify the answer against sources
        """
        # First pass: get answer
        initial_result = self.query(question, k=k, include_sources=True)

        if initial_result['confidence'] == 'error' or not initial_result['sources']:
            return initial_result

        # Second pass: verify answer
        verification_prompt = f"""Please verify whether the following answer is completely based on the provided information without adding extra content.

Original Question: {question}

Generated Answer: {initial_result['answer']}

Reference Information Summary:
{chr(10).join([s['preview'] for s in initial_result['sources'][:3]])}

Verification Requirements:
1. Can all facts in the answer be found in the reference information?
2. Does the answer add content not present in the reference information?
3. If there are issues, please point them out and provide correction suggestions.

Please reply in JSON format:
{{"verified": true/false, "issues": ["list of issues"], "suggestion": "correction suggestion or empty string"}}"""

        try:
            verification_response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant specialized in verifying whether answers are consistent with source materials."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            # Parse verification (simplified - in production use proper JSON parsing)
            verification_text = verification_response.choices[0].message.content
            initial_result['verification'] = verification_text

            # If verification finds issues, add warning
            if '"verified": false' in verification_text.lower() or '"verified":false' in verification_text.lower():
                initial_result['answer'] += "\n\n[Note: Some content in this answer may need further verification]"
                initial_result['confidence'] = 'low'

        except Exception as e:
            initial_result['verification_error'] = str(e)

        return initial_result


class ConversationalRAGPipeline(PersonalRAGPipeline):
    """
    RAG Pipeline with conversation memory for multi-turn interactions
    """

    def __init__(self, retriever, llm_model: str = "gpt-3.5-turbo", memory_size: int = 5):
        super().__init__(retriever, llm_model)
        self.conversation_history = []
        self.memory_size = memory_size

    def _get_conversation_context(self) -> str:
        """Get recent conversation history"""
        if not self.conversation_history:
            return ""

        history_text = "\n=== Conversation History ===\n"
        for turn in self.conversation_history[-self.memory_size:]:
            history_text += f"User: {turn['question']}\n"
            history_text += f"Assistant: {turn['answer'][:200]}...\n\n"

        return history_text

    def query(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Query with conversation context"""
        # Add conversation history to the prompt
        conv_context = self._get_conversation_context()

        if conv_context:
            # Modify question to include context
            enhanced_question = f"{conv_context}\nCurrent Question: {question}"
        else:
            enhanced_question = question

        # Get answer using parent method
        result = super().query(enhanced_question, k=k, include_sources=include_sources)
        result['question'] = question  # Keep original question

        # Update conversation history
        self.conversation_history.append({
            'question': question,
            'answer': result['answer']
        })

        return result

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


if __name__ == "__main__":
    # Test the pipeline
    from retriever import PersonalRAGRetriever
    from pathlib import Path

    # Load retriever
    retriever_path = Path(__file__).parent.parent.parent / "data" / "processed" / "retriever.pkl"

    if retriever_path.exists():
        retriever = PersonalRAGRetriever()
        retriever.load(str(retriever_path))

        # Initialize pipeline
        pipeline = PersonalRAGPipeline(retriever)

        # Test queries
        test_questions = [
            "What technologies are you most proficient in?",
            "Tell me about your proudest project",
            "What are your contact details?"
        ]

        for question in test_questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            result = pipeline.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nConfidence: {result['confidence']}")
            print(f"Retrieval scores: {result['retrieval_scores']}")
    else:
        print(f"Retriever not found at {retriever_path}")
        print("Please run retriever.py first to build and save the retriever.")
