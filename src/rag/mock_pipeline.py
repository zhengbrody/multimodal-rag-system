"""
Mock RAG Pipeline for Testing Without API

Generates simple template-based answers instead of calling LLM
No API costs - runs completely locally
"""

from typing import List, Dict, Any


class MockRAGPipeline:
    """
    Mock RAG Pipeline that generates answers without LLM
    Uses retrieved documents to create template-based responses
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def _generate_mock_answer(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate a simple answer based on retrieved documents"""
        if not retrieved_docs:
            return "Sorry, I could not find relevant information for your question."

        # Get the most relevant document
        top_doc = retrieved_docs[0]
        doc_type = top_doc['metadata'].get('type', 'unknown')

        # Create answer based on document type and content
        answer_parts = [f"Based on my knowledge base, here's what I found:\n"]

        for i, doc in enumerate(retrieved_docs[:3], 1):
            content = doc['content']
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."

            answer_parts.append(f"\n**Source {i}** ({doc['metadata'].get('type', 'info')}):\n{content}\n")

        answer_parts.append(f"\n---\n*This is a mock response generated without LLM. In production, an AI model would synthesize this information into a natural language answer.*")

        return "\n".join(answer_parts)

    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question through the mock RAG pipeline
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k=k, threshold=0.0)

        if not retrieved_docs:
            return {
                'question': question,
                'answer': 'Sorry, I could not find relevant information in the knowledge base for your question. You can try asking about skills, projects, education, or contact information.',
                'sources': [],
                'confidence': 'low',
                'retrieval_scores': []
            }

        # Step 2: Generate mock answer
        answer = self._generate_mock_answer(question, retrieved_docs)

        # Step 3: Assess confidence based on retrieval scores
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        max_score = max(doc['score'] for doc in retrieved_docs)

        if max_score > 0.5 and avg_score > 0.3:
            confidence = 'high'
        elif max_score > 0.3 and avg_score > 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Step 4: Build response
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

    def query_with_verification(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Mock verification - just return normal query result"""
        result = self.query(question, k=k, include_sources=True)
        result['verification'] = '{"verified": true, "issues": [], "suggestion": ""}'
        return result


class MockConversationalPipeline(MockRAGPipeline):
    """Mock conversational pipeline with memory"""

    def __init__(self, retriever, memory_size: int = 5):
        super().__init__(retriever)
        self.conversation_history = []
        self.memory_size = memory_size

    def query(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Query with conversation tracking"""
        result = super().query(question, k=k, include_sources=include_sources)

        # Update conversation history
        self.conversation_history.append({
            'question': question,
            'answer': result['answer']
        })

        # Keep only recent history
        if len(self.conversation_history) > self.memory_size:
            self.conversation_history = self.conversation_history[-self.memory_size:]

        return result

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


if __name__ == "__main__":
    from mock_retriever import MockRetriever
    from knowledge_processor import build_knowledge_base
    from pathlib import Path

    # Build knowledge base
    kb_path = Path(__file__).parent.parent.parent / "data" / "raw" / "knowledge_base.json"
    documents = build_knowledge_base(str(kb_path))

    # Create mock retriever
    retriever = MockRetriever()
    retriever.add_documents(documents)

    # Create mock pipeline
    pipeline = MockRAGPipeline(retriever)

    # Test queries
    test_questions = [
        "What technologies are you proficient in?",
        "Tell me about your projects",
        "What is your education?"
    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        result = pipeline.query(question, k=3)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
