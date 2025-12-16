"""
Mock Retriever for Testing Without API

Uses simple keyword matching instead of embeddings
No API costs - runs completely locally
"""

import numpy as np
from typing import List, Dict, Any
import pickle
from pathlib import Path
import re


class MockRetriever:
    """
    Simple keyword-based retriever for testing
    No API calls needed
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embedding_model = "mock-keyword-matching"

    def add_documents(self, documents: List[Any]):
        """Add documents to the retriever"""
        print(f"Adding {len(documents)} documents to mock retriever...")

        for doc in documents:
            self.documents.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'keywords': self._extract_keywords(doc.content)
            })

        print(f"Total documents: {len(self.documents)}")

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        # Simple keyword extraction
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        # Filter common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'with', 'has', 'have', 'from', 'this', 'that', 'what', 'your', 'more', 'will', 'home', 'about', 'which', 'their', 'there', 'been', 'many', 'some'}
        keywords = {w for w in words if w not in stop_words}
        return keywords

    def _calculate_score(self, query_keywords: set, doc_keywords: set) -> float:
        """Calculate similarity score based on keyword overlap"""
        if not query_keywords or not doc_keywords:
            return 0.0

        # Jaccard similarity
        intersection = len(query_keywords & doc_keywords)
        union = len(query_keywords | doc_keywords)

        if union == 0:
            return 0.0

        return intersection / union

    def _detect_query_intent(self, query: str) -> str:
        """Detect what category of information the query is asking for"""
        query_lower = query.lower()

        # Project-related queries
        if any(kw in query_lower for kw in ['project', 'built', 'developed', 'created', 'worked on']):
            return 'projects'

        # Experience-related queries (check before skills to prioritize)
        if any(kw in query_lower for kw in ['experience', 'work experience', 'work history', 'job', 'internship', 'company', 'allianz', 'penn state', 'employment']):
            return 'experience'

        # Skills-related queries
        if any(kw in query_lower for kw in ['skill', 'technology', 'technologies', 'proficient', 'know', 'framework', 'language', 'tools', 'tech stack']):
            return 'skills'

        # Education-related queries
        if any(kw in query_lower for kw in ['education', 'school', 'university', 'degree', 'study', 'major', 'graduate', 'ucsd', 'uc san diego']):
            return 'education'

        # Contact-related queries
        if any(kw in query_lower for kw in ['contact', 'email', 'phone', 'reach', 'linkedin', 'github']):
            return 'contact'

        # About/Personal info queries
        if any(kw in query_lower for kw in ['who are you', 'tell me about yourself', 'introduce', 'name', 'location', 'where are you']):
            return 'about'

        return 'general'

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents using keyword matching with intent-based boosting
        """
        if not self.documents:
            return []

        query_keywords = self._extract_keywords(query)
        query_intent = self._detect_query_intent(query)
        query_lower = query.lower()
        doc_content_lower = None  # Will be set in loop

        # Calculate scores for all documents
        scored_docs = []
        for doc in self.documents:
            doc_content_lower = doc['content'].lower()
            base_score = self._calculate_score(query_keywords, doc['keywords'])

            # Boost score based on intent matching
            doc_category = doc['metadata'].get('category', 'other')
            doc_type = doc['metadata'].get('type', 'unknown')

            boost = 1.0
            content_bonus = 0.0  # Additional score based on content matching

            # Special handling for work experience queries
            if query_intent == 'experience':
                # Check if document contains work experience keywords
                has_work_exp_keywords = any(kw in doc_content_lower for kw in ['work experience', 'professional internship', 'three professional', 'allianz', 'penn state', 'qingdao engineering'])
                
                # Add content-based bonus for work experience queries
                if 'work experience' in query_lower:
                    if 'work experience' in doc_content_lower:
                        content_bonus += 0.3
                    if 'three professional internships' in doc_content_lower or 'three professional' in doc_content_lower:
                        content_bonus += 0.2
                
                if doc_category == 'experience' or doc_type == 'experience':
                    boost = 3.5
                    if has_work_exp_keywords:
                        boost = 6.0  # Very strong boost for work experience content
                        content_bonus += 0.2
                elif has_work_exp_keywords:
                    boost = 5.0  # Boost even if category doesn't match
                    content_bonus += 0.15
                elif doc_category == 'skills':
                    boost = 0.15  # Strongly penalize skills docs for experience queries
                else:
                    boost = 1.0
            # Strong boost for category match
            elif query_intent == doc_category:
                boost = 3.0  # Increased boost for exact category match
            # Moderate boost for related categories
            elif query_intent == 'projects' and doc_type == 'project':
                boost = 3.0
            elif query_intent == 'skills' and doc_type in ['skills', 'soft_skills']:
                boost = 2.0
            elif query_intent == 'education' and doc_type == 'education':
                boost = 3.0
            elif query_intent == 'contact' and doc_type == 'contact':
                boost = 3.0

            # Penalize keyword_mapping documents slightly (prefer richer content)
            if doc_type == 'keyword_mapping':
                boost *= 0.7

            # Prefer Q&A documents that have question matching
            if doc_type in ['project', 'experience', 'skills', 'education', 'career_goals', 'achievement', 'personal_info']:
                question = doc['metadata'].get('question', '').lower()
                if question:
                    # Check if query words appear in the question
                    question_keywords = self._extract_keywords(question)
                    question_overlap = len(query_keywords & question_keywords)
                    if question_overlap > 0:
                        boost *= (1 + question_overlap * 0.3)

            final_score = (base_score + content_bonus) * boost
            scored_docs.append((doc, final_score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for doc, score in scored_docs[:k]:
            if score >= threshold:
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': min(score * 2, 1.0)  # Scale up scores for better display
                })

        return results

    def save(self, path: str):
        """Save retriever state to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'documents': self.documents,
            'embedding_model': self.embedding_model
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Mock retriever saved to {path}")

    def load(self, path: str):
        """Load retriever state from disk"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.documents = state['documents']
        self.embedding_model = state.get('embedding_model', 'mock-keyword-matching')

        print(f"Mock retriever loaded from {path}")
        print(f"Documents: {len(self.documents)}")

    def get_category_stats(self) -> Dict[str, int]:
        """Get document count by category"""
        stats = {}
        for doc in self.documents:
            category = doc['metadata'].get('category', 'other')
            stats[category] = stats.get(category, 0) + 1
        return stats


if __name__ == "__main__":
    # Test the mock retriever
    from knowledge_processor import build_knowledge_base

    kb_path = Path(__file__).parent.parent.parent / "data" / "raw" / "knowledge_base.json"
    documents = build_knowledge_base(str(kb_path))

    retriever = MockRetriever()
    retriever.add_documents(documents)

    # Test queries
    test_queries = [
        "What technologies are you proficient in?",
        "Tell me about your RAG project",
        "What is your education background?",
        "How can I contact you?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=3)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.3f}):")
            print(f"Type: {result['metadata'].get('type')}")
            print(f"Content: {result['content'][:100]}...")

    # Save
    save_path = Path(__file__).parent.parent.parent / "data" / "processed" / "mock_retriever.pkl"
    retriever.save(str(save_path))
