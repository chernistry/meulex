"""Prompt builder for RAG applications."""

import logging
from typing import List, Optional

from meulex.core.vector.base import Document
from meulex.llm.base import ChatMessage, LLMMode, MessageRole

logger = logging.getLogger(__name__)


class RAGPromptBuilder:
    """Prompt builder for RAG applications."""
    
    def __init__(self, max_context_tokens: int = 2000):
        """Initialize prompt builder.
        
        Args:
            max_context_tokens: Maximum tokens for context
        """
        self.max_context_tokens = max_context_tokens
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite your sources by referencing the document names or sections
4. Be concise but comprehensive in your responses
5. If multiple sources provide relevant information, synthesize them coherently

Context format: Each piece of context will be marked with [Source: filename] at the beginning."""
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters).
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _truncate_context(self, context_parts: List[str]) -> str:
        """Truncate context to fit within token limits.
        
        Args:
            context_parts: List of context strings
            
        Returns:
            Truncated context string
        """
        total_context = ""
        current_tokens = 0
        
        for part in context_parts:
            part_tokens = self._estimate_tokens(part)
            
            if current_tokens + part_tokens > self.max_context_tokens:
                # Try to fit partial content
                remaining_tokens = self.max_context_tokens - current_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space left
                    remaining_chars = remaining_tokens * 4
                    truncated_part = part[:remaining_chars] + "... [truncated]"
                    total_context += truncated_part + "\n\n"
                break
            
            total_context += part + "\n\n"
            current_tokens += part_tokens
        
        return total_context.strip()
    
    def build_rag_prompt(
        self,
        question: str,
        documents: List[Document],
        conversation_history: Optional[List[ChatMessage]] = None,
        mode: LLMMode = LLMMode.BALANCED
    ) -> List[ChatMessage]:
        """Build RAG prompt with context and question.
        
        Args:
            question: User question
            documents: Retrieved documents for context
            conversation_history: Optional conversation history
            mode: LLM generation mode
            
        Returns:
            List of chat messages
        """
        messages = []
        
        # Add system message
        system_message = self.system_prompt
        
        # Adjust system prompt based on mode
        if mode == LLMMode.CREATIVE:
            system_message += "\n\nBe creative and engaging in your responses while staying factual."
        elif mode == LLMMode.PRECISE:
            system_message += "\n\nBe precise and factual. Avoid speculation or creative interpretation."
        
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_message
        ))
        
        # Add conversation history if provided
        if conversation_history:
            # Limit history to avoid token overflow
            recent_history = conversation_history[-6:]  # Last 3 exchanges
            messages.extend(recent_history)
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i+1}")
            score = doc.score or 0.0
            
            context_part = f"[Source: {source}] (Relevance: {score:.3f})\n{doc.content}"
            context_parts.append(context_part)
        
        # Truncate context if needed
        context = self._truncate_context(context_parts)
        
        # Create the user message with context and question
        if context:
            user_content = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain sufficient information, please state that clearly."""
        else:
            user_content = f"""Question: {question}

I don't have any relevant context to answer this question. Please provide more specific information or rephrase your question."""
        
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=user_content
        ))
        
        logger.info(
            f"Built RAG prompt with {len(documents)} documents",
            extra={
                "question_length": len(question),
                "context_length": len(context),
                "estimated_tokens": self._estimate_tokens(user_content),
                "mode": mode.value,
                "has_history": bool(conversation_history)
            }
        )
        
        return messages
    
    def build_simple_prompt(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> List[ChatMessage]:
        """Build simple prompt without RAG context.
        
        Args:
            question: User question
            conversation_history: Optional conversation history
            
        Returns:
            List of chat messages
        """
        messages = []
        
        # Simple system message
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant. Answer questions clearly and concisely."
        ))
        
        # Add conversation history if provided
        if conversation_history:
            recent_history = conversation_history[-6:]  # Last 3 exchanges
            messages.extend(recent_history)
        
        # Add user question
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=question
        ))
        
        return messages
    
    def extract_citations(self, response_text: str) -> List[str]:
        """Extract citations from response text.
        
        Args:
            response_text: Generated response text
            
        Returns:
            List of cited sources
        """
        import re
        
        # Look for patterns like [Source: filename] or (Source: filename)
        citation_patterns = [
            r'\[Source:\s*([^\]]+)\]',
            r'\(Source:\s*([^\)]+)\)',
            r'according to ([^,\.\n]+)',
            r'based on ([^,\.\n]+)',
        ]
        
        citations = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            citations.update(matches)
        
        return list(citations)
