"""Text splitting and chunking utilities."""

import logging
import re
from typing import Any, Dict, List, Optional

import tiktoken
from pydantic import BaseModel

from meulex.config.settings import Settings
from meulex.observability import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class TextChunk(BaseModel):
    """Text chunk model."""
    
    content: str
    metadata: Dict[str, Any] = {}
    start_index: int = 0
    end_index: int = 0
    token_count: Optional[int] = None


class TextSplitter:
    """Base text splitter class."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        length_function: Optional[callable] = None
    ) -> None:
        """Initialize text splitter.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            length_function: Function to calculate text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to word count approximation
        return len(text.split())
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        raise NotImplementedError("Subclasses must implement split_text")


class RecursiveCharacterTextSplitter(TextSplitter):
    """Recursive character-based text splitter."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        length_function: Optional[callable] = None
    ) -> None:
        """Initialize recursive character text splitter.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
            keep_separator: Whether to keep separators in chunks
            length_function: Function to calculate text length
        """
        super().__init__(chunk_size, chunk_overlap, length_function)
        
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            " ",     # Words
            ""       # Characters
        ]
        self.keep_separator = keep_separator
    
    def _split_text_with_separator(
        self,
        text: str,
        separator: str
    ) -> List[str]:
        """Split text with a specific separator.
        
        Args:
            text: Text to split
            separator: Separator to use
            
        Returns:
            List of text parts
        """
        if separator:
            if self.keep_separator:
                # Split while keeping separator
                parts = text.split(separator)
                result = []
                for i, part in enumerate(parts):
                    if i > 0:
                        result.append(separator + part)
                    else:
                        result.append(part)
                return [p for p in result if p]
            else:
                return text.split(separator)
        else:
            # Character-level split
            return list(text)
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits into chunks with overlap.
        
        Args:
            splits: List of text splits
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # If adding this split would exceed chunk size, finalize current chunk
            if (current_length + split_length > self.chunk_size and 
                current_chunk):
                
                chunk_text = "".join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_length = 0
                overlap_parts = []
                
                # Add parts from the end of current chunk for overlap
                for part in reversed(current_chunk):
                    part_length = self.length_function(part)
                    if overlap_length + part_length <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_length += part_length
                    else:
                        break
                
                current_chunk = overlap_parts
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Split text into chunks recursively.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        with tracer.start_as_current_span("recursive_text_split") as span:
            span.set_attribute("text_length", len(text))
            span.set_attribute("chunk_size", self.chunk_size)
            span.set_attribute("chunk_overlap", self.chunk_overlap)
            
            if not text.strip():
                return []
            
            metadata = metadata or {}
            
            # Try each separator in order
            final_chunks = [text]
            
            for separator in self.separators:
                new_chunks = []
                
                for chunk in final_chunks:
                    if self.length_function(chunk) <= self.chunk_size:
                        new_chunks.append(chunk)
                    else:
                        # Split this chunk further
                        splits = self._split_text_with_separator(chunk, separator)
                        merged = self._merge_splits(splits)
                        new_chunks.extend(merged)
                
                final_chunks = new_chunks
                
                # Check if all chunks are small enough
                if all(self.length_function(chunk) <= self.chunk_size 
                       for chunk in final_chunks):
                    break
            
            # Create TextChunk objects
            chunks = []
            current_index = 0
            
            for i, chunk_text in enumerate(final_chunks):
                if not chunk_text.strip():
                    continue
                
                # Find the actual position in original text
                start_index = text.find(chunk_text, current_index)
                if start_index == -1:
                    start_index = current_index
                
                end_index = start_index + len(chunk_text)
                token_count = self._count_tokens(chunk_text)
                
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(final_chunks),
                    "token_count": token_count,
                    "char_count": len(chunk_text)
                }
                
                chunk = TextChunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count
                )
                
                chunks.append(chunk)
                current_index = end_index
            
            logger.info(
                f"Split text into {len(chunks)} chunks",
                extra={
                    "original_length": len(text),
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
                }
            )
            
            span.set_attribute("chunk_count", len(chunks))
            return chunks


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Markdown-aware text splitter."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        length_function: Optional[callable] = None
    ) -> None:
        """Initialize markdown text splitter.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            length_function: Function to calculate text length
        """
        # Markdown-specific separators
        separators = [
            "\n#{1,6} ",  # Headers
            "\n\n",       # Paragraphs
            "\n",         # Lines
            " ",          # Words
            ""            # Characters
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=length_function
        )
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Split markdown text preserving structure.
        
        Args:
            text: Markdown text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks with markdown context
        """
        # Extract headers for context
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headers = []
        
        for match in header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2)
            position = match.start()
            headers.append({
                "level": level,
                "title": title,
                "position": position
            })
        
        # Split text normally
        chunks = super().split_text(text, metadata)
        
        # Add header context to chunks
        for chunk in chunks:
            # Find the most relevant header for this chunk
            relevant_headers = []
            for header in headers:
                if header["position"] <= chunk.start_index:
                    relevant_headers.append(header)
            
            if relevant_headers:
                # Get the hierarchy of headers
                header_context = []
                current_level = 0
                
                for header in relevant_headers:
                    if header["level"] <= current_level or current_level == 0:
                        header_context = [header]
                        current_level = header["level"]
                    elif header["level"] > current_level:
                        header_context.append(header)
                        current_level = header["level"]
                
                chunk.metadata["headers"] = [h["title"] for h in header_context]
                chunk.metadata["header_hierarchy"] = header_context
        
        return chunks


def create_text_splitter(
    strategy: str = "recursive",
    settings: Optional[Settings] = None,
    **kwargs
) -> TextSplitter:
    """Create a text splitter instance.
    
    Args:
        strategy: Splitting strategy ("recursive", "markdown")
        settings: Optional settings to use for defaults
        **kwargs: Additional arguments for the splitter
        
    Returns:
        Text splitter instance
    """
    if settings:
        kwargs.setdefault("chunk_size", settings.chunk_size)
        kwargs.setdefault("chunk_overlap", settings.chunk_overlap)
    
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(**kwargs)
    elif strategy == "markdown":
        return MarkdownTextSplitter(**kwargs)
    else:
        raise ValueError(f"Unknown splitting strategy: {strategy}")
