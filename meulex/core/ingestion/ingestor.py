"""Document ingestion and processing."""

import asyncio
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from meulex.config.settings import Settings
from meulex.core.chunking.text_splitter import TextChunk, create_text_splitter
from meulex.core.embeddings.base import BaseEmbedder
from meulex.core.vector.base import VectorStore
from meulex.observability import DOCUMENTS_RETRIEVED, get_tracer
from meulex.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class DocumentIngestor:
    """Document ingestion and processing pipeline."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        settings: Settings
    ) -> None:
        """Initialize document ingestor.
        
        Args:
            embedder: Embeddings provider
            vector_store: Vector store
            settings: Application settings
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings
        
        # Create text splitter
        self.text_splitter = create_text_splitter(
            strategy="recursive",
            settings=settings
        )
        
        logger.info(
            "Document ingestor initialized",
            extra={
                "embedder": embedder.model_name,
                "vector_store": type(vector_store).__name__,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
        )
    
    def _load_file(self, file_path: Path) -> str:
        """Load content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.settings.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size:
                raise ValidationError(
                    f"File too large: {file_size} bytes (max: {max_size})",
                    {"file_path": str(file_path), "file_size": file_size}
                )
            
            # Check file extension
            file_ext = file_path.suffix.lower().lstrip('.')
            if file_ext not in self.settings.supported_file_types:
                raise ValidationError(
                    f"Unsupported file type: {file_ext}",
                    {
                        "file_path": str(file_path),
                        "supported_types": self.settings.supported_file_types
                    }
                )
            
            # Load content based on file type
            if file_ext in ["txt", "md", "py", "js", "html", "css", "json", "yaml", "yml"]:
                # Text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # For now, treat other supported types as text
                # In production, you'd want proper parsers for PDF, DOCX, etc.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Check content length
            if len(content) > self.settings.max_content_length:
                raise ValidationError(
                    f"Content too long: {len(content)} chars (max: {self.settings.max_content_length})",
                    {"file_path": str(file_path), "content_length": len(content)}
                )
            
            return content
            
        except UnicodeDecodeError as e:
            raise ValidationError(
                f"Failed to decode file: {e}",
                {"file_path": str(file_path)}
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to load file: {e}",
                {"file_path": str(file_path)}
            )
    
    def _generate_document_id(self, content: str, source: str) -> str:
        """Generate a deterministic document ID.
        
        Args:
            content: Document content
            source: Document source
            
        Returns:
            Document ID
        """
        # Create hash from content and source for deterministic IDs
        content_hash = hashlib.sha256(
            f"{source}:{content}".encode('utf-8')
        ).hexdigest()[:16]
        
        return f"doc_{content_hash}"
    
    async def ingest_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Ingest a single document.
        
        Args:
            content: Document content
            source: Document source (file path, URL, etc.)
            metadata: Optional additional metadata
            
        Returns:
            Ingestion result
        """
        with tracer.start_as_current_span("ingest_document") as span:
            span.set_attribute("source", source)
            span.set_attribute("content_length", len(content))
            
            try:
                # Validate content
                if not content.strip():
                    raise ValidationError("Document content is empty")
                
                # Generate document ID
                doc_id = self._generate_document_id(content, source)
                
                # Prepare base metadata
                base_metadata = {
                    "source": source,
                    "document_id": doc_id,
                    "ingested_at": str(asyncio.get_event_loop().time()),
                    "content_length": len(content),
                    **(metadata or {})
                }
                
                # Split into chunks
                chunks = self.text_splitter.split_text(content, base_metadata)
                
                if not chunks:
                    raise ValidationError("No chunks generated from document")
                
                span.set_attribute("chunk_count", len(chunks))
                
                # Generate embeddings
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self.embedder.embed_async_many(chunk_texts)
                
                # Prepare chunk metadata and IDs
                chunk_ids = []
                chunk_metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_ids.append(chunk_id)
                    
                    chunk_metadata = {
                        **chunk.metadata,
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "chunk_index": i,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index
                    }
                    chunk_metadatas.append(chunk_metadata)
                
                # Store in vector database
                stored_ids = await self.vector_store.add_embeddings(
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
                
                # Record metrics
                DOCUMENTS_RETRIEVED.labels(strategy="ingestion").inc(len(chunks))
                
                result = {
                    "document_id": doc_id,
                    "source": source,
                    "chunk_count": len(chunks),
                    "chunk_ids": stored_ids,
                    "content_length": len(content),
                    "status": "success"
                }
                
                logger.info(
                    f"Ingested document: {source}",
                    extra={
                        "document_id": doc_id,
                        "chunk_count": len(chunks),
                        "content_length": len(content)
                    }
                )
                
                span.set_attribute("success", True)
                return result
                
            except Exception as e:
                logger.error(f"Failed to ingest document {source}: {e}")
                span.set_attribute("error", str(e))
                
                return {
                    "source": source,
                    "status": "error",
                    "error": str(e)
                }
    
    async def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingest a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Ingestion result
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Load file content
        content = self._load_file(file_path)
        
        # Prepare metadata
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "file_modified": file_path.stat().st_mtime
        }
        
        # Ingest document
        return await self.ingest_document(
            content=content,
            source=str(file_path),
            metadata=metadata
        )
    
    async def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*"
    ) -> Dict[str, Any]:
        """Ingest all files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            pattern: File pattern to match
            
        Returns:
            Ingestion results
        """
        with tracer.start_as_current_span("ingest_directory") as span:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                raise ValidationError(f"Directory not found: {directory_path}")
            
            if not directory_path.is_dir():
                raise ValidationError(f"Path is not a directory: {directory_path}")
            
            span.set_attribute("directory", str(directory_path))
            span.set_attribute("recursive", recursive)
            span.set_attribute("pattern", pattern)
            
            # Find files
            if recursive:
                files = list(directory_path.rglob(pattern))
            else:
                files = list(directory_path.glob(pattern))
            
            # Filter to only files with supported extensions
            supported_files = []
            for file_path in files:
                if file_path.is_file():
                    ext = file_path.suffix.lower().lstrip('.')
                    if ext in self.settings.supported_file_types:
                        supported_files.append(file_path)
            
            span.set_attribute("files_found", len(supported_files))
            
            if not supported_files:
                return {
                    "directory": str(directory_path),
                    "files_found": 0,
                    "files_processed": 0,
                    "results": [],
                    "status": "no_files"
                }
            
            # Process files
            results = []
            successful = 0
            failed = 0
            
            for file_path in supported_files:
                try:
                    result = await self.ingest_file(file_path)
                    results.append(result)
                    
                    if result["status"] == "success":
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    results.append({
                        "source": str(file_path),
                        "status": "error",
                        "error": str(e)
                    })
                    failed += 1
            
            summary = {
                "directory": str(directory_path),
                "files_found": len(supported_files),
                "files_processed": len(results),
                "successful": successful,
                "failed": failed,
                "results": results,
                "status": "completed"
            }
            
            logger.info(
                f"Ingested directory: {directory_path}",
                extra={
                    "files_found": len(supported_files),
                    "successful": successful,
                    "failed": failed
                }
            )
            
            span.set_attribute("successful", successful)
            span.set_attribute("failed", failed)
            
            return summary
    
    async def close(self) -> None:
        """Close the ingestor and cleanup resources."""
        try:
            await self.embedder.close()
            logger.info("Document ingestor closed")
        except Exception as e:
            logger.warning(f"Error closing ingestor: {e}")
