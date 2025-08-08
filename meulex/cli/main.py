"""Main CLI application for Meulex."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from meulex.config.settings import get_settings
from meulex.core.embeddings.factory import create_embedder
from meulex.core.ingestion.ingestor import DocumentIngestor
from meulex.core.vector.qdrant_store import QdrantStore

# Create Typer app
app = typer.Typer(
    name="meulex",
    help="Meulex: Slack-native, compliance-aware agentic RAG copilot",
    add_completion=False
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    log_level: str = typer.Option("info", help="Log level")
) -> None:
    """Start the API server."""
    typer.echo(f"Starting Meulex API server on {host}:{port}")
    
    uvicorn.run(
        "meulex.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level.lower()
    )


@app.command()
def ingest_file(
    file_path: Path = typer.Argument(..., help="Path to file to ingest"),
    collection: Optional[str] = typer.Option(None, help="Collection name"),
    embedder: Optional[str] = typer.Option(None, help="Embedder to use"),
    vector_store: Optional[str] = typer.Option(None, help="Vector store to use")
) -> None:
    """Ingest a single file."""
    asyncio.run(_ingest_file_async(file_path, collection, embedder, vector_store))


@app.command()
def ingest_directory(
    directory_path: Path = typer.Argument(..., help="Path to directory to ingest"),
    collection: Optional[str] = typer.Option(None, help="Collection name"),
    embedder: Optional[str] = typer.Option(None, help="Embedder to use"),
    vector_store: Optional[str] = typer.Option(None, help="Vector store to use"),
    recursive: bool = typer.Option(True, help="Search recursively"),
    pattern: str = typer.Option("*", help="File pattern to match")
) -> None:
    """Ingest all files in a directory."""
    asyncio.run(_ingest_directory_async(
        directory_path, collection, embedder, vector_store, recursive, pattern
    ))


async def _ingest_file_async(
    file_path: Path,
    collection: Optional[str],
    embedder_name: Optional[str],
    vector_store_name: Optional[str]
) -> None:
    """Async implementation of file ingestion."""
    settings = get_settings()
    
    # Override settings if provided
    if collection:
        settings.collection_name = collection
    if embedder_name:
        settings.embedder_name = embedder_name
    if vector_store_name:
        settings.vector_store = vector_store_name
    
    typer.echo(f"Ingesting file: {file_path}")
    typer.echo(f"Collection: {settings.collection_name}")
    typer.echo(f"Embedder: {settings.embedder_name}")
    typer.echo(f"Vector store: {settings.vector_store}")
    
    try:
        # Initialize components
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        
        # Ensure collection exists
        await vector_store.create_collection()
        
        # Create ingestor
        ingestor = DocumentIngestor(embedder, vector_store, settings)
        
        # Ingest file
        result = await ingestor.ingest_file(file_path)
        
        # Print result
        typer.echo("\nIngestion Result:")
        typer.echo(json.dumps(result, indent=2))
        
        if result["status"] == "success":
            typer.echo(f"✅ Successfully ingested {result['chunk_count']} chunks")
        else:
            typer.echo(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
        
        # Cleanup
        await ingestor.close()
        await vector_store.close()
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


async def _ingest_directory_async(
    directory_path: Path,
    collection: Optional[str],
    embedder_name: Optional[str],
    vector_store_name: Optional[str],
    recursive: bool,
    pattern: str
) -> None:
    """Async implementation of directory ingestion."""
    settings = get_settings()
    
    # Override settings if provided
    if collection:
        settings.collection_name = collection
    if embedder_name:
        settings.embedder_name = embedder_name
    if vector_store_name:
        settings.vector_store = vector_store_name
    
    typer.echo(f"Ingesting directory: {directory_path}")
    typer.echo(f"Collection: {settings.collection_name}")
    typer.echo(f"Embedder: {settings.embedder_name}")
    typer.echo(f"Vector store: {settings.vector_store}")
    typer.echo(f"Recursive: {recursive}")
    typer.echo(f"Pattern: {pattern}")
    
    try:
        # Initialize components
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        
        # Ensure collection exists
        await vector_store.create_collection()
        
        # Create ingestor
        ingestor = DocumentIngestor(embedder, vector_store, settings)
        
        # Ingest directory
        result = await ingestor.ingest_directory(
            directory_path, recursive=recursive, pattern=pattern
        )
        
        # Print summary
        typer.echo("\nIngestion Summary:")
        typer.echo(f"Files found: {result['files_found']}")
        typer.echo(f"Files processed: {result['files_processed']}")
        typer.echo(f"Successful: {result['successful']}")
        typer.echo(f"Failed: {result['failed']}")
        
        # Print detailed results
        if result['results']:
            typer.echo("\nDetailed Results:")
            for file_result in result['results']:
                status_icon = "✅" if file_result['status'] == 'success' else "❌"
                source = file_result['source']
                
                if file_result['status'] == 'success':
                    chunk_count = file_result.get('chunk_count', 0)
                    typer.echo(f"{status_icon} {source} ({chunk_count} chunks)")
                else:
                    error = file_result.get('error', 'Unknown error')
                    typer.echo(f"{status_icon} {source} - Error: {error}")
        
        if result['failed'] > 0:
            typer.echo(f"\n⚠️  {result['failed']} files failed to process")
            raise typer.Exit(1)
        else:
            typer.echo(f"\n✅ Successfully processed all {result['successful']} files")
        
        # Cleanup
        await ingestor.close()
        await vector_store.close()
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show system information."""
    settings = get_settings()
    
    typer.echo("Meulex System Information")
    typer.echo("=" * 30)
    typer.echo(f"Service: {settings.service_name}")
    typer.echo(f"Version: {settings.service_version}")
    typer.echo(f"Environment: {settings.environment}")
    typer.echo(f"Debug: {settings.debug}")
    typer.echo()
    typer.echo("Configuration:")
    typer.echo(f"  LLM Provider: {settings.llm_provider}")
    typer.echo(f"  LLM Model: {settings.llm_model}")
    typer.echo(f"  Embedder: {settings.embedder_name}")
    typer.echo(f"  Vector Store: {settings.vector_store}")
    typer.echo(f"  Collection: {settings.collection_name}")
    typer.echo()
    typer.echo("Features:")
    typer.echo(f"  Hybrid Retrieval: {settings.enable_hybrid_retrieval}")
    typer.echo(f"  Reranker: {settings.enable_reranker}")
    typer.echo(f"  Streaming: {settings.enable_streaming}")
    typer.echo(f"  Metrics: {settings.metrics_enabled}")
    typer.echo(f"  Tracing: {settings.tracing_enabled}")


@app.command()
def test_connection() -> None:
    """Test connections to external services."""
    asyncio.run(_test_connection_async())


async def _test_connection_async() -> None:
    """Async implementation of connection testing."""
    settings = get_settings()
    
    typer.echo("Testing connections...")
    typer.echo()
    
    # Test vector store
    typer.echo("🔍 Testing vector store connection...")
    try:
        vector_store = QdrantStore(settings)
        info = await vector_store.get_collection_info()
        typer.echo(f"✅ Vector store connected: {info}")
        await vector_store.close()
    except Exception as e:
        typer.echo(f"❌ Vector store connection failed: {e}")
    
    # Test embedder
    typer.echo("\n🧠 Testing embedder connection...")
    try:
        embedder = create_embedder(settings)
        test_embedding = await embedder.embed_async_single("test")
        typer.echo(f"✅ Embedder connected: {len(test_embedding)} dimensions")
        await embedder.close()
    except Exception as e:
        typer.echo(f"❌ Embedder connection failed: {e}")


if __name__ == "__main__":
    app()
