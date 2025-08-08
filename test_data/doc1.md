# Meulex Documentation

## Overview

Meulex is a Slack-native, compliance-aware agentic RAG copilot designed to provide intelligent assistance within Slack workspaces. It combines the power of retrieval-augmented generation with enterprise-grade security and compliance features.

## Key Features

### Slack Integration
- Native Slack bot integration
- Slash command support
- Real-time message processing
- Ephemeral and public responses

### RAG Capabilities
- Hybrid retrieval (dense + sparse)
- Document chunking and embedding
- Semantic search
- Citation tracking

### Security & Compliance
- PII masking and detection
- Request signature verification
- Rate limiting
- Audit logging

## Architecture

The system consists of several key components:

1. **API Layer**: FastAPI-based REST API with OpenAPI documentation
2. **Vector Store**: Qdrant for efficient similarity search
3. **Embeddings**: Jina AI for high-quality text embeddings
4. **LLM Integration**: OpenAI-compatible API with fallback support
5. **Observability**: Prometheus metrics and OpenTelemetry tracing

## Getting Started

To get started with Meulex:

1. Install dependencies: `pip install -e .`
2. Configure environment variables in `.env`
3. Start services: `docker compose up -d`
4. Run the API: `meulex api`
5. Ingest documents: `meulex ingest-directory ./docs`

## Configuration

Meulex uses environment variables for configuration. Key settings include:

- `OPENAI_API_KEY`: OpenAI API key for LLM
- `JINA_API_KEY`: Jina API key for embeddings
- `QDRANT_URL`: Qdrant vector database URL
- `SLACK_BOT_TOKEN`: Slack bot token
- `SLACK_SIGNING_SECRET`: Slack signing secret

## Support

For support and questions, please refer to the documentation or contact the development team.
