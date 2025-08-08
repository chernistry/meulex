# Frequently Asked Questions

## General Questions

### What is Meulex?
Meulex is an AI-powered assistant that integrates directly with Slack to provide intelligent responses based on your organization's documents and knowledge base.

### How does it work?
Meulex uses retrieval-augmented generation (RAG) to find relevant information from your documents and generates contextual responses using large language models.

### Is it secure?
Yes, Meulex includes enterprise-grade security features including PII masking, request verification, and audit logging.

## Technical Questions

### What vector database does it use?
Meulex uses Qdrant as its vector database for efficient similarity search and document retrieval.

### Which embedding model is supported?
Currently, Meulex supports Jina AI embeddings, with plans to add support for other providers.

### Can I use my own LLM?
Yes, Meulex supports any OpenAI-compatible API endpoint, including local models via Ollama.

### How do I ingest documents?
You can ingest documents using the CLI:
```bash
meulex ingest-file document.pdf
meulex ingest-directory ./docs
```

## Troubleshooting

### The bot is not responding
1. Check that the Slack bot token is correctly configured
2. Verify the signing secret matches your Slack app
3. Ensure the bot has the necessary permissions in your workspace

### Embeddings are failing
1. Verify your Jina API key is valid
2. Check network connectivity to the Jina API
3. Consider using the mock embedder for testing

### Vector search returns no results
1. Ensure documents have been properly ingested
2. Check that the collection exists in Qdrant
3. Verify the embedding dimensions match

## Performance

### How fast is document retrieval?
Typical retrieval times are under 100ms for collections with up to 100k documents.

### What's the maximum document size?
By default, documents are limited to 10MB and 50k characters. This can be configured.

### How many concurrent users can it handle?
The system is designed to handle hundreds of concurrent users with proper scaling.
