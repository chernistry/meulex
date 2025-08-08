"""Integration tests for the complete API."""

import json
import time
import pytest
from fastapi.testclient import TestClient

from meulex.api.app import app


class TestAPIIntegration:
    """Integration tests for the complete API."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mock environment."""
        import os
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        os.environ["USE_MOCK_LLM"] = "true"
        os.environ["ENABLE_CACHE"] = "false"  # Disable cache for consistent testing
        
        return TestClient(app)
    
    def test_health_endpoints(self, client):
        """Test all health endpoints."""
        # Basic health check
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "timestamp" in data
        
        # Ready check
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        
        # Live check
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_info_endpoint(self, client):
        """Test service info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "meulex"
        assert "version" in data
        assert "features" in data
        assert isinstance(data["features"], dict)
        
        # Check expected features
        features = data["features"]
        assert "hybrid_retrieval" in features
        assert "reranker" in features
        assert "metrics" in features
        assert "tracing" in features
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus format
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        metrics_text = response.text
        assert "meulex_" in metrics_text
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text
    
    def test_embed_endpoint(self, client):
        """Test document embedding endpoint."""
        payload = {
            "content": "This is a test document about machine learning and AI.",
            "metadata": {
                "source": "test_doc.txt",
                "category": "technology"
            }
        }
        
        response = client.post("/embed", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["count"] == 1
        assert len(data["ids"]) == 1
        assert isinstance(data["ids"][0], str)
    
    def test_embed_endpoint_validation(self, client):
        """Test embed endpoint input validation."""
        # Missing content
        response = client.post("/embed", json={})
        assert response.status_code == 422
        
        # Empty content
        response = client.post("/embed", json={"content": ""})
        assert response.status_code == 422
        
        # Content too long
        long_content = "x" * 100000  # Exceeds max length
        response = client.post("/embed", json={"content": long_content})
        assert response.status_code == 422
    
    def test_chat_endpoint_basic(self, client):
        """Test basic chat functionality."""
        # First, add some content to search
        embed_payload = {
            "content": "Meulex is a RAG system that combines dense and sparse retrieval with LLM generation.",
            "metadata": {"source": "meulex_docs.txt"}
        }
        client.post("/embed", json=embed_payload)
        
        # Now test chat
        chat_payload = {
            "question": "What is Meulex?",
            "top_k": 3
        }
        
        response = client.post("/chat", json=chat_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data
        
        # Check answer
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        
        # Check sources
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) <= 3
        
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "source" in source
            assert "score" in source
            assert isinstance(source["score"], (int, float))
        
        # Check metadata
        metadata = data["metadata"]
        assert "processing_time" in metadata
        assert "model_used" in metadata
        assert "retrieval_stats" in metadata
        assert metadata["success"] is True
    
    def test_chat_endpoint_with_history(self, client):
        """Test chat with conversation history."""
        payload = {
            "question": "What are the benefits?",
            "history": [
                {"role": "user", "content": "Tell me about RAG systems"},
                {"role": "assistant", "content": "RAG systems combine retrieval and generation..."}
            ],
            "top_k": 2,
            "temperature": 0.5
        }
        
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert data["metadata"]["success"] is True
    
    def test_chat_endpoint_validation(self, client):
        """Test chat endpoint input validation."""
        # Missing question
        response = client.post("/chat", json={})
        assert response.status_code == 422
        
        # Empty question
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 422
        
        # Question too long
        long_question = "x" * 3000
        response = client.post("/chat", json={"question": long_question})
        assert response.status_code == 422
        
        # Invalid top_k
        response = client.post("/chat", json={"question": "test", "top_k": 0})
        assert response.status_code == 422
        
        response = client.post("/chat", json={"question": "test", "top_k": 25})
        assert response.status_code == 422
        
        # Invalid temperature
        response = client.post("/chat", json={"question": "test", "temperature": -0.1})
        assert response.status_code == 422
        
        response = client.post("/chat", json={"question": "test", "temperature": 2.1})
        assert response.status_code == 422
    
    def test_slack_events_url_verification(self, client):
        """Test Slack URL verification challenge."""
        payload = {
            "token": "test_token",
            "challenge": "test_challenge_12345",
            "type": "url_verification"
        }
        
        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=mock_signature",
            "Content-Type": "application/json"
        }
        
        response = client.post("/slack/events", json=payload, headers=headers)
        
        # Should succeed with mock signature (no secret configured)
        assert response.status_code == 200
        data = response.json()
        assert data["challenge"] == "test_challenge_12345"
    
    def test_slack_events_validation(self, client):
        """Test Slack events validation."""
        # Missing headers
        response = client.post("/slack/events", json={"type": "url_verification"})
        assert response.status_code == 400
        
        # Invalid payload
        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=mock_signature"
        }
        
        response = client.post("/slack/events", json={"invalid": "payload"}, headers=headers)
        assert response.status_code == 400
    
    def test_ingest_to_chat_workflow(self, client):
        """Test complete workflow from ingestion to chat."""
        # Step 1: Ingest multiple documents
        documents = [
            {
                "content": "Python is a versatile programming language used for web development, data science, and automation.",
                "metadata": {"source": "python_intro.txt", "category": "programming"}
            },
            {
                "content": "Machine learning models can be trained using Python libraries like TensorFlow and PyTorch.",
                "metadata": {"source": "ml_guide.txt", "category": "ai"}
            },
            {
                "content": "FastAPI is a modern web framework for building APIs with Python, featuring automatic documentation.",
                "metadata": {"source": "fastapi_docs.txt", "category": "web"}
            }
        ]
        
        ingested_ids = []
        for doc in documents:
            response = client.post("/embed", json=doc)
            assert response.status_code == 200
            data = response.json()
            ingested_ids.extend(data["ids"])
        
        assert len(ingested_ids) == 3
        
        # Step 2: Query the ingested content
        queries = [
            "What is Python used for?",
            "How do you train machine learning models?",
            "What is FastAPI?"
        ]
        
        for query in queries:
            response = client.post("/chat", json={"question": query, "top_k": 2})
            assert response.status_code == 200
            
            data = response.json()
            assert len(data["answer"]) > 0
            assert len(data["sources"]) > 0
            
            # Should find relevant sources
            sources_text = " ".join(source["text"] for source in data["sources"])
            if "Python" in query:
                assert "python" in sources_text.lower()
            elif "machine learning" in query:
                assert any(term in sources_text.lower() for term in ["machine", "learning", "models"])
            elif "FastAPI" in query:
                assert "fastapi" in sources_text.lower()
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test method not allowed
        response = client.get("/chat")  # Should be POST
        assert response.status_code == 405
        
        # Test malformed JSON
        response = client.post(
            "/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/chat")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        
        # Should have security headers
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"
        
        assert "x-frame-options" in headers
        assert headers["x-frame-options"] == "DENY"
        
        assert "x-xss-protection" in headers
        assert headers["x-xss-protection"] == "1; mode=block"
    
    def test_request_id_tracking(self, client):
        """Test request ID is tracked in responses."""
        response = client.get("/health")
        
        # Should have request ID in headers
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0


@pytest.mark.asyncio
class TestAPIPerformance:
    """Performance tests for the API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        import os
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        os.environ["USE_MOCK_LLM"] = "true"
        
        return TestClient(app)
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond in < 100ms
    
    def test_embed_endpoint_performance(self, client):
        """Test embed endpoint performance."""
        payload = {
            "content": "This is a test document for performance testing."
        }
        
        start_time = time.time()
        response = client.post("/embed", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should complete in < 1 second
    
    def test_chat_endpoint_performance(self, client):
        """Test chat endpoint performance."""
        # First ingest some content
        client.post("/embed", json={
            "content": "Performance test document about fast response times."
        })
        
        payload = {
            "question": "What is this about?",
            "top_k": 3
        }
        
        start_time = time.time()
        response = client.post("/chat", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should complete reasonably quickly with mock providers
        assert (end_time - start_time) < 2.0  # Should complete in < 2 seconds
        
        # Check reported processing time
        data = response.json()
        processing_time = data["metadata"]["processing_time"]
        assert processing_time < 1.0  # Internal processing should be fast
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)
        
        # All should have unique request IDs
        request_ids = [response.headers.get("x-request-id") for response in responses]
        assert len(set(request_ids)) == 10  # All unique
