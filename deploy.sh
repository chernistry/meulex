#!/bin/bash

# Meulex Deployment Script
set -e

echo "🚀 Deploying Meulex RAG Copilot"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Parse command line arguments
PROFILE="default"
DETACH=false
BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --monitoring)
            PROFILE="monitoring"
            shift
            ;;
        --proxy)
            PROFILE="proxy"
            shift
            ;;
        --detach|-d)
            DETACH=true
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --monitoring    Deploy with Prometheus and Grafana"
            echo "  --proxy         Deploy with Nginx reverse proxy"
            echo "  --detach, -d    Run in detached mode"
            echo "  --build         Force rebuild of images"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs

# Set up environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Meulex Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production

# API Keys (replace with real values for production)
# OPENAI_API_KEY=sk-your-openai-key
# JINA_API_KEY=jina_your-jina-key
# SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
# SLACK_SIGNING_SECRET=your-slack-signing-secret
# METRICS_TOKEN=your-metrics-protection-token

# Mock providers (set to false for production)
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
EOF
    echo "⚠️  Please edit .env file with your API keys for production use"
fi

# Build command
BUILD_CMD=""
if [ "$BUILD" = true ]; then
    BUILD_CMD="--build"
fi

# Profile command
PROFILE_CMD=""
if [ "$PROFILE" != "default" ]; then
    PROFILE_CMD="--profile $PROFILE"
fi

# Detach command
DETACH_CMD=""
if [ "$DETACH" = true ]; then
    DETACH_CMD="-d"
fi

# Deploy the stack
echo "🐳 Starting Docker services..."
if [ "$PROFILE" = "monitoring" ]; then
    echo "📊 Including monitoring stack (Prometheus + Grafana)"
    docker-compose --profile monitoring up $BUILD_CMD $DETACH_CMD
elif [ "$PROFILE" = "proxy" ]; then
    echo "🔀 Including Nginx reverse proxy"
    docker-compose --profile proxy up $BUILD_CMD $DETACH_CMD
else
    docker-compose up $BUILD_CMD $DETACH_CMD
fi

# Wait for services to be ready
if [ "$DETACH" = true ]; then
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    echo "🔍 Checking service health..."
    
    # Check Meulex API
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Meulex API is healthy"
    else
        echo "❌ Meulex API is not responding"
    fi
    
    # Check Qdrant
    if curl -f http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is healthy"
    else
        echo "❌ Qdrant is not responding"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is healthy"
    else
        echo "❌ Redis is not responding"
    fi
    
    echo ""
    echo "🎉 Meulex is deployed and running!"
    echo ""
    echo "📍 Service URLs:"
    echo "   • API: http://localhost:8000"
    echo "   • API Docs: http://localhost:8000/docs"
    echo "   • Health: http://localhost:8000/health"
    echo "   • Metrics: http://localhost:8000/metrics"
    echo "   • Qdrant: http://localhost:6333"
    
    if [ "$PROFILE" = "monitoring" ]; then
        echo "   • Prometheus: http://localhost:9090"
        echo "   • Grafana: http://localhost:3000 (admin/admin)"
    fi
    
    echo ""
    echo "🚀 Quick test:"
    echo "   curl -X POST http://localhost:8000/chat \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"question\": \"What is Meulex?\"}'"
    echo ""
    echo "📚 Documentation: README.meulex.md"
fi

echo "✅ Deployment complete!"
