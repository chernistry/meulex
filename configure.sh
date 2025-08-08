#!/bin/bash

# Meulex Configuration Helper
set -e

echo "🔧 Meulex Configuration Helper"
echo "=============================="

# Check if .env exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists. Backing up to .env.backup"
    cp .env .env.backup
fi

# Copy template
if [ -f .env.example ]; then
    cp .env.example .env
    echo "✅ Created .env from template"
else
    echo "❌ .env.example not found. Creating basic configuration..."
    cat > .env << 'EOF'
# Basic Meulex Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development

# Use mock providers for quick start
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true

# Enable all features
ENABLE_CACHE=true
ENABLE_SPARSE_RETRIEVAL=true
ENABLE_RERANKER=true
ENABLE_RATE_LIMITING=true
ENABLE_SECURITY_HEADERS=true

# Default settings
DEFAULT_TOP_K=3
TEMPERATURE=0.7
EOF
fi

echo ""
echo "🚀 Configuration Options:"
echo ""

# Ask user for configuration preference
echo "Choose your configuration:"
echo "1) Development (mock providers, no API keys needed)"
echo "2) Production (real providers, requires API keys)"
echo "3) Slack-ready (includes Slack configuration)"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "📝 Configuring for development..."
        cat >> .env << 'EOF'

# Development Configuration
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
LLM_PROVIDER=mock
FALLBACK_LLM_PROVIDER=mock
EMBEDDER_NAME=mock
EOF
        echo "✅ Development configuration applied"
        echo "💡 You can start the system immediately with: meulex api start"
        ;;
    
    2)
        echo "📝 Configuring for production..."
        echo ""
        echo "⚠️  You'll need to configure API keys in .env file:"
        echo "   - OPENAI_API_KEY=your-key"
        echo "   - JINA_API_KEY=your-key (optional, can use mock)"
        echo ""
        
        cat >> .env << 'EOF'

# Production Configuration
USE_MOCK_EMBEDDINGS=false
USE_MOCK_LLM=false
LLM_PROVIDER=openai
FALLBACK_LLM_PROVIDER=mock
EMBEDDER_NAME=jina

# TODO: Add your API keys
# OPENAI_API_KEY=sk-your-openai-key
# JINA_API_KEY=jina_your-jina-key
EOF
        echo "✅ Production configuration template applied"
        echo "📝 Please edit .env file to add your API keys"
        ;;
    
    3)
        echo "📝 Configuring for Slack integration..."
        echo ""
        echo "⚠️  You'll need to configure Slack app credentials:"
        echo "   - SLACK_BOT_TOKEN=xoxb-your-token"
        echo "   - SLACK_SIGNING_SECRET=your-secret"
        echo "   - SLACK_BOT_USER_ID=U1234567890"
        echo ""
        
        cat >> .env << 'EOF'

# Slack-Ready Configuration
USE_MOCK_EMBEDDINGS=false
USE_MOCK_LLM=false
LLM_PROVIDER=openai
FALLBACK_LLM_PROVIDER=mock
EMBEDDER_NAME=jina

# TODO: Add your API keys
# OPENAI_API_KEY=sk-your-openai-key
# JINA_API_KEY=jina_your-jina-key

# TODO: Add your Slack app credentials
# SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
# SLACK_SIGNING_SECRET=your-slack-signing-secret
# SLACK_BOT_USER_ID=U1234567890
EOF
        echo "✅ Slack configuration template applied"
        echo "📝 Please edit .env file to add your API keys and Slack credentials"
        echo "🔗 See README.meulex.md for Slack app setup instructions"
        ;;
    
    *)
        echo "❌ Invalid choice. Using development configuration as default."
        cat >> .env << 'EOF'

# Default Development Configuration
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
EOF
        ;;
esac

echo ""
echo "📁 Configuration files created:"
echo "   - .env (your configuration)"
echo "   - .env.example (template for reference)"
if [ -f .env.backup ]; then
    echo "   - .env.backup (your previous configuration)"
fi

echo ""
echo "🔍 Next steps:"
echo "1. Review and edit .env file if needed"
echo "2. Start infrastructure: docker compose up -d"
echo "3. Start Meulex: meulex api start"
echo "4. Test the system: curl http://localhost:8000/health"

echo ""
echo "📚 For more information, see README.meulex.md"
echo "✅ Configuration complete!"
