# 🔧 Meulex Troubleshooting Guide

## Common Issues and Solutions

### 1. Jina API Error: "Extra inputs are not permitted"

**Error Message:**
```
Jina API error: 422 - {'detail': '[RID: ...] Extra inputs are not permitted'}
```

**Cause:** The Jina API model doesn't support the `task` parameter that was being sent.

**Solutions:**

**Option A: Use Mock Embeddings (Recommended for Development)**
```bash
# Add to your .env file
USE_MOCK_EMBEDDINGS=true
```

**Option B: Use Different Jina Model**
```bash
# In .env file, try a different model
EMBEDDER_NAME=jina
JINA_MODEL_NAME=jina-embeddings-v3
```

**Option C: Configure Jina API Key Properly**
```bash
# Make sure you have a valid Jina API key
JINA_API_KEY=jina_your-actual-api-key-here
```

### 2. Fallback LLM Provider Error: "Unsupported LLM provider: ollama"

**Error Message:**
```
Failed to create fallback provider: Unsupported LLM provider: ollama. Available: openai, mock
```

**Solution:** The system is trying to use Ollama as fallback but it's not configured. Use mock instead:

```bash
# In .env file
FALLBACK_LLM_PROVIDER=mock
```

Or restart the server after the configuration change.

### 3. Qdrant Version Compatibility Warning

**Warning Message:**
```
Qdrant client version 1.15.0 is incompatible with server version 1.7.0
```

**Solution:** This is just a warning and doesn't break functionality. To fix:

**Option A: Update Qdrant Server**
```bash
# In docker-compose.yml, update Qdrant version
qdrant:
  image: qdrant/qdrant:v1.15.0  # Match client version
```

**Option B: Ignore Version Check**
```bash
# Add to .env file
QDRANT_CHECK_COMPATIBILITY=false
```

### 4. Qdrant Insecure Connection Warning

**Warning Message:**
```
Api key is used with an insecure connection.
```

**Solution:** This happens when using API key with HTTP instead of HTTPS:

**Option A: Remove API Key for Local Development**
```bash
# In .env file, comment out or remove
# QDRANT_API_KEY=your-key
```

**Option B: Use HTTPS URL**
```bash
# For production Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

### 5. Slack Integration Not Working

**Issue:** Slack events are received but responses fail.

**Debugging Steps:**

1. **Check Slack Configuration:**
```bash
# Verify these are set in .env
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your-secret
SLACK_BOT_USER_ID=U1234567890
```

2. **Check Bot Permissions:**
   - `app_mentions:read`
   - `chat:write`
   - `channels:read`
   - `groups:read`
   - `im:read`
   - `mpim:read`

3. **Verify Event Subscriptions:**
   - URL: `https://your-domain.com/slack/events`
   - Events: `app_mention`

4. **Test with Mock Providers:**
```bash
# Temporarily use mock providers to isolate the issue
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
```

### 6. Redis Connection Issues

**Error:** Connection refused to Redis

**Solutions:**

**Option A: Start Redis with Docker**
```bash
docker compose up -d redis
```

**Option B: Disable Caching**
```bash
# In .env file
ENABLE_CACHE=false
```

**Option C: Use Different Redis URL**
```bash
# If Redis is running elsewhere
REDIS_URL=redis://your-redis-host:6379
```

### 7. Performance Issues

**Issue:** Slow response times or timeouts

**Solutions:**

1. **Enable Caching:**
```bash
ENABLE_CACHE=true
```

2. **Use Mock Providers for Testing:**
```bash
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
```

3. **Reduce Top-K Results:**
```bash
DEFAULT_TOP_K=3  # Instead of higher values
```

4. **Disable Expensive Features:**
```bash
ENABLE_SPARSE_RETRIEVAL=false
ENABLE_RERANKER=false
```

### 8. API Key Issues

**Issue:** Invalid API key errors

**Solutions:**

1. **Verify API Keys:**
   - OpenAI: Should start with `sk-`
   - Jina: Should start with `jina_`

2. **Check API Key Permissions:**
   - OpenAI: Ensure sufficient credits and permissions
   - Jina: Verify account is active

3. **Use Environment Variables:**
```bash
# Don't put keys directly in code
export OPENAI_API_KEY="your-key"
export JINA_API_KEY="your-key"
```

## Quick Fixes

### Reset to Working Configuration

If you're having multiple issues, reset to a known working configuration:

```bash
# Run the configuration helper
./configure.sh

# Choose option 1 (Development) for immediate working setup
```

### Test System Health

```bash
# Check all endpoints
curl http://localhost:8000/health
curl http://localhost:8000/info
curl http://localhost:8000/metrics

# Test basic functionality
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, how are you?"}'
```

### Enable Debug Logging

```bash
# In .env file
LOG_LEVEL=DEBUG
```

Then restart the server to see detailed logs.

## Getting Help

### Check Logs

The system provides detailed tracing information. Look for:
- Error messages in the console output
- OpenTelemetry trace information
- HTTP status codes and response details

### Verify Configuration

```bash
# Check your current configuration
cat .env

# Verify environment variables are loaded
meulex api start --help
```

### Test Individual Components

```bash
# Test embeddings
python -c "
from meulex.core.embeddings.factory import create_embedder
from meulex.config.settings import Settings
import asyncio

async def test():
    settings = Settings()
    embedder = create_embedder(settings)
    result = await embedder.embed_async_single('test')
    print('Embeddings working:', len(result) > 0)

asyncio.run(test())
"
```

### Common Environment Variables for Quick Fix

```bash
# Add these to .env for a working development setup
LOG_LEVEL=INFO
USE_MOCK_EMBEDDINGS=true
USE_MOCK_LLM=true
ENABLE_CACHE=false
FALLBACK_LLM_PROVIDER=mock
ENABLE_SPARSE_RETRIEVAL=true
ENABLE_RERANKER=false
```

## Still Having Issues?

1. **Check the logs** for specific error messages
2. **Verify your .env configuration** matches the examples
3. **Test with mock providers** to isolate API issues
4. **Check network connectivity** to external services
5. **Ensure Docker services are running** (`docker compose ps`)

For additional support, check the README.meulex.md file or create an issue with:
- Your .env configuration (without API keys)
- Error messages from logs
- Steps to reproduce the issue
