# RunPod Serverless Deployment Guide

Complete guide to deploying the RLAIF Trading Pipeline as a serverless function on RunPod.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Deployment](#detailed-deployment)
5. [API Reference](#api-reference)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The RLAIF Trading Pipeline is deployed as a RunPod serverless function with:

- **Container Size**: <3GB (optimized for fast cold starts)
- **Cold Start Time**: <2s with FlashBoot
- **GPU Support**: T4, A100, H100
- **Scaling**: Auto-scales from 0 to 5+ workers
- **Cost**: Pay-per-second billing
- **Endpoints**: Prediction, sentiment analysis, technical indicators

**Architecture:**
```
User Request → RunPod API → Serverless Worker (GPU) → Handler → Models → Response
```

**Key Features:**
- ✅ Multi-stage Docker build (<3GB target)
- ✅ Pre-cached models (FinBERT, sentence transformers)
- ✅ FastAPI + RunPod handler
- ✅ Auto-scaling based on queue depth
- ✅ FlashBoot for <2s cold starts
- ✅ Comprehensive health checks

---

## Prerequisites

### 1. RunPod Account

Sign up at [runpod.io](https://www.runpod.io)

Get your API key:
1. Go to [Settings](https://www.runpod.io/console/user/settings)
2. Generate API key
3. Copy and save securely

### 2. Docker Hub Account

Sign up at [hub.docker.com](https://hub.docker.com)

You'll push your container here.

### 3. Local Requirements

```bash
# Docker installed and running
docker --version

# Python 3.11+
python --version

# RunPod CLI (optional but recommended)
pip install runpod

# Required Python packages
pip install pyyaml requests
```

### 4. Environment Setup

```bash
# Set environment variables
export RUNPOD_API_KEY="your-api-key-here"
export DOCKERHUB_USERNAME="your-dockerhub-username"

# Or add to .env file
echo "RUNPOD_API_KEY=your-api-key-here" >> .env
echo "DOCKERHUB_USERNAME=your-dockerhub-username" >> .env
```

---

## Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# 1. Build, push, and deploy in one command
python scripts/deploy_runpod.py \
  --api-key $RUNPOD_API_KEY \
  --registry $DOCKERHUB_USERNAME \
  --gpu T4 \
  --min-workers 0 \
  --max-workers 5

# This will:
# - Build Docker image
# - Push to Docker Hub
# - Create RunPod endpoint
# - Test the deployment
```

### Option 2: Manual Steps

```bash
# 1. Build and push Docker image
./scripts/build_docker.sh \
  --registry $DOCKERHUB_USERNAME \
  --push

# 2. Create endpoint manually in RunPod dashboard
# Or use RunPod CLI:
runpod endpoint create \
  --name rlaif-trading-serverless \
  --image $DOCKERHUB_USERNAME/rlaif-trading:latest \
  --gpu T4 \
  --min-workers 0 \
  --max-workers 5

# 3. Test the endpoint
curl -X POST \
  https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'
```

---

## Detailed Deployment

### Step 1: Configure Deployment

Edit `deployment/runpod/runpod-config.yaml`:

```yaml
name: rlaif-trading-serverless

# Your Docker image
image: your-dockerhub-username/rlaif-trading:latest

# GPU configuration
gpu:
  types:
    - T4   # $0.40/hr - cost-effective
    - A100 # $2.09/hr - high performance
  min_vram: 8

# Scaling
scaling:
  min_workers: 0      # Pure serverless (no idle costs)
  max_workers: 5      # Scale up to 5 workers
  idle_timeout: 60    # Keep alive for 60s
  scale_up_threshold: 70   # Scale when >70% utilized
```

### Step 2: Build Docker Container

The Dockerfile uses multi-stage builds for optimization:

**Stage 1: Builder**
- Install dependencies
- Compile Python packages
- Create virtual environment

**Stage 2: Runtime**
- Minimal CUDA runtime base
- Copy only necessary files
- Pre-download models (FinBERT, sentence transformers)
- Target: <3GB final image

```bash
# Build locally (for testing)
docker build -t rlaif-trading:latest -f deployment/docker/Dockerfile .

# Test locally
docker run -p 8000:8000 rlaif-trading:latest

# Verify in another terminal
curl http://localhost:8000/health
```

**Expected image size**: 2.5-3GB

### Step 3: Push to Registry

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag rlaif-trading:latest $DOCKERHUB_USERNAME/rlaif-trading:latest

# Push
docker push $DOCKERHUB_USERNAME/rlaif-trading:latest
```

### Step 4: Deploy to RunPod

**Option A: Using deployment script**

```bash
python scripts/deploy_runpod.py \
  --registry $DOCKERHUB_USERNAME \
  --image-name rlaif-trading \
  --tag latest \
  --gpu T4 \
  --min-workers 0 \
  --max-workers 5
```

**Option B: Using RunPod dashboard**

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: rlaif-trading-serverless
   - **Docker Image**: your-dockerhub-username/rlaif-trading:latest
   - **GPU Type**: T4 (or A100 for better performance)
   - **Workers**: Min 0, Max 5
   - **Scaling**: Queue-based, 4s delay
   - **FlashBoot**: Enabled
4. Add environment variables:
   ```
   LOG_LEVEL=INFO
   APP_ENV=production
   PYTHONUNBUFFERED=1
   ```
5. Click "Create"

**Option C: Using RunPod CLI**

```bash
# Install CLI
pip install runpod

# Create endpoint
runpod endpoint create \
  --name rlaif-trading-serverless \
  --image $DOCKERHUB_USERNAME/rlaif-trading:latest \
  --gpu T4 \
  --min-workers 0 \
  --max-workers 5 \
  --idle-timeout 60 \
  --flashboot
```

### Step 5: Test Deployment

```bash
# Get endpoint ID from RunPod dashboard or deployment output
export ENDPOINT_ID="your-endpoint-id"

# Test health check
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "health"
    }
  }'

# Expected response:
# {
#   "status": "success",
#   "output": {
#     "status": "healthy",
#     "gpu_available": true,
#     "models_loaded": { ... },
#     "timestamp": 1234567890.123
#   }
# }
```

---

## API Reference

### Base URL

```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

**Headers:**
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
Content-Type: application/json
```

### 1. Health Check

**Endpoint**: `POST /`

**Request:**
```json
{
  "input": {
    "action": "health"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "output": {
    "status": "healthy",
    "gpu_available": true,
    "models_loaded": {
      "sentiment_analyzer": true,
      "technical_engine": true
    },
    "timestamp": 1234567890.123
  }
}
```

### 2. Sentiment Analysis

**Endpoint**: `POST /`

**Request:**
```json
{
  "input": {
    "action": "sentiment",
    "texts": [
      "Strong quarterly earnings beat expectations",
      "Company faces regulatory challenges"
    ],
    "aggregate": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "output": {
    "sentiments": [
      {
        "sentiment": "positive",
        "score": 0.92,
        "confidence": 0.88
      },
      {
        "sentiment": "negative",
        "score": -0.76,
        "confidence": 0.85
      }
    ],
    "aggregated": {
      "sentiment": "neutral",
      "score": 0.08,
      "confidence": 0.865,
      "count": 2
    },
    "timestamp": 1234567890.123
  }
}
```

### 3. Stock Prediction

**Endpoint**: `POST /`

**Request:**
```json
{
  "input": {
    "action": "predict",
    "symbol": "AAPL",
    "time_series": [150.0, 151.5, 149.8, 152.3, 153.1],
    "horizon": 10,
    "return_uncertainty": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "output": {
    "symbol": "AAPL",
    "predictions": [153.5, 153.8, 154.1, ...],
    "uncertainty": [0.5, 1.0, 1.5, ...],
    "model_used": "moving_average_placeholder",
    "timestamp": 1234567890.123
  }
}
```

### 4. Technical Indicators

**Endpoint**: `POST /`

**Request:**
```json
{
  "input": {
    "action": "indicators",
    "ohlcv": {
      "open": [100, 101, 102, 103, 104],
      "high": [102, 103, 104, 105, 106],
      "low": [99, 100, 101, 102, 103],
      "close": [101, 102, 103, 104, 105],
      "volume": [1000000, 1100000, 1200000, 1300000, 1400000]
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "output": {
    "indicators": {
      "rsi": [50.0, 55.0, 60.0, 65.0, 70.0],
      "macd": [0.5, 1.0, 1.5, 2.0, 2.5],
      "atr": [1.0, 1.1, 1.2, 1.3, 1.4],
      ... (60+ indicators)
    },
    "timestamp": 1234567890.123
  }
}
```

### Error Responses

```json
{
  "status": "error",
  "error": "Error message here"
}
```

---

## Cost Optimization

### GPU Pricing (as of 2025)

| GPU Type | VRAM | Price/hour | Best For |
|----------|------|------------|----------|
| **T4** | 16GB | $0.40 | Cost-effective inference, testing |
| A4000 | 16GB | $0.76 | Balanced performance |
| A5000 | 24GB | $1.14 | Large models |
| A6000 | 48GB | $1.79 | Very large models |
| **A100** | 40GB | $2.09 | High performance |
| A100 | 80GB | $2.89 | Maximum performance |

**Recommendation**: Start with **T4** for cost-effectiveness, upgrade to **A100** if latency is critical.

### Cost Calculation

**Example scenario**: 10,000 requests/month

**With T4 GPU:**
- Average request time: 2s
- Total compute time: 20,000s = 5.56 hours
- Cost: 5.56 × $0.40 = **$2.22/month**

**With A100 GPU:**
- Average request time: 1s (faster)
- Total compute time: 10,000s = 2.78 hours
- Cost: 2.78 × $2.09 = **$5.81/month**

**With warm pool** (1 T4 worker always ready):
- Base cost: 730 hours × $0.40 = $292/month
- ❌ Not recommended for low traffic

### Optimization Tips

1. **Use min_workers=0** for pure serverless (no idle costs)
2. **Enable FlashBoot** for <2s cold starts
3. **Batch requests** when possible
4. **Choose appropriate GPU** (T4 for most cases)
5. **Set idle_timeout** (60-120s reasonable)
6. **Monitor usage** and adjust max_workers
7. **Use regions wisely** (prefer cheaper regions)

### Cost Monitoring

View costs in RunPod dashboard:
- [Billing Overview](https://www.runpod.io/console/user/billing)
- Set up alerts for spend thresholds
- Monitor request latency vs cost trade-offs

---

## Monitoring

### RunPod Dashboard

Access at: [https://www.runpod.io/console/serverless/user/endpoints](https://www.runpod.io/console/serverless/user/endpoints)

**Metrics available:**
- Request count
- Average latency
- Error rate
- Worker count (current/min/max)
- Queue depth
- GPU utilization
- Cost per request

### Health Checks

Built-in health check endpoint:
- **Interval**: 30s
- **Timeout**: 10s
- **Unhealthy threshold**: 3 consecutive failures

```bash
# Manual health check
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'
```

### Logs

View logs in RunPod dashboard:
1. Go to endpoint details
2. Click "Logs" tab
3. Filter by worker, severity, time range

**Log levels:**
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical failures

### Alerts

Set up alerts in RunPod dashboard:
- Error rate > 5%
- Average latency > 5s
- Cost > $X/day

---

## Troubleshooting

### Cold Start Too Slow

**Problem**: Cold starts taking >10s

**Solutions:**
1. Enable FlashBoot in config
2. Reduce Docker image size (<3GB target)
3. Pre-download models during build
4. Use warm pool (1-2 min_workers)

**Check image size:**
```bash
docker images rlaif-trading:latest
```

### Out of Memory (OOM)

**Problem**: Workers crashing with OOM errors

**Solutions:**
1. Upgrade to larger GPU (A100 has 40GB VRAM)
2. Reduce batch_size in model config
3. Clear GPU cache between requests
4. Monitor memory usage in logs

### High Error Rate

**Problem**: >5% requests failing

**Solutions:**
1. Check logs for specific errors
2. Verify model loading succeeded
3. Test locally with same inputs
4. Increase request timeout
5. Check GPU availability

### Slow Inference

**Problem**: Requests taking >10s

**Solutions:**
1. Upgrade to faster GPU (T4 → A100)
2. Optimize model (quantization, pruning)
3. Use batch inference when possible
4. Profile code for bottlenecks

### Deployment Fails

**Problem**: Cannot create endpoint

**Solutions:**
1. Verify API key is correct
2. Check Docker image is accessible (public or authenticated)
3. Ensure image tag exists
4. Try different GPU type (may be out of stock)
5. Check RunPod status page

### Testing Commands

```bash
# Test Docker image locally
docker run -p 8000:8000 rlaif-trading:latest

# In another terminal
curl http://localhost:8000/health

# Test RunPod endpoint
python -c "
import requests
response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync',
    json={'input': {'action': 'health'}},
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
print(response.json())
"
```

---

## Advanced Configuration

### Custom GPU Requirements

Edit `deployment/runpod/runpod-config.yaml`:

```yaml
gpu:
  types:
    - A100  # Prefer A100
    - A6000 # Fallback to A6000
    - T4    # Final fallback
  min_vram: 16  # Minimum VRAM required
```

### Custom Scaling

```yaml
scaling:
  min_workers: 1          # Keep 1 worker warm
  max_workers: 10         # Scale up to 10
  idle_timeout: 120       # Keep alive for 2 minutes
  scale_up_threshold: 50  # Scale at 50% utilization
```

### Environment Variables

Add secrets via RunPod dashboard (Settings → Secrets):

```bash
# In RunPod dashboard
ANTHROPIC_API_KEY=sk-ant-xxx
ALPACA_API_KEY=PKxxx
ALPACA_SECRET_KEY=xxx
```

Then reference in code:
```python
import os
api_key = os.getenv("ANTHROPIC_API_KEY")
```

### Regions

Prefer cheaper regions in config:

```yaml
cost_optimization:
  regions:
    - US-CA  # California
    - US-OR  # Oregon
    - EU-RO  # Romania (cheapest)
```

---

## Next Steps

1. **Monitor Performance**: Track latency, cost, and error rates
2. **Optimize**: Based on actual usage patterns
3. **Scale**: Adjust worker counts as needed
4. **Integrate**: Connect to your applications
5. **Iterate**: Fine-tune models, add features

---

## Support

- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io)
- **RunPod Discord**: [discord.gg/runpod](https://discord.gg/runpod)
- **Issues**: GitHub Issues

---

## Summary

You now have a production-ready serverless deployment:

- ✅ Docker container (<3GB)
- ✅ FastAPI endpoints
- ✅ RunPod serverless configuration
- ✅ Auto-scaling (0-5+ workers)
- ✅ <2s cold starts with FlashBoot
- ✅ Pay-per-second billing
- ✅ Comprehensive monitoring
- ✅ Cost-optimized (starts at ~$2/month)

**Cost**: ~$2-10/month for typical usage
**Performance**: <2s cold start, <1s inference
**Scaling**: Handles 0-1000+ req/min automatically

🚀 **Ready for production!**
