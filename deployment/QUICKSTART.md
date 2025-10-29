# RunPod Deployment - Quick Reference

**TL;DR**: Deploy RLAIF Trading to RunPod serverless in 3 commands.

---

## Prerequisites

```bash
# Set environment variables
export RUNPOD_API_KEY="your-api-key"
export DOCKERHUB_USERNAME="your-username"
```

---

## Deploy (Automatic)

```bash
# One-command deployment
python scripts/deploy_runpod.py \
  --registry $DOCKERHUB_USERNAME \
  --gpu T4 \
  --min-workers 0 \
  --max-workers 5
```

This will:
1. Build Docker image (<3GB)
2. Push to Docker Hub
3. Create RunPod endpoint
4. Test deployment

**Time**: ~10-15 minutes total

---

## Deploy (Manual)

```bash
# 1. Build and push Docker image
./scripts/build_docker.sh \
  --registry $DOCKERHUB_USERNAME \
  --push

# 2. Create endpoint (via RunPod dashboard)
# Go to: https://www.runpod.io/console/serverless
# Click "New Endpoint"
# Use image: your-username/rlaif-trading:latest

# 3. Test
curl -X POST \
  https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'
```

---

## Usage Examples

### Health Check

```bash
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "health"
    }
  }'
```

### Sentiment Analysis

```bash
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "sentiment",
      "texts": [
        "Strong earnings beat expectations",
        "Facing regulatory challenges"
      ],
      "aggregate": true
    }
  }'
```

### Stock Prediction

```bash
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "predict",
      "symbol": "AAPL",
      "time_series": [150.0, 151.5, 149.8, 152.3, 153.1],
      "horizon": 10,
      "return_uncertainty": true
    }
  }'
```

### Technical Indicators

```bash
curl -X POST \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

## Python Client

```python
import requests

class RLAIFClient:
    def __init__(self, endpoint_id: str, api_key: str):
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def predict(self, symbol: str, time_series: list, horizon: int = 30):
        response = requests.post(
            self.url,
            json={
                "input": {
                    "action": "predict",
                    "symbol": symbol,
                    "time_series": time_series,
                    "horizon": horizon,
                    "return_uncertainty": True
                }
            },
            headers=self.headers
        )
        return response.json()

    def sentiment(self, texts: list):
        response = requests.post(
            self.url,
            json={
                "input": {
                    "action": "sentiment",
                    "texts": texts,
                    "aggregate": True
                }
            },
            headers=self.headers
        )
        return response.json()

    def indicators(self, ohlcv: dict):
        response = requests.post(
            self.url,
            json={
                "input": {
                    "action": "indicators",
                    "ohlcv": ohlcv
                }
            },
            headers=self.headers
        )
        return response.json()

# Usage
client = RLAIFClient(
    endpoint_id="your-endpoint-id",
    api_key="your-api-key"
)

# Predict
result = client.predict("AAPL", [150.0, 151.5, 149.8], horizon=10)
print(result)

# Sentiment
result = client.sentiment(["Strong earnings growth"])
print(result)
```

---

## Cost Estimate

**T4 GPU** ($0.40/hour):
- 10,000 requests/month @ 2s each = 5.56 hours
- **Cost: ~$2.22/month**

**A100 GPU** ($2.09/hour):
- 10,000 requests/month @ 1s each = 2.78 hours
- **Cost: ~$5.81/month**

**Recommendation**: Start with T4, upgrade to A100 if needed.

---

## Monitoring

**Dashboard**: [https://www.runpod.io/console/serverless/user/endpoints](https://www.runpod.io/console/serverless/user/endpoints)

View:
- Request count
- Average latency
- Error rate
- Cost per request
- GPU utilization

---

## Troubleshooting

### Cold start slow?
- Enable FlashBoot ✅ (already enabled)
- Reduce image size (target <3GB)
- Use warm pool (min_workers=1)

### Out of memory?
- Upgrade to A100 (40GB VRAM)
- Reduce batch size in config

### Requests failing?
- Check logs in RunPod dashboard
- Verify API key is correct
- Test locally first

---

## Support

- **Full Guide**: See `DEPLOYMENT.md`
- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io)
- **Issues**: GitHub Issues

---

## Summary

- ✅ Deploy in <15 minutes
- ✅ Auto-scales 0→5+ workers
- ✅ <2s cold starts
- ✅ $2-10/month typical cost
- ✅ Production-ready

**Next**: Monitor usage, optimize, scale as needed.
