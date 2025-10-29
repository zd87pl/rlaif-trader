#!/usr/bin/env python3
"""
Deploy RLAIF Trading Pipeline to RunPod Serverless

This script automates the deployment process:
1. Build Docker image
2. Push to registry
3. Create/update RunPod serverless endpoint
4. Test the deployment
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RunPodDeployer:
    """Deploy to RunPod serverless"""

    def __init__(self, api_key: str, config_path: str = "deployment/runpod/runpod-config.yaml"):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        print(f"Loaded configuration: {self.config['name']}")

    def build_and_push_image(
        self,
        registry: str,
        image_name: str,
        tag: str = "latest",
    ) -> str:
        """Build and push Docker image"""
        full_image = f"{registry}/{image_name}:{tag}"

        print(f"\n{'='*60}")
        print(f"Building and pushing image: {full_image}")
        print(f"{'='*60}\n")

        # Build
        print("[1/2] Building Docker image...")
        build_cmd = [
            "./scripts/build_docker.sh",
            "--name",
            image_name,
            "--tag",
            tag,
            "--registry",
            registry,
            "--push",
        ]

        result = subprocess.run(build_cmd, check=True)

        if result.returncode != 0:
            raise RuntimeError("Docker build failed")

        print("✓ Image built and pushed successfully\n")

        return full_image

    def create_endpoint(
        self,
        image: str,
        gpu_type: str = "T4",
        min_workers: int = 0,
        max_workers: int = 5,
    ) -> dict:
        """Create RunPod serverless endpoint"""
        print(f"\n{'='*60}")
        print(f"Creating RunPod Serverless Endpoint")
        print(f"{'='*60}\n")

        # GraphQL mutation to create endpoint
        mutation = """
        mutation CreateEndpoint($input: EndpointInput!) {
          createEndpoint(input: $input) {
            id
            name
            gpuType
            scalerType
            scalerValue
            image
          }
        }
        """

        variables = {
            "input": {
                "name": self.config["name"],
                "image": image,
                "gpuType": gpu_type,
                "scalerType": "QUEUE_DELAY",
                "scalerValue": 4,  # Scale up after 4 seconds in queue
                "minWorkers": min_workers,
                "maxWorkers": max_workers,
                "idleTimeout": self.config["scaling"].get("idle_timeout", 60),
                "flashboot": self.config["network"].get("flashboot", True),
                "env": [
                    {"name": k, "value": v}
                    for k, v in self.config["container"]["env"].items()
                ],
            }
        }

        response = requests.post(
            self.base_url,
            json={"query": mutation, "variables": variables},
            headers=self.headers,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to create endpoint: {response.text}")

        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        endpoint = data["data"]["createEndpoint"]

        print(f"✓ Endpoint created successfully")
        print(f"  ID: {endpoint['id']}")
        print(f"  Name: {endpoint['name']}")
        print(f"  GPU: {endpoint['gpuType']}")
        print(f"  Image: {endpoint['image']}\n")

        return endpoint

    def get_endpoint(self, endpoint_id: str) -> dict:
        """Get endpoint details"""
        query = """
        query GetEndpoint($id: ID!) {
          endpoint(id: $id) {
            id
            name
            gpuType
            image
            scalerType
            scalerValue
            minWorkers
            maxWorkers
            idleTimeout
          }
        }
        """

        response = requests.post(
            self.base_url,
            json={"query": query, "variables": {"id": endpoint_id}},
            headers=self.headers,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to get endpoint: {response.text}")

        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        return data["data"]["endpoint"]

    def test_endpoint(self, endpoint_id: str):
        """Test the deployed endpoint"""
        print(f"\n{'='*60}")
        print(f"Testing Endpoint")
        print(f"{'='*60}\n")

        # Get endpoint URL
        endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

        # Test health check
        print("[1/3] Testing health check...")
        health_payload = {"input": {"action": "health"}}

        response = requests.post(
            endpoint_url,
            json=health_payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Health check passed: {result['output']['status']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            print(f"  Response: {response.text}")

        # Test sentiment analysis
        print("\n[2/3] Testing sentiment analysis...")
        sentiment_payload = {
            "input": {
                "action": "sentiment",
                "texts": [
                    "Strong quarterly earnings beat expectations",
                    "Company faces regulatory challenges",
                ],
                "aggregate": True,
            }
        }

        response = requests.post(
            endpoint_url,
            json=sentiment_payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                agg = result["output"]["aggregated"]
                print(f"✓ Sentiment analysis passed")
                print(f"  Aggregated sentiment: {agg['sentiment']} (score: {agg['score']:.2f})")
            else:
                print(f"✗ Sentiment analysis returned error: {result.get('error')}")
        else:
            print(f"✗ Sentiment analysis failed: {response.status_code}")

        # Test prediction
        print("\n[3/3] Testing prediction...")
        prediction_payload = {
            "input": {
                "action": "predict",
                "symbol": "AAPL",
                "time_series": [150.0, 151.5, 149.8, 152.3, 153.1, 154.0, 152.8],
                "horizon": 5,
                "return_uncertainty": True,
            }
        }

        response = requests.post(
            endpoint_url,
            json=prediction_payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                preds = result["output"]["predictions"]
                print(f"✓ Prediction passed")
                print(f"  Predictions (first 3): {preds[:3]}")
            else:
                print(f"✗ Prediction returned error: {result.get('error')}")
        else:
            print(f"✗ Prediction failed: {response.status_code}")

        print(f"\n{'='*60}")
        print(f"Testing complete!")
        print(f"{'='*60}")
        print(f"\nEndpoint URL: {endpoint_url}")
        print(f"Use this URL to make requests to your serverless function.")


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy RLAIF Trading to RunPod")
    parser.add_argument(
        "--api-key",
        help="RunPod API key (or set RUNPOD_API_KEY env var)",
        default=os.getenv("RUNPOD_API_KEY"),
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Docker registry (e.g., your-dockerhub-username)",
    )
    parser.add_argument(
        "--image-name",
        default="rlaif-trading",
        help="Docker image name",
    )
    parser.add_argument("--tag", default="latest", help="Docker image tag")
    parser.add_argument(
        "--gpu",
        default="T4",
        choices=["T4", "A4000", "A5000", "A6000", "A100"],
        help="GPU type to use",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=0,
        help="Minimum workers (0 for pure serverless)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum workers",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker build (use existing image)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip endpoint testing",
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("Error: RunPod API key required")
        print("Set RUNPOD_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Initialize deployer
    deployer = RunPodDeployer(args.api_key)

    try:
        # Build and push image
        if not args.skip_build:
            full_image = deployer.build_and_push_image(
                registry=args.registry,
                image_name=args.image_name,
                tag=args.tag,
            )
        else:
            full_image = f"{args.registry}/{args.image_name}:{args.tag}"
            print(f"Using existing image: {full_image}")

        # Create endpoint
        endpoint = deployer.create_endpoint(
            image=full_image,
            gpu_type=args.gpu,
            min_workers=args.min_workers,
            max_workers=args.max_workers,
        )

        # Test endpoint
        if not args.skip_test:
            print("\nWaiting 30 seconds for endpoint to initialize...")
            time.sleep(30)
            deployer.test_endpoint(endpoint["id"])

        print("\n" + "=" * 60)
        print("Deployment Successful! 🚀")
        print("=" * 60)
        print(f"\nEndpoint ID: {endpoint['id']}")
        print(f"Endpoint URL: https://api.runpod.ai/v2/{endpoint['id']}/runsync")
        print(f"\nView in RunPod dashboard:")
        print(f"https://www.runpod.io/console/serverless/user/endpoints")

    except Exception as e:
        print(f"\nDeployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
