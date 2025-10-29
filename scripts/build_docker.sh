#!/bin/bash
# Build Docker image for RunPod deployment

set -e  # Exit on error

echo "============================================"
echo "Building RLAIF Trading Docker Image"
echo "============================================"

# Configuration
IMAGE_NAME="${IMAGE_NAME:-rlaif-trading}"
TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # Set to your Docker Hub username or registry

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Full image name
if [ -n "$REGISTRY" ]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
  FULL_IMAGE="${IMAGE_NAME}:${TAG}"
fi

echo "Image: $FULL_IMAGE"
echo ""

# Build image
echo "[1/3] Building Docker image..."
docker build \
  -t "$FULL_IMAGE" \
  -f deployment/docker/Dockerfile \
  .

echo "✓ Build complete"
echo ""

# Get image size
echo "[2/3] Image information:"
docker images "$FULL_IMAGE" --format "Size: {{.Size}}"
echo ""

# Push if requested
if [ "$PUSH" = true ]; then
  echo "[3/3] Pushing to registry..."

  if [ -z "$REGISTRY" ]; then
    echo "Error: REGISTRY not set. Cannot push."
    exit 1
  fi

  docker push "$FULL_IMAGE"
  echo "✓ Push complete"
else
  echo "[3/3] Skipping push (use --push to push to registry)"
fi

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo "Image: $FULL_IMAGE"
echo ""
echo "To run locally:"
echo "  docker run -p 8000:8000 $FULL_IMAGE"
echo ""
echo "To push to registry:"
echo "  $0 --push --registry <your-dockerhub-username>"
echo ""
echo "To test:"
echo "  curl http://localhost:8000/health"
echo "============================================"
