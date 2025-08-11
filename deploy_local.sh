#!/bin/bash

# Deploy MLOps Housing Model Locally
echo "🚀 Deploying MLOps Housing Model locally..."

# Pull the latest image from Docker Hub
echo "📥 Pulling latest image from Docker Hub..."
docker pull abhansi/mlops-assignment-housing-model:latest

# Stop and remove any existing container
echo "🛑 Stopping existing container (if any)..."
docker stop mlops-housing-container 2>/dev/null || true
docker rm mlops-housing-container 2>/dev/null || true

# Run the container
echo "▶️ Starting new container..."
docker run -d \
  --name mlops-housing-container \
  -p 5000:5000 \
  --restart unless-stopped \
  udaykiran1997/mlops-model:latest

# Wait a moment for container to start
sleep 5

# Check if container is running
if docker ps | grep -q mlops-housing-container; then
    echo "✅ Container is running successfully!"
    echo "🌐 API is available at: http://localhost:5000"
    echo ""
    echo "📊 Container status:"
    docker ps | grep mlops-housing-container
    echo ""
    echo "🧪 Test your API with:"
    echo "curl -X POST http://localhost:5000/predict \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"features\": [8.3252, 41.0, 6.984, 1.02, 322.0, 2.555, 37.88, -122.23]}'"
else
    echo "❌ Container failed to start. Check logs:"
    docker logs mlops-housing-container
    exit 1
fi