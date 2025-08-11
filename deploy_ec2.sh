#!/bin/bash

# Deploy MLOps Housing Model to EC2
# Usage: ./deploy_ec2.sh <EC2_HOST> <PEM_KEY_PATH>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <EC2_HOST> <PEM_KEY_PATH>"
    echo "Example: $0 ec2-user@your-ec2-instance.amazonaws.com ~/.ssh/your-key.pem"
    exit 1
fi

EC2_HOST=$1
PEM_KEY=$2

echo "🚀 Deploying MLOps Housing Model to EC2..."
echo "🎯 Target: $EC2_HOST"

# Create deployment script to run on EC2
cat > ec2_deploy_commands.sh << 'EOF'
#!/bin/bash

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "✅ Docker installed successfully!"
fi

# Pull the latest image
echo "📥 Pulling latest image..."
sudo docker pull abhansi/mlops-assignment-housing-model:latest

# Stop and remove existing container
echo "🛑 Stopping existing container (if any)..."
sudo docker stop mlops-housing-container 2>/dev/null || true
sudo docker rm mlops-housing-container 2>/dev/null || true

# Run the new container
echo "▶️ Starting new container..."
sudo docker run -d \
  --name mlops-housing-container \
  -p 80:5000 \
  --restart unless-stopped \
  abhansi/mlops-assignment-housing-model:latest

# Wait for container to start
sleep 10

# Check container status
if sudo docker ps | grep -q mlops-housing-container; then
    echo "✅ Deployment successful!"
    echo "🌐 API is available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
    sudo docker ps | grep mlops-housing-container
else
    echo "❌ Deployment failed. Check logs:"
    sudo docker logs mlops-housing-container
    exit 1
fi
EOF

# Copy and execute the deployment script on EC2
echo "📤 Uploading deployment script to EC2..."
scp -i "$PEM_KEY" -o StrictHostKeyChecking=no ec2_deploy_commands.sh "$EC2_HOST:~/"

echo "🔧 Executing deployment on EC2..."
ssh -i "$PEM_KEY" -o StrictHostKeyChecking=no "$EC2_HOST" 'chmod +x ec2_deploy_commands.sh && ./ec2_deploy_commands.sh'

# Clean up local temp file
rm ec2_deploy_commands.sh

echo "🎉 EC2 deployment completed!"
echo "💡 Don't forget to:"
echo "   - Open port 80 in your EC2 Security Group"
echo "   - Test the API at your EC2 public IP"