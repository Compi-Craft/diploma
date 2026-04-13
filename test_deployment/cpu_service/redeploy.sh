#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔧 Rebuilding cpu-service inside minikube..."
eval $(minikube docker-env)
docker build -t cpu-service:latest "$SCRIPT_DIR"

echo "📦 Applying manifests..."
kubectl apply -f "$SCRIPT_DIR/../cpu-service.yaml"

echo "🔄 Restarting deployment..."
kubectl rollout restart deployment cpu-service
kubectl rollout status deployment cpu-service

echo "✅ Done!"
