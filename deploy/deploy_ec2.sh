#!/bin/bash
# =============================================================
# Deploy script para EC2 — Fund Scoring Dashboard
# Ejecutar en EC2 Instance Connect (usuario ubuntu)
#
# Prerrequisitos:
#   1. Conectarse via AWS Console → EC2 → Connect → EC2 Instance Connect
#   2. Username: ubuntu
# =============================================================
set -euo pipefail

echo "=============================="
echo "PASO 1: Instalar Docker"
echo "=============================="
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-plugin git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Ejecutar docker sin necesidad de reconectar sesión
echo ""
echo "=============================="
echo "PASO 2: Clonar repositorio"
echo "=============================="
cd ~
if [ -d "fund-scoring" ]; then
    echo "Directorio fund-scoring ya existe, haciendo pull..."
    cd fund-scoring
    git pull origin master
else
    git clone https://github.com/Mentalistdg/Bloomberg.git fund-scoring
    cd fund-scoring
fi

echo ""
echo "=============================="
echo "PASO 3: Build Docker image"
echo "=============================="
sudo docker build -t fund-scoring .

echo ""
echo "=============================="
echo "PASO 4: Levantar container"
echo "=============================="
# Detener container anterior si existe
sudo docker stop scoring-dashboard 2>/dev/null || true
sudo docker rm scoring-dashboard 2>/dev/null || true

sudo docker run -d \
    --name scoring-dashboard \
    -p 8080:80 \
    --restart unless-stopped \
    fund-scoring

echo ""
echo "=============================="
echo "PASO 5: Verificar"
echo "=============================="
sleep 5
echo "--- Health check scoring (8080) ---"
curl -s http://localhost:8080/api/health && echo ""
echo "--- Cronnos sigue en :80 ---"
curl -s -o /dev/null -w "HTTP %{http_code}" http://localhost:80 && echo ""

echo ""
echo "=============================="
echo "DEPLOY COMPLETO"
echo "=============================="
echo "Scoring dashboard: http://localhost:8080"
echo "Cronnos (sin tocar): http://localhost:80"
echo ""
echo "Siguiente paso: instalar Cloudflare Tunnel (ver deploy_cloudflare.sh)"
