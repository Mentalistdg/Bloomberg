#!/bin/bash
# =============================================================
# Cloudflare Tunnel setup — scoring.<your-domain>
# Ejecutar DESPUÉS de deploy_ec2.sh
#
# IMPORTANTE: Este script es interactivo en un paso —
# cloudflared login abrirá una URL que debes pegar en tu browser
# para autorizar la cuenta de Cloudflare.
#
# Antes de ejecutar, reemplaza <your-domain> por tu dominio real.
# =============================================================
set -euo pipefail

DOMAIN="${DOMAIN:-<your-domain>}"

echo "=============================="
echo "PASO 1: Instalar cloudflared"
echo "=============================="
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
rm /tmp/cloudflared.deb

echo ""
echo "=============================="
echo "PASO 2: Login en Cloudflare"
echo "=============================="
echo ">>> Se abrirá una URL. Cópiala y pégala en tu browser para autorizar."
echo ">>> Selecciona la zona '${DOMAIN}'."
echo ""
cloudflared tunnel login

echo ""
echo "=============================="
echo "PASO 3: Crear tunnel"
echo "=============================="
cloudflared tunnel create scoring

# Obtener el TUNNEL_ID del output
TUNNEL_ID=$(cloudflared tunnel list | grep scoring | awk '{print $1}')
echo "Tunnel ID: $TUNNEL_ID"

echo ""
echo "=============================="
echo "PASO 4: Configurar ingress"
echo "=============================="
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << HEREDOC
tunnel: ${TUNNEL_ID}
credentials-file: /home/ubuntu/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: scoring.${DOMAIN}
    service: http://localhost:8080
  - service: http_status:404
HEREDOC

echo "Config escrito en ~/.cloudflared/config.yml"
cat ~/.cloudflared/config.yml

echo ""
echo "=============================="
echo "PASO 5: Registrar DNS en Cloudflare"
echo "=============================="
cloudflared tunnel route dns scoring scoring.${DOMAIN}

echo ""
echo "=============================="
echo "PASO 6: Instalar como servicio"
echo "=============================="
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

echo ""
echo "=============================="
echo "TUNNEL CONFIGURADO"
echo "=============================="
echo ""
echo "URL final (una vez que SSL esté en modo Full en Cloudflare dashboard):"
echo "  https://scoring.${DOMAIN}   → Dashboard scoring fondos"
echo ""
echo "PASO MANUAL RESTANTE:"
echo "  Cloudflare Dashboard → SSL/TLS → modo 'Full'"
echo "  Edge Certificates → Always Use HTTPS: ON"
