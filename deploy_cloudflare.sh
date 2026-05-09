#!/bin/bash
# =============================================================
# Cloudflare Tunnel setup — scoring.davidgonzalez.cl + cronnos.davidgonzalez.cl
# Ejecutar DESPUÉS de deploy_ec2.sh
#
# IMPORTANTE: Este script es interactivo en un paso —
# cloudflared login abrirá una URL que debes pegar en tu browser
# para autorizar la cuenta de Cloudflare.
# =============================================================
set -euo pipefail

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
echo ">>> Selecciona la zona 'davidgonzalez.cl'."
echo ""
cloudflared tunnel login

echo ""
echo "=============================="
echo "PASO 3: Crear tunnel"
echo "=============================="
cloudflared tunnel create davidgonzalez

# Obtener el TUNNEL_ID del output
TUNNEL_ID=$(cloudflared tunnel list | grep davidgonzalez | awk '{print $1}')
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
  - hostname: scoring.davidgonzalez.cl
    service: http://localhost:8080
  - hostname: cronnos.davidgonzalez.cl
    service: http://localhost:80
  - hostname: davidgonzalez.cl
    service: http://localhost:80
  - service: http_status:404
HEREDOC

echo "Config escrito en ~/.cloudflared/config.yml"
cat ~/.cloudflared/config.yml

echo ""
echo "=============================="
echo "PASO 5: Registrar DNS en Cloudflare"
echo "=============================="
cloudflared tunnel route dns davidgonzalez scoring.davidgonzalez.cl
cloudflared tunnel route dns davidgonzalez cronnos.davidgonzalez.cl
cloudflared tunnel route dns davidgonzalez davidgonzalez.cl

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
echo "URLs finales (una vez que SSL esté en modo Full en Cloudflare dashboard):"
echo "  https://scoring.davidgonzalez.cl   → Dashboard scoring fondos"
echo "  https://cronnos.davidgonzalez.cl   → Dashboard tesis Cronnos"
echo "  https://davidgonzalez.cl           → Landing (Cronnos por ahora)"
echo ""
echo "PASO MANUAL RESTANTE:"
echo "  Cloudflare Dashboard → SSL/TLS → modo 'Full'"
echo "  Edge Certificates → Always Use HTTPS: ON"
