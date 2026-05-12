# Deployment — Fund Scoring Dashboard

Guia operacional para mantener y actualizar el dashboard de scoring desplegado en AWS EC2.

## Infraestructura

| Recurso | Valor |
|---------|-------|
| EC2 Instance ID | `<your-ec2-instance-id>` |
| Instance Type | `t3.micro` (1 vCPU, 1GB RAM + 2GB swap) |
| Region / AZ | `us-east-2` / `us-east-2c` |
| OS | Ubuntu 24.04 LTS |
| Security Group | `<your-security-group-id>` |
| Puertos abiertos | 22 (SSH), 80 (HTTP), 443 (HTTPS), 8080 (Scoring) |
| IP publica | Cambia con cada stop/start (no es Elastic IP) |
| Dominio | `<your-domain>` (Cloudflare DNS) |
| Tunnel ID | `<your-tunnel-id>` |

## Arquitectura en EC2

```
EC2 t3.micro
  ├── Docker container    ← Scoring dashboard
  │   ├── nginx :80 (mapeado a host:8080)
  │   └── uvicorn :8000 (interno al container)
  └── cloudflared         ← Tunnel a Cloudflare (servicio systemd)
```

## URLs

| URL | Destino |
|-----|---------|
| `https://scoring.<your-domain>` | Dashboard scoring fondos |

## Conectarse a la EC2

La instancia usa key pair configurado en AWS. Se accede via **EC2 Instance Connect**.

### Opcion A: Desde terminal local

Requiere AWS CLI configurado y una key temporal en `~/.ssh/<your-key>`.

```bash
# Obtener IP actual (cambia con cada stop/start)
aws ec2 describe-instances --instance-ids <your-ec2-instance-id> --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text

# Inyectar key temporal (valida 60 segundos) y conectarse
aws ec2-instance-connect send-ssh-public-key \
  --instance-id <your-ec2-instance-id> \
  --instance-os-user ubuntu \
  --ssh-public-key "file://~/.ssh/<your-key>.pub" \
  --availability-zone us-east-2c \
  --region us-east-2 && \
ssh -i "~/.ssh/<your-key>" \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  ubuntu@<IP_ACTUAL>
```

**Importante:** La key inyectada expira en 60 segundos. El `send-ssh-public-key` y el `ssh` deben ejecutarse en cadena (`&&`).

### Opcion B: Desde AWS Console (browser)

1. AWS Console → EC2 → Instances → seleccionar tu instancia
2. Click **Connect** → pestaña **EC2 Instance Connect**
3. Username: `ubuntu`
4. Click **Connect**

## Re-deploy del scoring dashboard

Despues de hacer cambios al codigo y pushear a GitHub (`master` branch):

```bash
# En la EC2
cd ~/fund-scoring
git pull origin master

# Rebuild imagen Docker
sudo docker build -t fund-scoring .

# Reemplazar container
sudo docker stop scoring-dashboard && sudo docker rm scoring-dashboard
sudo docker run -d --name scoring-dashboard \
  -p 8080:80 \
  --restart unless-stopped \
  fund-scoring

# Verificar
curl -s http://localhost:8080/api/health
# → {"status":"ok"}
```

### Re-deploy rapido desde local (un comando)

```bash
# Obtener IP primero
IP=$(aws ec2 describe-instances --instance-ids <your-ec2-instance-id> --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

# Inyectar key + SSH + rebuild completo
aws ec2-instance-connect send-ssh-public-key \
  --instance-id <your-ec2-instance-id> \
  --instance-os-user ubuntu \
  --ssh-public-key "file://~/.ssh/<your-key>.pub" \
  --availability-zone us-east-2c \
  --region us-east-2 && \
ssh -i "~/.ssh/<your-key>" \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=20 \
  ubuntu@$IP \
  "cd ~/fund-scoring && git pull origin master && sudo docker build -t fund-scoring . && sudo docker stop scoring-dashboard && sudo docker rm scoring-dashboard && sudo docker run -d --name scoring-dashboard -p 8080:80 --restart unless-stopped fund-scoring && sleep 3 && curl -s http://localhost:8080/api/health"
```

## Actualizar datos del pipeline

Si regeneras los JSONs del pipeline (paso 05), los archivos afectados son:

```
app/backend/data/meta.json
app/backend/data/funds_summary.json
app/backend/data/fund_detail.json
app/backend/data/drivers.json
app/backend/data/backtest.json
```

Despues de regenerarlos localmente:

```bash
git add app/backend/data/*.json
git commit -m "Update scoring data"
git push origin master
```

Luego hacer re-deploy en EC2 (ver seccion anterior).

## Repo GitHub

| Campo | Valor |
|-------|-------|
| URL | `https://github.com/<your-user>/<your-repo>` |
| Branch de deploy | `master` |
| Ruta en EC2 | `/home/ubuntu/fund-scoring` |

## Comandos utiles en EC2

```bash
# Estado del container scoring
sudo docker ps -a | grep scoring

# Logs del container
sudo docker logs scoring-dashboard
sudo docker logs scoring-dashboard --tail 50 -f   # follow

# Reiniciar container sin rebuild
sudo docker restart scoring-dashboard

# Estado del tunnel Cloudflare
sudo systemctl status cloudflared
sudo journalctl -u cloudflared --no-pager -n 50

# RAM y swap
free -m

# Espacio en disco
df -h
sudo docker system df   # espacio Docker
```

## Cloudflare Tunnel

Config en EC2: `/etc/cloudflared/config.yml`

```yaml
tunnel: <your-tunnel-id>
credentials-file: /etc/cloudflared/<your-tunnel-id>.json

ingress:
  - hostname: scoring.<your-domain>
    service: http://localhost:8080
  - service: http_status:404
```

El tunnel es un servicio systemd que se inicia automaticamente al boot. No depende de la IP publica — conecta de salida a Cloudflare.

```bash
# Reiniciar tunnel
sudo systemctl restart cloudflared

# Re-login si expira el certificado (interactivo — da URL para browser)
cloudflared tunnel login
```

## Cloudflare Dashboard

- SSL/TLS → modo **Full**
- Edge Certificates → Always Use HTTPS: **ON**

## Gestion de la instancia via AWS CLI

```bash
# Ver IP actual
aws ec2 describe-instances --instance-ids <your-ec2-instance-id> --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text

# Ver estado
aws ec2 describe-instances --instance-ids <your-ec2-instance-id> --region us-east-2 \
  --query "Reservations[0].Instances[0].[State.Name,InstanceType,PublicIpAddress]" --output text

# Reboot (sin downtime largo)
aws ec2 reboot-instances --instance-ids <your-ec2-instance-id> --region us-east-2

# Resize temporal a t3.small (para Docker builds pesados)
aws ec2 stop-instances --instance-ids <your-ec2-instance-id> --region us-east-2
# Esperar a stopped...
aws ec2 modify-instance-attribute --instance-id <your-ec2-instance-id> \
  --instance-type "{\"Value\":\"t3.small\"}" --region us-east-2
aws ec2 start-instances --instance-ids <your-ec2-instance-id> --region us-east-2
# Despues del build, volver a t3.micro con el mismo proceso
```

## Troubleshooting

### SSH no conecta (timeout)

- La IP cambia con cada stop/start. Verificar IP actual con `aws ec2 describe-instances`.
- Si el Docker build esta corriendo, consume toda la CPU/RAM del t3.micro y SSH no responde. Esperar o usar AWS Console → EC2 Instance Connect.
- Verificar security group tenga puerto 22 abierto.

### Dashboard scoring no carga

```bash
# Verificar container
sudo docker ps | grep scoring
# Si no aparece, iniciar:
sudo docker start scoring-dashboard
# Si fallo, ver logs:
sudo docker logs scoring-dashboard
```

### Tunnel Cloudflare caido

```bash
sudo systemctl status cloudflared
sudo systemctl restart cloudflared
# Si no arranca, verificar config:
sudo cat /etc/cloudflared/config.yml
sudo cloudflared tunnel run --config /etc/cloudflared/config.yml
```

### Disco lleno

```bash
# Limpiar imagenes Docker viejas
sudo docker image prune -a
sudo docker system prune
```

### Build Docker falla por memoria

Subir temporalmente a t3.small (ver seccion "Gestion de la instancia"). El swap de 2GB esta configurado en `/etc/fstab` y persiste entre reboots.
