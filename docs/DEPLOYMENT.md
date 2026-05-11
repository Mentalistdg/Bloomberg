# Deployment — Fund Scoring Dashboard

Guia operacional para mantener y actualizar el dashboard de scoring desplegado en AWS EC2.

## Infraestructura

| Recurso | Valor |
|---------|-------|
| EC2 Instance ID | `i-0bae26943c56844d7` |
| Instance Type | `t3.micro` (1 vCPU, 1GB RAM + 2GB swap) |
| Region / AZ | `us-east-2` / `us-east-2c` |
| OS | Ubuntu 24.04 LTS |
| Security Group | `sg-07ed04df0a4a31fd3` (launch-wizard-1) |
| Puertos abiertos | 22 (SSH), 80 (Cronnos), 443 (HTTPS), 8000 (Cronnos API), 8080 (Scoring) |
| IP publica | Cambia con cada stop/start (no es Elastic IP) |
| Dominio | `davidgonzalez.cl` (Cloudflare DNS) |
| Tunnel ID | `eac5852d-a55b-43e0-ba0c-8caa1f1afec8` |

## Arquitectura en EC2

```
EC2 t3.micro
  ├── Nginx :80           ← Cronnos (NO TOCAR)
  ├── Uvicorn :8000       ← Cronnos backend (NO TOCAR)
  ├── Docker container    ← Scoring dashboard
  │   ├── nginx :80 (mapeado a host:8080)
  │   └── uvicorn :8000 (interno al container)
  └── cloudflared         ← Tunnel a Cloudflare (servicio systemd)
```

## URLs

| URL | Destino |
|-----|---------|
| https://scoring.davidgonzalez.cl | Dashboard scoring fondos |
| https://cronnos.davidgonzalez.cl | Dashboard tesis Cronnos |
| https://davidgonzalez.cl | Cronnos / landing |

## Conectarse a la EC2

La instancia usa key pair `tesis-key` (no disponible localmente). Se accede via **EC2 Instance Connect**.

### Opcion A: Desde terminal local (Windows)

Requiere AWS CLI configurado y la key temporal en `~/.ssh/ec2_temp`.

```bash
# Obtener IP actual (cambia con cada stop/start)
aws ec2 describe-instances --instance-ids i-0bae26943c56844d7 --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text

# Inyectar key temporal (valida 60 segundos) y conectarse
aws ec2-instance-connect send-ssh-public-key \
  --instance-id i-0bae26943c56844d7 \
  --instance-os-user ubuntu \
  --ssh-public-key "file://C:/Users/dgonz/.ssh/ec2_temp.pub" \
  --availability-zone us-east-2c \
  --region us-east-2 && \
"C:/Windows/System32/OpenSSH/ssh.exe" \
  -i "C:/Users/dgonz/.ssh/ec2_temp" \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  ubuntu@<IP_ACTUAL>
```

**Importante:** La key inyectada expira en 60 segundos. El `send-ssh-public-key` y el `ssh` deben ejecutarse en cadena (`&&`).

### Opcion B: Desde AWS Console (browser)

1. AWS Console → EC2 → Instances → seleccionar `i-0bae26943c56844d7`
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

### Re-deploy rapido desde Windows (un comando)

```bash
# Obtener IP primero
IP=$(aws ec2 describe-instances --instance-ids i-0bae26943c56844d7 --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

# Inyectar key + SSH + rebuild completo
aws ec2-instance-connect send-ssh-public-key \
  --instance-id i-0bae26943c56844d7 \
  --instance-os-user ubuntu \
  --ssh-public-key "file://C:/Users/dgonz/.ssh/ec2_temp.pub" \
  --availability-zone us-east-2c \
  --region us-east-2 && \
"C:/Windows/System32/OpenSSH/ssh.exe" \
  -i "C:/Users/dgonz/.ssh/ec2_temp" \
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
# En Windows
git add app/backend/data/*.json
git commit -m "Update scoring data"
git push origin master
```

Luego hacer re-deploy en EC2 (ver seccion anterior).

## Repo GitHub

| Campo | Valor |
|-------|-------|
| URL | https://github.com/Mentalistdg/Bloomberg |
| Branch de deploy | `master` |
| Visibilidad | **Privado** (hacerlo publico temporalmente si se necesita clonar en otra maquina) |
| Ruta en EC2 | `/home/ubuntu/fund-scoring` |

**IMPORTANTE:** Si necesitas clonar en una maquina nueva, hacer publico temporalmente y volver a privado despues del clone.

## Comandos utiles en EC2

```bash
# Estado del container scoring
sudo docker ps -a | grep scoring

# Logs del container
sudo docker logs scoring-dashboard
sudo docker logs scoring-dashboard --tail 50 -f   # follow

# Reiniciar container sin rebuild
sudo docker restart scoring-dashboard

# Estado de Cronnos
systemctl status nginx
ps aux | grep uvicorn

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
tunnel: eac5852d-a55b-43e0-ba0c-8caa1f1afec8
credentials-file: /etc/cloudflared/eac5852d-a55b-43e0-ba0c-8caa1f1afec8.json

ingress:
  - hostname: scoring.davidgonzalez.cl
    service: http://localhost:8080
  - hostname: cronnos.davidgonzalez.cl
    service: http://localhost:80
  - hostname: davidgonzalez.cl
    service: http://localhost:80
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
aws ec2 describe-instances --instance-ids i-0bae26943c56844d7 --region us-east-2 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text

# Ver estado
aws ec2 describe-instances --instance-ids i-0bae26943c56844d7 --region us-east-2 \
  --query "Reservations[0].Instances[0].[State.Name,InstanceType,PublicIpAddress]" --output text

# Reboot (sin downtime largo)
aws ec2 reboot-instances --instance-ids i-0bae26943c56844d7 --region us-east-2

# Resize temporal a t3.small (para Docker builds pesados)
aws ec2 stop-instances --instance-ids i-0bae26943c56844d7 --region us-east-2
# Esperar a stopped...
aws ec2 modify-instance-attribute --instance-id i-0bae26943c56844d7 \
  --instance-type "{\"Value\":\"t3.small\"}" --region us-east-2
aws ec2 start-instances --instance-ids i-0bae26943c56844d7 --region us-east-2
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

### Cronnos no carga

```bash
sudo systemctl status nginx
sudo systemctl restart nginx
# Verificar uvicorn de Cronnos
ps aux | grep uvicorn
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
