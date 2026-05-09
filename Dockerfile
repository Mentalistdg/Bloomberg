# --- Stage 1: Build frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY app/frontend/package.json app/frontend/package-lock.json* ./
RUN npm install
COPY app/frontend/ ./
RUN npm run build

# --- Stage 2: Production ---
FROM python:3.13-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx supervisor curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn[standard]

WORKDIR /app
COPY app/backend/main.py ./main.py
COPY app/backend/data/ ./data/
COPY --from=frontend-build /build/dist/ /usr/share/nginx/html/
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost/api/health || exit 1
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
