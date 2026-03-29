# Multi-stage build: Node frontend → Python+Node runtime
# Build context: repo root (needs Python files + explorer/)

# --- Stage 1: Build frontend assets ---
FROM node:22-slim AS frontend-build
WORKDIR /app/explorer
COPY explorer/package*.json ./
RUN npm ci
COPY explorer/ .
RUN npm run build

# --- Stage 2: Production runtime (Python + Node) ---
FROM python:3.12-slim

# Install Node.js 22
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node dependencies (production only)
COPY explorer/package*.json ./explorer/
RUN cd explorer && npm ci --omit=dev

# Copy Python simulation code (repo root)
COPY *.py ./
COPY params/ ./params/
COPY data/ ./data/

# Copy server source + shared types
COPY explorer/server/ ./explorer/server/
COPY explorer/src/types/ ./explorer/src/types/
COPY explorer/src/data/ ./explorer/src/data/

# Copy pre-built frontend from Stage 1
COPY --from=frontend-build /app/explorer/dist ./explorer/dist

# Persistent data volume mount point
RUN mkdir -p /app/explorer/server/data

WORKDIR /app/explorer
EXPOSE 3201

CMD ["npx", "tsx", "server/index.ts"]
