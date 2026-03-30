FROM node:20-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY explorer/package.json explorer/package-lock.json ./explorer/
WORKDIR /app/explorer
RUN npm ci

WORKDIR /app
COPY . .

WORKDIR /app/explorer
RUN npm run build

ENV NODE_ENV=production \
    PORT=8080 \
    PYTHON_EXECUTABLE=python3 \
    SIMULATION_REPO_ROOT=/app \
    SIMULATION_WORKERS=6 \
    SIMULATION_QUEUE_TTL_MS=900000

EXPOSE 8080

CMD ["npm", "run", "start"]
