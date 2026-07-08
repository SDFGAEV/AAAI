#!/bin/bash
# C-ACT Ubuntu Setup — run once on the target machine
set -euo pipefail

echo "=== C-ACT Ubuntu Setup ==="

echo "[1/4] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev openjdk-17-jdk libgl1 ffmpeg xvfb

echo "[2/4] Python venv..."
python3.10 -m venv venv && source venv/bin/activate

echo "[3/4] Python deps..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy fastapi uvicorn requests Pillow omegaconf hydra-core transformers accelerate sentence-transformers shortuuid psutil pyyaml

echo "[4/4] Health check..."
python experiments/health_check.py

echo "=== Done. source venv/bin/activate && bash experiments/run_all.sh --workers 4 ==="
