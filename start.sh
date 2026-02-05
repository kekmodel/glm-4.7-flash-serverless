#!/bin/bash
set -e

echo "Starting SGLang server..."

python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.9 \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000 &

# Wait for server to be ready
echo "Waiting for SGLang server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "SGLang server is ready!"

# Start RunPod handler
echo "Starting RunPod handler..."
python3 handler.py
