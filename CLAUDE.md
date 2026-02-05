# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RunPod serverless deployment for GLM-4.7-Flash LLM using SGLang inference server. The server exposes OpenAI-compatible APIs and supports tool calling/reasoning parsers specific to GLM-4.7.

## Build & Deploy

```bash
# Build Docker image
docker build -t kekmodel/glm-4.7-flash:latest .

# Push to registry (triggers GitHub Actions on main)
docker push kekmodel/glm-4.7-flash:latest

# Local testing with GPU
docker compose up
```

GitHub Actions automatically builds and pushes on commits to `main`.

## Testing the Endpoint

Use `test_endpoint.ipynb` to test deployed RunPod endpoints. Requires `.env` with:
- `RUNPOD_API_KEY` - RunPod API key
- `ENDPOINT_ID` - RunPod endpoint ID

## Architecture

- **Dockerfile**: Builds SGLang server image with GLM-4.7-Flash model and EAGLE speculative decoding
- **docker-compose.yml**: Local GPU testing configuration
- **test_endpoint.ipynb**: Integration tests for RunPod endpoint (health, models, chat, generate, tokenizer)

## SGLang Server Configuration

The Dockerfile configures SGLang with:
- Model: `zai-org/GLM-4.7-Flash`
- Tool call parser: `glm47`
- Reasoning parser: `glm45`
- EAGLE speculative decoding (3 steps, topk=1, 4 draft tokens)
- Memory fraction: 0.8
- Served model name: `glm-4.7-flash`

## RunPod Request Format

Requests go through RunPod's proxy using `/runsync`:
```json
{
  "input": {
    "openai_route": "/v1/chat/completions",
    "openai_input": { ... }
  }
}
```

## Key Dependencies (in Docker)

- sglang 0.3.2.dev with PR-17247 for GLM-4.7 support
- transformers from specific commit for GLM-4.7 compatibility
