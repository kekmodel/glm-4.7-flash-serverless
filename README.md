# GLM-4.7-Flash SGLang Server

순정 SGLang 서버. OpenAI-compatible API 전체 지원.

## 배포

```bash
docker build -t YOUR_DOCKERHUB/glm-4.7-flash:latest .
docker push YOUR_DOCKERHUB/glm-4.7-flash:latest
```

## RunPod 설정

| 설정 | 값 |
|------|-----|
| Docker Image | `YOUR_DOCKERHUB/glm-4.7-flash:latest` |
| GPU | RTX 4090 (24GB) |
| Min Workers | 0 |
| Container Disk | 20GB |
| Volume | `/runpod-volume` |

## 지원 엔드포인트

- `POST /v1/chat/completions` - Chat
- `POST /v1/completions` - Text completion
- `GET /v1/models` - 모델 목록
- `GET /health` - 헬스체크

## 사용 예시

```bash
curl http://YOUR_ENDPOINT:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
