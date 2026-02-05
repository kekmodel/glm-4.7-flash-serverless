#!/bin/bash
# GitHub Actions 빌드 트리거

set -e

cd "$(dirname "$0")/.."

# 변경사항 있으면 커밋 & 푸시
if [[ -n $(git status --porcelain) ]]; then
    echo "변경사항 커밋 중..."
    git add .
    git commit -m "Update: $(date +%Y-%m-%d_%H:%M:%S)"
    git push
    echo "푸시 완료 → 자동 빌드 시작됨"
else
    echo "변경사항 없음. 수동 빌드 트리거..."
    gh workflow run docker-build.yml
fi

echo ""
echo "빌드 상태 확인: gh run watch"
echo "또는: https://github.com/kekmodel/glm-4.7-flash-serverless/actions"
