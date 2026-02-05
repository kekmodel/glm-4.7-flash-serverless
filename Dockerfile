# GLM-4.7-Flash with SGLang
FROM lmsysorg/sglang:latest

ENV HF_HOME=/runpod-volume/huggingface

# Upgrade to GLM-4.7 supported versions
RUN pip install --break-system-packages \
    sglang==0.3.2.dev9039+pr-17247.g90c446848 \
    --extra-index-url https://sgl-project.github.io/whl/pr/ && \
    pip install --break-system-packages \
    git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa

EXPOSE 8000

CMD ["python3", "-m", "sglang.launch_server", \
     "--model-path", "zai-org/GLM-4.7-Flash", \
     "--tp-size", "1", \
     "--tool-call-parser", "glm47", \
     "--reasoning-parser", "glm45", \
     "--speculative-algorithm", "EAGLE", \
     "--speculative-num-steps", "3", \
     "--speculative-eagle-topk", "1", \
     "--speculative-num-draft-tokens", "4", \
     "--mem-fraction-static", "0.8", \
     "--served-model-name", "glm-4.7-flash", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
