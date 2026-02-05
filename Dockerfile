# GLM-4.7-Flash with SGLang
FROM lmsysorg/sglang:latest

ENV HF_HOME=/runpod-volume/huggingface

# Upgrade to GLM-4.7 supported versions + handler dependencies
RUN pip install --break-system-packages \
    sglang==0.3.2.dev9039+pr-17247.g90c446848 \
    --extra-index-url https://sgl-project.github.io/whl/pr/ && \
    pip install --break-system-packages \
    git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa && \
    pip install --break-system-packages runpod requests

WORKDIR /app
COPY handler.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
