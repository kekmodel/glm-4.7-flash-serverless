# GLM-4.7-Flash with SGLang (Official Worker Pattern)
FROM lmsysorg/sglang:latest

ENV HF_HOME=/runpod-volume/huggingface

# GLM-4.7-Flash specific versions
RUN pip install --break-system-packages \
    sglang==0.3.2.dev9039+pr-17247.g90c446848 \
    --extra-index-url https://sgl-project.github.io/whl/pr/ && \
    pip install --break-system-packages \
    git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa && \
    pip install --break-system-packages runpod requests

WORKDIR /app
COPY engine.py .
COPY handler.py .

# Default environment variables for GLM-4.7-Flash
ENV MODEL_NAME=zai-org/GLM-4.7-Flash
ENV SERVED_MODEL_NAME=glm-4.7-flash
ENV TOOL_CALL_PARSER=glm47
ENV REASONING_PARSER=glm45
ENV MEM_FRACTION_STATIC=0.9
ENV TENSOR_PARALLEL_SIZE=1
# Speculative decoding (comment out if OOM)
ENV SPECULATIVE_ALGORITHM=EAGLE
ENV SPECULATIVE_NUM_STEPS=3
ENV SPECULATIVE_EAGLE_TOPK=1
ENV SPECULATIVE_NUM_DRAFT_TOKENS=4

EXPOSE 8000

CMD ["python3", "handler.py"]
