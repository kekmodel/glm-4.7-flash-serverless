"""
RunPod Serverless Handler for GLM-4.7-Flash via SGLang
Based on runpod-workers/worker-sglang
"""
import requests
import runpod
import os
from engine import SGlangEngine

# Initialize the engine
engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()


def get_max_concurrency(default=300):
    """Returns the maximum concurrency value."""
    return int(os.getenv("MAX_CONCURRENCY", default))


async def async_handler(job):
    """Handle the requests asynchronously with streaming support."""
    job_input = job["input"]

    # Case 1: OpenAI style payload with route specified
    if job_input.get("openai_route"):
        openai_route = job_input.get("openai_route")
        openai_input = job_input.get("openai_input", {})

        openai_url = f"{engine.base_url}{openai_route}"
        headers = {"Content-Type": "application/json"}

        # All routes use POST (SGLang convention)
        response = requests.post(openai_url, headers=headers, json=openai_input)

        if openai_input.get("stream", False):
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")
        else:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    # Case 2: Direct chat completions (messages format)
    elif "messages" in job_input:
        openai_url = f"{engine.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Set default model if not specified
        if "model" not in job_input:
            job_input["model"] = os.getenv("SERVED_MODEL_NAME", "glm-4.7-flash")

        response = requests.post(openai_url, headers=headers, json=job_input)

        if job_input.get("stream", False):
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")
        else:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    # Case 3: Direct /generate endpoint (SGLang native)
    else:
        generate_url = f"{engine.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        response = requests.post(generate_url, json=job_input, headers=headers)

        if response.status_code == 200:
            yield response.json()
        else:
            yield {
                "error": f"Generate request failed with status code {response.status_code}",
                "details": response.text,
            }


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": get_max_concurrency,
        "return_aggregate_stream": True,
    }
)
