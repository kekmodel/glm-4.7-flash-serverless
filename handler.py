"""
RunPod Serverless Handler for GLM-4.7-Flash via SGLang
"""
import runpod
import requests

SGLANG_URL = "http://localhost:8000"


def handler(job):
    """
    RunPod serverless handler.
    Forwards requests to local SGLang server.
    """
    job_input = job.get("input", {})

    # OpenAI-compatible route
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        payload = job_input.get("openai_input", {})

        if route.startswith("/v1/") or route == "/generate":
            response = requests.post(
                f"{SGLANG_URL}{route}",
                json=payload,
                timeout=300
            )
            return response.json()
        elif route == "/v1/models":
            response = requests.get(f"{SGLANG_URL}{route}", timeout=30)
            return response.json()

    # Direct chat completion
    if "messages" in job_input:
        payload = {
            "model": job_input.get("model", "glm-4.7-flash"),
            "messages": job_input["messages"],
            "max_tokens": job_input.get("max_tokens", 2048),
            "temperature": job_input.get("temperature", 1.0),
            "top_p": job_input.get("top_p", 0.95),
        }
        response = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        return response.json()

    # Direct generate
    if "text" in job_input or "prompt" in job_input:
        text = job_input.get("text") or job_input.get("prompt")
        payload = {
            "text": text,
            "sampling_params": job_input.get("sampling_params", {
                "max_new_tokens": 2048,
                "temperature": 1.0,
            })
        }
        response = requests.post(
            f"{SGLANG_URL}/generate",
            json=payload,
            timeout=300
        )
        return response.json()

    return {"error": "Invalid input. Provide 'messages', 'text', 'prompt', or 'openai_route'."}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
