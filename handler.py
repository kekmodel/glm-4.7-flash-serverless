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

    try:
        # OpenAI-compatible route
        if "openai_route" in job_input:
            route = job_input["openai_route"]
            payload = job_input.get("openai_input", {})

            # GET endpoints
            if route in ["/v1/models", "/health", "/get_model_info"]:
                response = requests.get(f"{SGLANG_URL}{route}", timeout=30)
                return response.json()

            # POST endpoints
            response = requests.post(
                f"{SGLANG_URL}{route}",
                json=payload,
                timeout=300
            )
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
            # Optional parameters
            if "tools" in job_input:
                payload["tools"] = job_input["tools"]
            if "tool_choice" in job_input:
                payload["tool_choice"] = job_input["tool_choice"]

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

    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Try reducing max_tokens or simplifying the request."}
    except requests.exceptions.ConnectionError:
        return {"error": "SGLang server not responding. Container may still be initializing."}
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
