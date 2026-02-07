"""
RunPod Serverless Handler for GLM-4.7-Flash via SGLang
Based on runpod-workers/worker-sglang
"""

import os

import requests
import runpod

from engine import SGlangEngine

engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()

BASE_URL = engine.base_url
DEFAULT_MODEL = os.getenv("SERVED_MODEL_NAME", "glm-4.7-flash")
HEADERS = {"Content-Type": "application/json"}
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))


def get_max_concurrency(_current=None):
    try:
        return int(os.getenv("MAX_CONCURRENCY", "300"))
    except ValueError:
        return 300


def _error_response(message, details=None):
    error = {"error": message}
    if details:
        error["details"] = details
    return error


def _build_upstream_request(job_input):
    if job_input.get("openai_route"):
        openai_route = str(job_input["openai_route"])
        if not openai_route.startswith("/"):
            openai_route = f"/{openai_route}"
        payload = job_input.get("openai_input", {})
        if not isinstance(payload, dict):
            payload = {}
        return f"{BASE_URL}{openai_route}", payload, True

    if "messages" in job_input:
        payload = dict(job_input)
        payload.setdefault("model", DEFAULT_MODEL)
        return f"{BASE_URL}/v1/chat/completions", payload, True

    return f"{BASE_URL}/generate", job_input, False


async def async_handler(job):
    job_input = job.get("input") if isinstance(job, dict) else None
    if not isinstance(job_input, dict):
        yield _error_response("Invalid input: `job['input']` must be an object.")
        return

    url, payload, should_stream = _build_upstream_request(job_input)

    try:
        if not should_stream:
            response = requests.post(
                url,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                yield response.json()
            else:
                yield _error_response(
                    f"Generate request failed: {response.status_code}",
                    response.text,
                )
            return

        with requests.post(
            url,
            headers=HEADERS,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                yield _error_response(
                    f"Request failed: {response.status_code}",
                    response.text,
                )
                return

            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")
    except requests.RequestException as exc:
        yield _error_response("Upstream request failed.", str(exc))
    except ValueError as exc:
        yield _error_response("Invalid upstream JSON response.", str(exc))


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": get_max_concurrency,
        "return_aggregate_stream": True,
    }
)
