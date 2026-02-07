"""
SGLang Engine for GLM-4.7-Flash
"""

import os
import shlex
import subprocess
import time

import requests

OPTIONS = {
    "TOKENIZER_PATH": "--tokenizer-path",
    "TOKENIZER_MODE": "--tokenizer-mode",
    "LOAD_FORMAT": "--load-format",
    "DTYPE": "--dtype",
    "CONTEXT_LENGTH": "--context-length",
    "SERVED_MODEL_NAME": "--served-model-name",
    "CHAT_TEMPLATE": "--chat-template",
    "QUANTIZATION": "--quantization",
    "KV_CACHE_DTYPE": "--kv-cache-dtype",
    "MEM_FRACTION_STATIC": "--mem-fraction-static",
    "MAX_RUNNING_REQUESTS": "--max-running-requests",
    "MAX_TOTAL_TOKENS": "--max-total-tokens",
    "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
    "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
    "SCHEDULE_POLICY": "--schedule-policy",
    "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
    "PAGE_SIZE": "--page-size",
    "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    "DATA_PARALLEL_SIZE": "--data-parallel-size",
    "PIPELINE_PARALLEL_SIZE": "--pipeline-parallel-size",
    "LOAD_BALANCE_METHOD": "--load-balance-method",
    "ATTENTION_BACKEND": "--attention-backend",
    "DECODE_ATTENTION_BACKEND": "--decode-attention-backend",
    "PREFILL_ATTENTION_BACKEND": "--prefill-attention-backend",
    "SAMPLING_BACKEND": "--sampling-backend",
    "LOG_LEVEL": "--log-level",
    "LOG_LEVEL_HTTP": "--log-level-http",
    "STREAM_INTERVAL": "--stream-interval",
    "RANDOM_SEED": "--random-seed",
    "API_KEY": "--api-key",
    "FILE_STORAGE_PATH": "--file-storage-path",
    "TOOL_CALL_PARSER": "--tool-call-parser",
    "REASONING_PARSER": "--reasoning-parser",
    "SPECULATIVE_ALGORITHM": "--speculative-algorithm",
    "SPECULATIVE_DRAFT_MODEL_PATH": "--speculative-draft-model-path",
    "SPECULATIVE_NUM_STEPS": "--speculative-num-steps",
    "SPECULATIVE_EAGLE_TOPK": "--speculative-eagle-topk",
    "SPECULATIVE_NUM_DRAFT_TOKENS": "--speculative-num-draft-tokens",
    "SPECULATIVE_ACCEPT_THRESHOLD_SINGLE": "--speculative-accept-threshold-single",
    "SPECULATIVE_ACCEPT_THRESHOLD_ACC": "--speculative-accept-threshold-acc",
    "SPECULATIVE_DRAFT_ATTENTION_BACKEND": "--speculative-draft-attention-backend",
    "HICACHE_RATIO": "--hicache-ratio",
    "HICACHE_SIZE": "--hicache-size",
    "HICACHE_WRITE_POLICY": "--hicache-write-policy",
}

BOOLEAN_FLAGS = [
    "SKIP_TOKENIZER_INIT",
    "TRUST_REMOTE_CODE",
    "LOG_REQUESTS",
    "SHOW_TIME_COST",
    "DISABLE_RADIX_CACHE",
    "DISABLE_CUDA_GRAPH",
    "DISABLE_OUTLINES_DISK_CACHE",
    "DISABLE_FLASHINFER_AUTOTUNE",
    "ENABLE_TORCH_COMPILE",
    "ENABLE_P2P_CHECK",
    "ENABLE_FLASHINFER_MLA",
    "ENABLE_METRICS",
    "ENABLE_HIERARCHICAL_CACHE",
    "ENABLE_MULTI_LAYER_EAGLE",
    "ENABLE_PIECEWISE_CUDA_GRAPH",
    "TRITON_ATTENTION_REDUCE_IN_FP32",
]

TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled(env_var):
    return os.getenv(env_var, "").strip().lower() in TRUE_VALUES


class SGlangEngine:
    def __init__(self, model=None, host=None, port=None):
        self.model = model or os.getenv("MODEL_NAME", "zai-org/GLM-4.7-Flash")
        self.host = host or os.getenv("HOST", "0.0.0.0")
        if port is None:
            try:
                port = int(os.getenv("PORT", "8000"))
            except ValueError:
                port = 8000
        self.port = port
        request_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        self.base_url = f"http://{request_host}:{self.port}"
        self.process = None

    def _build_command(self):
        command = ["python3", "-m", "sglang.launch_server", "--model-path", self.model]
        command.extend(["--host", self.host, "--port", str(self.port)])

        for env_var, flag in OPTIONS.items():
            value = os.getenv(env_var)
            if value:
                command.extend([flag, value])

        for flag in BOOLEAN_FLAGS:
            if _is_enabled(flag):
                command.append(f"--{flag.lower().replace('_', '-')}")

        return command

    def start_server(self):
        if self.process and self.process.poll() is None:
            print("SGLang server is already running.")
            return

        command = self._build_command()
        print(f"Starting SGLang: {shlex.join(command)}")
        self.process = subprocess.Popen(command)
        print(f"Server PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5, request_timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {self.process.returncode}"
                )
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=request_timeout)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError(f"Server failed to start within {timeout} seconds.")

    def shutdown(self, timeout=30):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()
            print("Server shut down.")
        self.process = None
