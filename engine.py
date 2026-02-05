"""
SGLang Engine for GLM-4.7-Flash
Updated with latest SGLang options (2026-02)
"""
import subprocess
import time
import requests
import os


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_NAME", "zai-org/GLM-4.7-Flash"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host", self.host,
            "--port", str(self.port),
        ]

        # All options - synced with latest SGLang
        options = {
            # Model & Tokenizer
            "MODEL_NAME": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "TOKENIZER_MODE": "--tokenizer-mode",
            "LOAD_FORMAT": "--load-format",
            "DTYPE": "--dtype",
            "CONTEXT_LENGTH": "--context-length",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",

            # Quantization
            "QUANTIZATION": "--quantization",
            "KV_CACHE_DTYPE": "--kv-cache-dtype",

            # Memory & Scheduling
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_RUNNING_REQUESTS": "--max-running-requests",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
            "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
            "SCHEDULE_POLICY": "--schedule-policy",
            "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
            "PAGE_SIZE": "--page-size",

            # Parallelism
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "DATA_PARALLEL_SIZE": "--data-parallel-size",
            "PIPELINE_PARALLEL_SIZE": "--pipeline-parallel-size",
            "LOAD_BALANCE_METHOD": "--load-balance-method",

            # Attention Backends
            "ATTENTION_BACKEND": "--attention-backend",
            "DECODE_ATTENTION_BACKEND": "--decode-attention-backend",
            "PREFILL_ATTENTION_BACKEND": "--prefill-attention-backend",
            "SAMPLING_BACKEND": "--sampling-backend",

            # Logging
            "LOG_LEVEL": "--log-level",
            "LOG_LEVEL_HTTP": "--log-level-http",
            "STREAM_INTERVAL": "--stream-interval",
            "RANDOM_SEED": "--random-seed",

            # API & Storage
            "API_KEY": "--api-key",
            "FILE_STORAGE_PATH": "--file-storage-path",

            # Tool Calling & Reasoning
            "TOOL_CALL_PARSER": "--tool-call-parser",
            "REASONING_PARSER": "--reasoning-parser",

            # Speculative Decoding (EAGLE, etc.)
            "SPECULATIVE_ALGORITHM": "--speculative-algorithm",
            "SPECULATIVE_DRAFT_MODEL_PATH": "--speculative-draft-model-path",
            "SPECULATIVE_NUM_STEPS": "--speculative-num-steps",
            "SPECULATIVE_EAGLE_TOPK": "--speculative-eagle-topk",
            "SPECULATIVE_NUM_DRAFT_TOKENS": "--speculative-num-draft-tokens",
            "SPECULATIVE_ACCEPT_THRESHOLD_SINGLE": "--speculative-accept-threshold-single",
            "SPECULATIVE_ACCEPT_THRESHOLD_ACC": "--speculative-accept-threshold-acc",
            "SPECULATIVE_DRAFT_ATTENTION_BACKEND": "--speculative-draft-attention-backend",

            # Hierarchical Cache
            "HICACHE_RATIO": "--hicache-ratio",
            "HICACHE_SIZE": "--hicache-size",
            "HICACHE_WRITE_POLICY": "--hicache-write-policy",
        }

        # Boolean flags
        boolean_flags = [
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

        # Add options from environment variables
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags
        for flag in boolean_flags:
            if os.getenv(flag, "").lower() in ("true", "1", "yes"):
                command.append(f"--{flag.lower().replace('_', '-')}")

        print(f"Starting SGLang with command: {' '.join(command)}")
        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5):
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server shut down.")
