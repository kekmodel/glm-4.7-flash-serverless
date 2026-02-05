"""
SGLang Engine for GLM-4.7-Flash
Based on runpod-workers/worker-sglang
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

        # Model and basic options
        options = {
            "MODEL_NAME": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "DTYPE": "--dtype",
            "QUANTIZATION": "--quantization",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "CONTEXT_LENGTH": "--context-length",
            "TOOL_CALL_PARSER": "--tool-call-parser",
            "REASONING_PARSER": "--reasoning-parser",
            # Speculative decoding
            "SPECULATIVE_ALGORITHM": "--speculative-algorithm",
            "SPECULATIVE_NUM_STEPS": "--speculative-num-steps",
            "SPECULATIVE_EAGLE_TOPK": "--speculative-eagle-topk",
            "SPECULATIVE_NUM_DRAFT_TOKENS": "--speculative-num-draft-tokens",
        }

        # Boolean flags
        boolean_flags = [
            "TRUST_REMOTE_CODE",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
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
