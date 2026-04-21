"""Robust 70B download via snapshot_download with retry loop."""
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

MODEL = "NousResearch/Llama-2-70b-hf"

for attempt in range(1, 30):
    print(f"=== attempt {attempt} ===", flush=True)
    try:
        path = snapshot_download(
            MODEL,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer.*"],
            max_workers=4,
            etag_timeout=60,
        )
        print(f"=== SUCCESS: {path} ===", flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(15)
else:
    print("=== FAILED after 30 attempts ===", flush=True)
