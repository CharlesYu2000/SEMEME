import os
from pathlib import Path


def set_hf_token_env(token_file=".hf_token"):
    if os.environ.get("HF_TOKEN"):
        return
    path = Path(token_file)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
