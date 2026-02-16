import os

# Device: "cuda" for GPU, "cpu" for CPU-only (very slow)
DEVICE = os.getenv("DEVICE", "cuda")

# GPU IDs (comma-separated, e.g. "0" or "0,1")
DEVICE_IDS = os.getenv("DEVICE_IDS", "0")

# Model variant: "full" or "lite"
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "lite")

# HuggingFace: both variants live under one repo as subfolders
HF_REPO_ID = "tencent/HY-Motion-1.0"

# Subfolder names within the HF repo
MODEL_FOLDERS = {
    "full": "HY-Motion-1.0",
    "lite": "HY-Motion-1.0-Lite",
}

# Local model cache path (for HuggingFace downloads)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/data/models")

# Resolved model path: either pre-downloaded or will be auto-downloaded
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_CACHE_DIR, MODEL_FOLDERS[MODEL_VARIANT]))

# Output directory for generated animations
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/output")

# Default generation settings
DEFAULT_CFG_SCALE = float(os.getenv("DEFAULT_CFG_SCALE", "5.0"))
DEFAULT_NUM_SEEDS = int(os.getenv("DEFAULT_NUM_SEEDS", "1"))
DEFAULT_MAX_DURATION = float(os.getenv("DEFAULT_MAX_DURATION", "5.0"))

# Whether to use LLM prompt rewriting (uses extra VRAM)
ENABLE_PROMPT_REWRITE = os.getenv("ENABLE_PROMPT_REWRITE", "false").lower() == "true"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8100"))
