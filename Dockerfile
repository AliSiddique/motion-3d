FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy HY-Motion repo from submodule and install its dependencies
COPY vendor/HY-Motion-1.0 /opt/HY-Motion-1.0
RUN pip3 install --no-cache-dir -r /opt/HY-Motion-1.0/requirements.txt

ENV HYMOTION_REPO_PATH=/opt/HY-Motion-1.0
ENV PYTHONPATH=/opt/HY-Motion-1.0

# Symlink repo directories so relative paths from WORKDIR resolve correctly
# (HY-Motion code uses relative paths for body model assets and stats)
RUN ln -s /opt/HY-Motion-1.0/scripts scripts \
    && ln -s /opt/HY-Motion-1.0/stats stats

# Install app Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir "huggingface-hub[hf_xet]"

# Pre-download ALL model weights into the image so nothing is fetched at runtime.
# 1) HY-Motion Lite diffusion model (~500 MB)
# 2) Qwen3-8B text encoder (~16 GB)
# 3) CLIP ViT-L/14 sentence embeddings (~1 GB)
ARG HF_TOKEN=""
ARG MODEL_VARIANT=lite
ENV HF_TOKEN=$HF_TOKEN
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
folders = {'full': 'HY-Motion-1.0', 'lite': 'HY-Motion-1.0-Lite'}; \
variant = '${MODEL_VARIANT}'; \
snapshot_download('tencent/HY-Motion-1.0', allow_patterns=f'{folders[variant]}/**', local_dir='/data/models')"

RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPTokenizer, CLIPTextModel; \
import torch; \
print('Downloading Qwen3-8B...'); \
AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16); \
print('Downloading CLIP ViT-L/14...'); \
CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', max_length=77); \
CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14'); \
print('All models downloaded.')"

# Clear token and block all runtime HuggingFace downloads
ENV HF_TOKEN=""
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY app/ ./app/

# Output volume only — models are baked into the image
VOLUME ["/data/output"]

EXPOSE 8100

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
