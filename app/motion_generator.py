"""
HY-Motion 1.0 inference wrapper.

Loads the Tencent HY-Motion model via T2MRuntime and generates SMPL-H motion
data from text prompts. Supports both full (1.0B) and lite (0.46B) variants.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app import config

logger = logging.getLogger(__name__)


class MotionGenerator:
    """Wraps HY-Motion T2MRuntime for motion generation."""

    def __init__(self):
        self.runtime = None
        self.device = config.DEVICE
        self._loaded = False

    def load(self):
        """Load the HY-Motion model. Call once at startup."""
        if self._loaded:
            return

        model_path = config.MODEL_PATH
        logger.info(f"Loading HY-Motion model from: {model_path} on device={self.device}")

        # Ensure the repo is on the Python path
        repo_path = os.getenv("HYMOTION_REPO_PATH", "/opt/HY-Motion-1.0")
        import sys
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Auto-download from HuggingFace if model not present locally
        cfg_path = os.path.join(model_path, "config.yml")
        if not os.path.exists(cfg_path):
            logger.info(f"Config not found at {cfg_path}, downloading from HuggingFace...")
            os.environ.setdefault("USE_HF_MODELS", "1")
            from huggingface_hub import snapshot_download

            target_folder = config.MODEL_FOLDERS[config.MODEL_VARIANT]
            local_dir = snapshot_download(
                repo_id=config.HF_REPO_ID,
                allow_patterns=f"{target_folder}/*",
                local_dir=config.MODEL_CACHE_DIR,
            )
            model_path = os.path.join(local_dir, target_folder)
            cfg_path = os.path.join(model_path, "config.yml")
            logger.info(f"Downloaded model to: {model_path}")

        ckpt_path = os.path.join(model_path, "latest.ckpt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found at {ckpt_path}, will attempt loading anyway")

        from hymotion.utils.t2m_runtime import T2MRuntime

        force_cpu = self.device != "cuda"
        device_ids = [int(d) for d in config.DEVICE_IDS.split(",")] if not force_cpu else None

        self.runtime = T2MRuntime(
            config_path=cfg_path,
            ckpt_name=ckpt_path,
            device_ids=device_ids,
            force_cpu=force_cpu,
            disable_prompt_engineering=not config.ENABLE_PROMPT_REWRITE,
        )

        if force_cpu:
            # Cast all pipelines to float32 — bfloat16 on AMD EPYC (no AMX)
            # is ~300x slower than float32 and can't use MKL multi-threading.
            for p in self.runtime.pipelines:
                p.float()
                logger.info("Cast pipeline to float32 for CPU inference")
            logger.warning(
                "Running on CPU — expect ~1-3 min per generation with float32. "
                "GPU with 24+ GB VRAM is strongly recommended."
            )

        self._loaded = True
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        duration: Optional[float] = None,
        cfg_scale: float = config.DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
    ) -> dict:
        """Generate motion from a text prompt.

        Args:
            prompt: Text description of the desired motion (English or Chinese).
            duration: Duration in seconds (0.5-12.0). None = auto-estimate.
            cfg_scale: Classifier-free guidance scale (1.0-10.0).
            seed: Random seed for reproducibility. None = random.

        Returns:
            dict with keys:
                - "motion": np.ndarray of shape (num_frames, 201)
                - "fps": int (typically 30)
                - "duration": float (actual duration in seconds)
                - "prompt": str (possibly rewritten prompt)
                - "generation_time": float (seconds taken)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if seed is None:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        if duration is None:
            duration = config.DEFAULT_MAX_DURATION
        duration = max(0.5, min(12.0, duration))

        logger.info(f"Generating motion: prompt='{prompt}', duration={duration}s, seed={seed}")
        start_time = time.time()

        # T2MRuntime.generate_motion returns (html, fbx_files, model_output)
        # model_output is a dict with: latent_denorm (B,L,201), rot6d, transl, keypoints3d, text
        _html, _fbx_files, model_output = self.runtime.generate_motion(
            text=prompt,
            seeds_csv=str(seed),
            duration=duration,
            cfg_scale=cfg_scale,
            output_format="dict",
        )

        # Extract smoothed rot6d (B, L, J, 6) and transl (B, L, 3) from model output.
        # These are produced by _decode_o6dp which applies slerp smoothing on rotations
        # and Savitzky-Golay smoothing on translation, plus ground alignment.
        rot6d = model_output["rot6d"]
        transl = model_output["transl"]
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.cpu().numpy()
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu().numpy()
        if rot6d.ndim == 4:
            rot6d = rot6d[0]    # (L, J, 6) — take first batch element
        if transl.ndim == 3:
            transl = transl[0]  # (L, 3)

        fps = 30  # HY-Motion outputs at 30fps
        num_frames = rot6d.shape[0]
        actual_duration = num_frames / fps

        gen_time = time.time() - start_time
        logger.info(
            f"Generated {num_frames} frames in {gen_time:.1f}s "
            f"({actual_duration:.1f}s of motion)"
        )

        return {
            "rot6d": rot6d,       # (L, J, 6) — 22 joints in 6D rotation format
            "transl": transl,     # (L, 3) — root translation
            "fps": fps,
            "duration": actual_duration,
            "prompt": prompt,
            "generation_time": gen_time,
        }


# Singleton instance
_generator: Optional[MotionGenerator] = None


def get_generator() -> MotionGenerator:
    global _generator
    if _generator is None:
        _generator = MotionGenerator()
    return _generator
