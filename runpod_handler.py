"""
Runpod Serverless handler for text-to-motion.

Wraps HY-Motion (app.motion_generator) + SMPL→Mixamo retargeting
(app.retargeter), generates N variations, and uploads each result straight to
Cloudflare R2 — so the main app just records the returned keys/urls (this mirrors
how we already call fal: submit a job, poll, read the output).

Input (event["input"]):
    prompt        (str, required)
    duration      (float, optional, 0.5–12)
    seeds         (list[int], optional) — one variation per seed
    num_variations(int, optional, default 4) — used when seeds omitted
    formats       (list[str], optional, default ["glb"]) — "glb" and/or "bvh"
    loop          (bool, optional) — crossfade for seamless looping
    in_place      (bool, optional) — zero root XZ translation
    cfg_scale     (float, optional, default 5.0)

Output:
    {
      "job_id": str, "prompt": str, "fps": 30,
      "variations": [
        {"seed": int, "num_frames": int, "duration": float,
         "assets": {"glb": {"key","url"}, "bvh": {"key","url"}}},
        ...
      ]
    }
"""

import logging
import os
import random
import uuid

import boto3
import numpy as np
from botocore.config import Config as BotoConfig

from app.motion_generator import get_generator
from app.retargeter import (
    export_bvh,
    export_glb_animation,
    make_loopable,
    retarget_to_mixamo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("runpod_handler")

# ── R2 (S3-compatible) ───────────────────────────────────────────────
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")

_s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    config=BotoConfig(signature_version="s3v4", region_name="auto"),
)

_CONTENT_TYPE = {"glb": "model/gltf-binary", "bvh": "text/plain"}


def _upload(local_path: str, key: str, fmt: str) -> dict:
    _s3.upload_file(local_path, R2_BUCKET, key, ExtraArgs={"ContentType": _CONTENT_TYPE[fmt]})
    url = f"{R2_PUBLIC_URL}/{key}" if R2_PUBLIC_URL else None
    return {"key": key, "url": url}


# Load the model once per worker (paid for on cold start, then reused).
_generator = get_generator()
_generator.load()


def _ground_corrected_transl(result):
    """Re-align feet to the ground using foot-joint Y (matches app/main.py)."""
    transl = result["transl"]
    kp = result.get("keypoints3d")
    if kp is not None:
        min_foot_y = float(np.min(kp[:, [10, 11], 1]))
        if min_foot_y > 0:
            transl = transl.copy()
            transl[:, 1] -= min_foot_y
    return transl


def handler(event):
    inp = (event or {}).get("input") or {}

    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    duration = inp.get("duration")
    cfg_scale = float(inp.get("cfg_scale") or 5.0)
    loop = bool(inp.get("loop"))
    zero_root_xz = bool(inp.get("in_place"))
    formats = [f for f in (inp.get("formats") or ["glb"]) if f in _CONTENT_TYPE] or ["glb"]

    seeds = inp.get("seeds")
    if not seeds:
        n = max(1, min(8, int(inp.get("num_variations") or 4)))
        seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]

    job_id = uuid.uuid4().hex[:12]
    fps = 30
    variations = []

    for i, seed in enumerate(seeds):
        result = _generator.generate(
            prompt=prompt, duration=duration, cfg_scale=cfg_scale, seed=int(seed)
        )
        fps = result["fps"]
        transl = _ground_corrected_transl(result)

        retargeted = retarget_to_mixamo(
            rot6d=result["rot6d"], transl=transl, fps=fps, zero_root_xz=zero_root_xz
        )
        if loop:
            make_loopable(retargeted)

        assets = {}
        base_key = f"motion/{job_id}/variation_{i}/animation"

        if "glb" in formats:
            path = f"/tmp/{job_id}_{i}.glb"
            export_glb_animation(retargeted, path)
            assets["glb"] = _upload(path, f"{base_key}.glb", "glb")
            os.remove(path)

        if "bvh" in formats:
            path = f"/tmp/{job_id}_{i}.bvh"
            export_bvh(retargeted, path)
            assets["bvh"] = _upload(path, f"{base_key}.bvh", "bvh")
            os.remove(path)

        variations.append(
            {
                "seed": int(seed),
                "num_frames": retargeted["num_frames"],
                "duration": result["duration"],
                "assets": assets,
            }
        )
        logger.info("variation %d/%d done (seed=%s)", i + 1, len(seeds), seed)

    return {
        "job_id": job_id,
        "prompt": prompt,
        "rewritten_prompt": result.get("prompt") if result.get("prompt") != prompt else None,
        "fps": fps,
        "variations": variations,
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
