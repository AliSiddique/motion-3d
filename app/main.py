"""
Animation Generation Web Service

FastAPI service that wraps Tencent HY-Motion 1.0 for text-to-animation generation,
with SMPL → Mixamo/RPM retargeting built in.

Endpoints:
    POST /generate         — Generate animation from text prompt
    GET  /health           — Health check
    GET  /animations/{id}  — Download a previously generated animation
"""

import hashlib
import logging
import os
import time
import uuid

import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.motion_generator import get_generator
from app.retargeter import (
    extract_rest_from_glb,
    export_bvh,
    export_glb_animation,
    export_gltf_animation,
    make_loopable,
    retarget_to_mixamo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Device: {config.DEVICE}, Model: {config.MODEL_VARIANT}")

    generator = get_generator()
    generator.load()

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="GoodThoughts Animation Generator",
    description="Text-to-animation service using Tencent HY-Motion 1.0",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ───────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text description of the motion")
    duration: float | None = Field(None, ge=0.5, le=12.0, description="Duration in seconds (auto if omitted)")
    cfg_scale: float = Field(config.DEFAULT_CFG_SCALE, ge=1.0, le=10.0, description="Guidance scale")
    seed: int | None = Field(None, description="Random seed (random if omitted)")
    format: str = Field("bvh", description="Output format: 'bvh', 'glb', or 'gltf' (ASCII debug)")
    zero_root_xz: bool = Field(False, description="Zero out root XZ translation (in-place animation)")
    scale: float = Field(1.0, description="Position scale factor (1.0=meters, 100.0=centimeters)")
    loop: bool = Field(False, description="Crossfade end→start for seamless looping")


class GenerateResponse(BaseModel):
    id: str
    prompt: str
    rewritten_prompt: str | None
    duration: float
    fps: int
    num_frames: int
    format: str
    generation_time: float
    download_url: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_variant: str
    device: str


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    generator = get_generator()
    return HealthResponse(
        status="ok",
        model_loaded=generator._loaded,
        model_variant=config.MODEL_VARIANT,
        device=config.DEVICE,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate animation from JSON body (no custom rig)."""
    return await _generate(
        prompt=req.prompt,
        duration=req.duration,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        format=req.format,
        zero_root_xz=req.zero_root_xz,
        scale=req.scale,
        loop=req.loop,
        avatar_glb_bytes=None,
    )


@app.post("/generate/upload", response_model=GenerateResponse)
async def generate_with_upload(
    prompt: str = Form(...),
    duration: float | None = Form(None),
    cfg_scale: float = Form(config.DEFAULT_CFG_SCALE),
    seed: int | None = Form(None),
    format: str = Form("glb"),
    zero_root_xz: bool = Form(False),
    scale: float = Form(1.0),
    loop: bool = Form(False),
    avatar_glb: UploadFile | None = File(None),
):
    """Generate animation from multipart form data (supports custom rig GLB upload)."""
    avatar_bytes = None
    if avatar_glb is not None:
        avatar_bytes = await avatar_glb.read()
        if len(avatar_bytes) == 0:
            avatar_bytes = None

    return await _generate(
        prompt=prompt,
        duration=duration,
        cfg_scale=cfg_scale,
        seed=seed,
        format=format,
        zero_root_xz=zero_root_xz,
        scale=scale,
        loop=loop,
        avatar_glb_bytes=avatar_bytes,
    )


async def _generate(
    prompt: str,
    duration: float | None,
    cfg_scale: float,
    seed: int | None,
    format: str,
    zero_root_xz: bool,
    scale: float,
    loop: bool,
    avatar_glb_bytes: bytes | None,
) -> GenerateResponse:
    """Shared generation logic for both JSON and multipart endpoints."""
    generator = get_generator()

    if not generator._loaded:
        raise HTTPException(503, "Model not loaded yet")

    # Generate motion
    result = generator.generate(
        prompt=prompt,
        duration=duration,
        cfg_scale=cfg_scale,
        seed=seed,
    )

    fps = result["fps"]

    # Extract custom rest pose from avatar GLB if provided
    rest_rotations = None
    if avatar_glb_bytes is not None:
        try:
            rest_rotations = extract_rest_from_glb(avatar_glb_bytes)
        except Exception as e:
            logger.warning(f"Failed to extract rest pose from avatar GLB: {e}")
            # Fall back to RPM defaults

    # Fix ground alignment: HY-Motion aligns ground using the lowest SMPL *mesh*
    # vertex, but retargeting only transfers joint rotations/positions. The SMPL foot
    # mesh extends below its foot joints, so the mesh-based offset makes RPM feet
    # float above ground. Re-align using foot joint positions instead.
    transl = result["transl"]
    keypoints3d = result.get("keypoints3d")
    if keypoints3d is not None:
        # SMPL joints 10 (left_foot) and 11 (right_foot) are the toe/foot-tip joints,
        # closest to actual ground contact. After mesh-based alignment their Y > 0
        # by the foot mesh thickness — subtract that so joints define the ground.
        foot_y = keypoints3d[:, [10, 11], 1]  # (L, 2)
        min_foot_y = float(np.min(foot_y))
        if min_foot_y > 0:
            logger.info(f"Correcting ground offset: foot joints were {min_foot_y:.4f}m above mesh ground")
            transl = transl.copy()
            transl[:, 1] -= min_foot_y

    # Retarget SMPL → Mixamo
    retargeted = retarget_to_mixamo(
        rot6d=result["rot6d"],
        transl=transl,
        fps=fps,
        zero_root_xz=zero_root_xz,
        scale=scale,
        rest_rotations=rest_rotations,
    )

    # Apply loop crossfade if requested
    if loop:
        make_loopable(retargeted)

    # Generate unique ID for this animation
    anim_id = uuid.uuid4().hex[:12]

    # Export to requested format
    fmt = format.lower()
    if fmt == "bvh":
        output_path = os.path.join(config.OUTPUT_DIR, f"{anim_id}.bvh")
        export_bvh(retargeted, output_path)
    elif fmt == "glb":
        output_path = os.path.join(config.OUTPUT_DIR, f"{anim_id}.glb")
        export_glb_animation(retargeted, output_path)
    elif fmt == "gltf":
        output_path = os.path.join(config.OUTPUT_DIR, f"{anim_id}.gltf")
        export_gltf_animation(retargeted, output_path)
    else:
        raise HTTPException(400, f"Unsupported format: {fmt}. Use 'bvh', 'glb', or 'gltf'.")

    return GenerateResponse(
        id=anim_id,
        prompt=prompt,
        rewritten_prompt=result.get("prompt") if result.get("prompt") != prompt else None,
        duration=result["duration"],
        fps=fps,
        num_frames=retargeted["num_frames"],
        format=fmt,
        generation_time=result["generation_time"],
        download_url=f"/animations/{anim_id}.{fmt}",
    )


@app.get("/animations/{filename}")
async def download_animation(filename: str):
    # Sanitize filename
    safe_name = Path(filename).name
    file_path = os.path.join(config.OUTPUT_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(404, "Animation not found")

    if safe_name.endswith(".bvh"):
        media_type = "text/plain"
    elif safe_name.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif safe_name.endswith(".gltf"):
        media_type = "model/gltf+json"
    else:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type, filename=safe_name)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )
