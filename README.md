# animation-gen

A FastAPI service that generates 3D skeletal animations from text prompts using [Tencent HY-Motion 1.0](https://github.com/Tencent/HY-Motion), with automatic retargeting to Mixamo/ReadyPlayerMe skeletons.

## What it does

1. Takes a text prompt (e.g. "A person is dancing")
2. HY-Motion's diffusion transformer generates SMPL-H motion data (22 joints in 6D rotation format at 30fps)
3. Retargets the SMPL joint rotations to Mixamo/ReadyPlayerMe bone names using quaternion-based rest pose calibration
4. Exports as BVH, GLB, or glTF — ready for Godot, Three.js, Blender, etc.

## CPU inference optimization

HY-Motion ships with bfloat16 as the default tensor dtype. This is fine on NVIDIA GPUs and Intel CPUs with AMX (Advanced Matrix eXtensions), but causes a severe performance problem on AMD CPUs.

### The bfloat16 bug on AMD

AMD EPYC (and most consumer Ryzen) processors do not have hardware support for bfloat16 arithmetic. When PyTorch encounters bfloat16 tensors on these CPUs, it falls back to software emulation, which is roughly **~300x slower** than native float32 and also prevents MKL multi-threading from kicking in. A generation that should take 1-3 minutes on CPU instead takes hours.

The fix: when running on CPU, we explicitly cast all model pipelines to float32 via `pipeline.float()` before inference. This gives us proper MKL-accelerated GEMM and brings CPU generation time down.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | `cuda` or `cpu` |
| `DEVICE_IDS` | `0` | Comma-separated GPU IDs for multi-GPU |
| `MODEL_VARIANT` | `lite` | `full` (1.0B params, 26GB VRAM) or `lite` (0.46B params, 24GB VRAM) |
| `MODEL_PATH` | `/data/models/HY-Motion-1.0-Lite` | Local path or HuggingFace repo ID |
| `ENABLE_PROMPT_REWRITE` | `false` | LLM-based prompt rewriting via Qwen3-8B |
| `DEFAULT_CFG_SCALE` | `5.0` | Classifier-free guidance scale |
| `DEFAULT_NUM_SEEDS` | `1` | Number of variations per generation (more = more VRAM) |
| `DEFAULT_MAX_DURATION` | `5.0` | Max motion duration in seconds (0.5-12s) |

## Retargeting features

- **20-bone SMPL → Mixamo mapping** (hips, spine chain, arms, legs, neck, head)
- **Custom rig support** — extract rest poses from your own GLB avatar
- **Loop crossfade** — 15-frame SLERP blend for seamless looping animations
- **Gimbal lock recovery** — detects 180-degree rotations and SLERP-interpolates unstable frames
- **Quaternion continuity** — prevents sign flips between frames for smooth interpolation
- **Position scaling** — 1.0 = meters, 100.0 = centimeters (for FBX-convention engines)
- **Zero root XZ** — lock root translation for in-place animations

## Running

```bash
# GPU (recommended)
docker compose up

# CPU (slow but works)
DEVICE=cpu docker compose up
```

## Usage

Generate a BVH animation:

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A person is dancing"}' \
  | jq .download_url
# => "/animations/abc123def456.bvh"

curl -O http://localhost:8100/animations/abc123def456.bvh
```

Generate a looping GLB with custom options:

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person waves hello",
    "format": "glb",
    "duration": 3.0,
    "loop": true,
    "zero_root_xz": true
  }' | jq .
```

Generate with a custom avatar rig (multipart upload):

```bash
curl -X POST http://localhost:8100/generate/upload \
  -F "prompt=A person is walking" \
  -F "format=glb" \
  -F "avatar_glb=@my_avatar.glb"
```

Health check:

```bash
curl http://localhost:8100/health | jq .
```

## License

MIT — see [LICENSE.md](LICENSE.md)
