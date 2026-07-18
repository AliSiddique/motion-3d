# Deploying to Runpod Serverless

This turns the service into a **Runpod Serverless endpoint** that our main app
calls like fal: submit a job → poll → read the output. Each job runs HY-Motion
for N seeds, retargets SMPL→Mixamo, and uploads the GLB/BVH results straight to
your R2 bucket. The response is just the R2 keys/urls.

Files added for this:
- `runpod_handler.py` — the serverless handler
- `Dockerfile.runpod` — the worker image (same proven base, handler entrypoint)

## 1. Build & push the image

The image bakes in the HY-Motion **lite** model + CLIP (prompt rewriting stays
off, so the 16 GB Qwen model is skipped). Runpod GPUs are x86, so build for
`linux/amd64`.

> ⚠️ Building on an Apple-Silicon Mac cross-compiles amd64 + downloads a couple GB
> of weights — slow and heavy. Easiest is to build on a **Linux x86 box** (a cheap
> cloud VM) or via **GitHub Actions**. If you do build on the Mac, add
> `--platform linux/amd64` (buildx).

```sh
cd hymotion-animation-gen-mixamo-retargeting-dev
git submodule update --init --recursive   # pull vendor/HY-Motion-1.0

# Replace <registry> with your Docker Hub user or ghcr.io/<user>
docker buildx build --platform linux/amd64 \
  -f Dockerfile.runpod \
  -t <registry>/hymotion-runpod:latest \
  --push .
```

## 2. Create the Serverless endpoint

Runpod console → **Serverless → New Endpoint → Import Custom**:

- **Container image:** `<registry>/hymotion-runpod:latest`
- **GPU:** 24 GB+ — **L40S / A40 / RTX 4090 / A6000** all work
- **Container disk:** ~15 GB (the image is a few GB)
- **Workers:** Active `0`, Max `1–3`, **idle timeout** ~30 s (scale to zero = pay only per generation)
- **Environment variables** (from your app's `.env` — the handler uploads to R2):

  | Key | Value |
  |---|---|
  | `R2_ACCOUNT_ID` | `de0f9e01badf5065b9e0c75d2f1162f7` |
  | `R2_ACCESS_KEY_ID` | *(your R2 access key)* |
  | `R2_SECRET_ACCESS_KEY` | *(your R2 secret)* |
  | `R2_BUCKET` | `3daicanvas` |
  | `R2_PUBLIC_URL` | `https://assets.3daicanvas.com` |

Deploy. Note the **Endpoint ID** (in the endpoint URL).

## 3. Get a Runpod API key

Runpod console → **Settings → API Keys → Create** (read/write). Copy it.

## 4. Test it

```sh
EP=<ENDPOINT_ID>; KEY=<RUNPOD_API_KEY>

# submit
curl -s -X POST https://api.runpod.ai/v2/$EP/run \
  -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -d '{"input":{"prompt":"A person waves hello","num_variations":2,"formats":["glb"]}}'
# => {"id":"<job>","status":"IN_QUEUE"}

# poll (first call is a cold start — a couple minutes while the worker boots + loads the model)
curl -s https://api.runpod.ai/v2/$EP/status/<job> -H "Authorization: Bearer $KEY" | jq .
# => when COMPLETED:
# output.variations[].assets.glb.url  → https://assets.3daicanvas.com/motion/<job_id>/variation_0/animation.glb
```

If a GLB URL loads and plays in a glTF viewer, the pipeline works end to end.

## 5. Hand off

Send me the **Endpoint ID** and the **Runpod API key** and I'll wire the main
app's backend to it (submit → poll → store → credits + a Text-to-Motion studio
page), exactly like the fal integration.

## Notes / gotchas
- **First request is slow** (cold start: worker boot + model load, ~1–3 min). Warm requests are ~1–2 min for 4 variations. Keep 1 active worker if you want instant responses (costs idle GPU).
- **Roblox R15 + FBX export** aren't in this repo yet (it does standard/Mixamo GLB+BVH). Those are the next add-ons once this baseline works.
- If the build fails on a missing dependency or CUDA mismatch, that's expected first-pass — send me the error and I'll adjust the Dockerfile.
