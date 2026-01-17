# BGFS — Body-part Geometry Fixer Sidecar (for Wan 2.2)

This repo gives you an anatomy-focused "sidecar" you can plug into a baseline open-source video diffusion model (starting with **Wan2.2**) to reduce systematic geometry failures (hands/fingers, faces, feet, etc.).

**What you get (end-to-end):**
1. **ROI cache builder**: extracts the user-selected region (e.g., hands) from your videos + produces pseudo-label keypoint heatmaps (via MediaPipe).
2. **Heatmap critic**: trains a small differentiable network that predicts keypoint heatmaps from the ROI crop.
3. **Post-hoc fixer**: a practical first win — optimize the VAE latents of each frame to reduce kinematic inconsistencies, guided by the critic.

This is intentionally modular:
- You can keep Wan as-is, and ship the sidecar as a separate "fixer".
- Later, you can upgrade to LoRA/adapter training inside Wan (2-stage denoiser, etc.) using the same critic losses.

---

## 1) Installation (Runpod)

Wan2.2 support in Diffusers is on the *main* docs track, which usually means installing Diffusers from source.

```bash
# system deps
apt-get update && apt-get install -y ffmpeg git

# python deps
pip install -U pip
pip install -e ".[mediapipe]"

# Diffusers from source (recommended for Wan2.2 pipelines)
pip install -U "git+https://github.com/huggingface/diffusers.git"
```

If your Wan2.2 checkpoint complains about missing components (e.g. image_encoder / image_processor), some users had to install a specific branch during the initial release period (see the model discussions on HF).

---

## 2) Data layout

Put training videos here:

```
/workspace/data/videos/
  person_0001.mp4
  person_0002.mp4
  ...
```

BGFS will sample short clips from these videos and build a cache under `cache_dir`.

---

## 3) Default configuration

Edit `configs/default.yaml`:
- `roi.region`: which body part to focus on (`hands`, `face`, `feet`, ...)
- `roi.out_hw`: size for ROI crops (default 256x256)

---

## 4) Build ROI cache (MediaPipe pseudo-labels)

```bash
bgfs precompute --config configs/default.yaml
```

This creates:
- `cache_dir/manifest.json`
- many `*.npz` files containing `roi` crops and `hm` keypoint heatmaps.

---

## 5) Train the critic

```bash
bgfs train-critic-cmd --config configs/default.yaml --out-dir outputs/critic --epochs 5
```

Output:
- `outputs/critic/critic_best.pt`

---

## 6) Generate with Wan2.2

```bash
bgfs generate \
  --model-id "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
  --prompt "a person waving at the camera" \
  --negative-prompt "extra fingers, fused fingers, deformed hands" \
  --out out.mp4 \
  --num-frames 81 \
  --guidance-scale 5.0 \
  --seed 42
```

---

## 7) Fix geometry (post-hoc latent optimization)

```bash
bgfs fix \
  --in out.mp4 \
  --out fixed.mp4 \
  --critic outputs/critic/critic_best.pt \
  --model-id "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
  --region hands \
  --steps 25 \
  --lr 0.07 \
  --strength 1.0
```

**What it does:**
- Reads the generated video
- Finds the region (via MediaPipe) per frame
- Encodes frames into the Wan VAE latent space
- Optimizes latents so the critic predicts more kinematically-consistent keypoints
- Decodes and writes a new video

---

## Model levers (what to tune)

### ROI / Tracking
- `roi.region`: hands / face / feet / upper_body / full_body
- `roi.margin`: larger margin makes the crop more forgiving but may dilute the loss
- `roi.min_conf`: skip low-confidence landmarks

### Critic
- `critic.heatmap_sigma`: tighter vs smoother keypoint supervision
- `critic.lr`, `critic.batch_size`, `critic.num_workers`

### Fixer
- `steps`: more steps = stronger correction but higher risk of texture drift
- `lr`: step size in latent space
- `strength`: anatomy loss weight

---

## Notes / next upgrades

This repo ships the fastest path to visible improvement (a post-hoc fixer). The next step is to push the *same anatomy losses* into Wan training:
- LoRA for `transformer` and optionally `transformer_2` (Wan2.2 uses two denoisers)
- Add losses at low-noise timesteps (where geometry errors are "decided")
- Distill the sidecar into the generator (so you don't need post-hoc optimization)

