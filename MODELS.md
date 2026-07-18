# Model Survey — Open Models for Onyx

**Status:** Research snapshot, July 2026
**Target use cases:** (A) 480p → 2x with quality enhancement, (B) 1080p → 4K,
(C) 25/30 fps → 60 fps interpolation.

## Verdict

Achievable. Frame interpolation and clean-source upscaling are solved by open
models at Topaz-or-better quality. Degraded-source restoration (the historically
hard case) became genuinely competitive with the 2025 wave of one-step diffusion
restoration models (SeedVR2, FlashVSR) — at the cost of being far slower than
Topaz's models. On a queue-based NAS server that renders overnight, that trade
is acceptable, which is the core thesis of this project.

## Use case C — frame interpolation (25/30 → 60 fps): SOLVED

| Model | License | Notes |
|---|---|---|
| **RIFE 4.25 / 4.26** (Practical-RIFE) | MIT | The gold standard. Real-time-ish on midrange GPUs, arbitrary-timestep (so 25→60 works directly, not just 2x). 4.25 is the recommended default. |
| rife-ncnn-vulkan | MIT | NCNN/Vulkan port — our AMD/Intel GPU path. |
| FILM (Google) | Apache-2.0 | Better on large motion gaps, much slower. Niche fallback. |

Quality is comparable to Topaz Apollo/Chronos for typical content. Requirements
for us: scene-change detection (skip interpolation across cuts — standard in
RIFE frontends) and a PAL heuristic (25 fps content is often 24 fps film sped
up; offer 25→50 and detelecine options, not just blind →60).

## Use case B — 1080p → 4K: STRONG

For reasonably clean 1080p sources this is the easy upscale.

| Model | License | Speed | Notes |
|---|---|---|---|
| **Real-ESRGAN / compact SRVGGNet** | BSD | Fast (near-realtime w/ TensorRT) | Workhorse tier. |
| **OpenModelDB community fine-tunes** | Mostly permissive | Fast | Hundreds of content-specific 2x models (anime, live-action, film grain preservation). Big advantage over Topaz: pick a model tuned for *your* content. |
| HAT / DAT / transformer SR | Apache-2.0 | Slow | Max-fidelity single-frame tier. |
| **FlashVSR** (CVPR 2026) | Apache-2.0 | ~17 fps @ 768×1408 on A100 | One-step diffusion *streaming* VSR; reported to beat SeedVR2 on longer real-footage HD input. Community forks run in ~12 GB VRAM with tiling. |

## Use case A — 480p restoration + 2x (the hard one): NOW ACHIEVABLE

This is where Topaz historically won outright. The 2025 open releases changed it:

| Model | License | VRAM | Notes |
|---|---|---|---|
| **SeedVR2** (ByteDance) 3B / 7B | Apache-2.0 (code); open weights | 7B fits 16 GB via block swap; GGUF quants run on 6–8 GB | One-step diffusion restoration. Blind tests score 7B-Sharp ~9.7/10 vs Topaz 9.8/10 — effectively at parity, sometimes better on compressed input. Slow: minutes per short clip. |
| QTGMC (VapourSynth) | GPL | n/a (CPU) | Not AI — still the best deinterlacer in existence. Mandatory pre-pass for DVD/VHS sources. |
| DVD/SD-specific ESRGAN fine-tunes (OpenModelDB) | Mostly permissive | 4–8 GB | The "fast tier": QTGMC → tuned compact model. Often good enough, 10–50× faster than diffusion. |
| RealBasicVSR / TSCUNet | Apache-2.0 | 8+ GB | Older temporal restoration; superseded by SeedVR2-class for quality but still useful mid-tier. |

## Honest comparison vs Topaz (2026)

- **Model quality:** effectively parity. Independent comparisons (including a
  Hugging Face community writeup) now rate tuned FlashVSR/SeedVR2 pipelines at
  or above Topaz output on many sources.
- **Speed:** Topaz's proprietary models are much faster than open
  diffusion-class models (near-realtime vs minutes-per-minute). Our answer:
  two-tier design — a **fast tier** (compact ESRGAN + RIFE, ~realtime, good for
  bulk/preview) and a **quality tier** (SeedVR2/FlashVSR, overnight renders).
- **Product polish:** Topaz wins today on one-click UX. That's the gap Onyx
  exists to close, with a workflow (queue, watch folders, API, preview slider)
  Topaz structurally can't offer.
- **Cost:** Topaz moved to subscriptions in Oct 2025 ($299–699/yr). Onyx is
  free on hardware you already own.

## Hardware reality check

- Fast tier: any NVIDIA GPU with ≥6 GB VRAM.
- Quality tier: 12 GB (RTX 3060-class) comfortable minimum; 16 GB+ ideal for
  SeedVR2-7B via block swap. System RAM matters too (~32 GB recommended for
  FlashVSR-class pipelines).

### Reference target: 12 GB (RTX 3080 Ti now → RTX 5070 later)

The primary dev/deploy target is a 12 GB card, so 12 GB is the hard budget for
default quality-tier presets:

- SeedVR2 **3B** (FP16) fits comfortably at 12 GB — make it the default
  quality-tier model. 7B is offered as an opt-in "max quality" preset using
  aggressive block swap (slower, still functional at 12 GB).
- FlashVSR runs at 12 GB via community tiling forks — enable tiling by default.
- Architecture gotchas the app must absorb:
  - **FP8 needs Ada/Hopper/Blackwell.** Ampere (3080 Ti) has no FP8 tensor
    cores — FP8 checkpoints save VRAM there but not time. Auto-pick precision
    per GPU generation (FP16 on Ampere, FP8 on Blackwell).
  - **TensorRT engines are per-architecture.** Cache built engines keyed by
    GPU compute capability + model hash; rebuild transparently after a GPU
    swap (sm_86 → sm_120).
  - **Blackwell requires CUDA 12.8+.** Base image must be new enough to cover
    both sm_86 and sm_120 in one build.

## Design implications

1. Two-tier model strategy baked into presets (fast vs quality).
2. QTGMC pre-pass is non-negotiable for interlaced SD sources — VapourSynth
   stays in the stack.
3. SeedVR2/FlashVSR are PyTorch/diffusion pipelines, not simple ONNX graphs —
   the inference layer needs a plugin-style backend abstraction (ONNX Runtime
   for compact models, torch pipeline runners for diffusion models), which
   affects the DESIGN.md §2.2 assumption that ONNX Runtime alone suffices.
4. Block-swap / tiling / VRAM budgeting must be first-class job parameters.
