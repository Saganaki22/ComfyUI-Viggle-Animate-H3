# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

ComfyUI nodes for **[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** — a 33.1 B
full finetune of MiniMax-H3's `ref2va` transformer for **character replacement in video**: it
takes a driving video and a reference still, and re-renders the performer(s) in the clip as the
character in the still. Motion, camera, timing, background and lighting come from the video;
identity comes from the image.

No text encoder, no prompt: conditioning is one frozen 362-token embedding computed once by the
Viggle team with Qwen3-VL (`assets/fixed_prompt.txt`), identical for every render. The sampler is
DMD2-distilled — **4 steps (3 forward passes)**, so a 124-frame shot is fast even on consumer
hardware.

## Nodes

| Node | What it does |
|---|---|
| **Load Text Conditioning (Viggle)** | Dropdown loader for frozen text conditioning in `models/text_cond/` |
| **Viggle-Animate Conditioning (H3)** | Builds conditioning + AV latent: video-first reference order, both references nested on the driving clip's short edge — the layout the finetune was trained with |

Everything else is ComfyUI core: **Load Diffusion Model**, **Load LoRA (Model Only)**,
**ModelSamplingMiniMaxH3** (shift_video 3.0), **KSampler**, **VAE Decode**, **Save Video**.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

### Weights — converted, ComfyUI-native: [drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

**Diffusion models** → `ComfyUI/models/diffusion_models/`

| File | Size | Notes |
|---|---|---|
| [minimax_h3_ref2va_viggle_bf16.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_bf16.safetensors) | 66.3 GB | Full precision — needs ≥ 96 GB VRAM or offloading |
| [minimax_h3_ref2va_viggle_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_int8_convrot.safetensors) | 46.3 GB | int8 weights (convrot-quantized) |
| [minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors) | 20.3 GB | + low-rank `adaln_proj` — **fits a 32 GB card, recommended** |

**DMD LoRAs** → `ComfyUI/models/loras/` (the distilled delta that collapses sampling to 4 steps —
required; it is a delta on the *finetuned* transformer, not the stock MiniMax one)

| File | Size | Notes |
|---|---|---|
| [viggle_animate_dmd_lora.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora.safetensors) | 3.5 GB | Original rank 128, exact conversion |
| [viggle_animate_dmd_lora_r64.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors) | 0.88 GB | SVD-truncated, 99.26% spectral energy |
| [viggle_animate_dmd_lora_r29.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r29.safetensors) | 0.40 GB | SVD-truncated, 98.2% spectral energy |

**Frozen text conditioning** → `ComfyUI/models/text_cond/`

| File | Notes |
|---|---|
| [fixed_embed_fwd_anyframe.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors) | The frozen 362-token embedding — replaces the text encoder entirely |

**VAEs** (from the base model, not the finetune) → `ComfyUI/models/vae/`
from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
`vae/minimax_h3_video_vae_fp16.safetensors` and `vae/minimax_h3_audio_vae_fp32.safetensors`

## Workflow recipe

```
Load Diffusion Model (viggle pruned_int8_convrot)
  -> Load LoRA (Model Only) (viggle_animate_dmd_lora, strength 1)
  -> ModelSamplingMiniMaxH3 (shift_video 3.0, shift_audio 3.0)
  -> KSampler (4 steps, cfg 1.0)

Load Video (24 fps, frame_load_cap = length) --+
Load Image (reference still) ------------------+--> Viggle-Animate Conditioning (H3) --+
Load VAE (minimax_h3 video VAE) ---------------+                                      |
Load Text Conditioning (Viggle) ---------------+                            KSampler --+--> VAE Decode -> Save Video
```

Settings that matter:

- **4 steps, cfg 1.0** — the evaluated operating point; more steps over-sharpens
- **shift_video 3.0** — the base default of 12 is wrong for the distilled model
- **Driving clip length = output length** (cap Load Video at your `length`, e.g. 124 @ 24 fps)
- **width/height 0** = inherit the driving clip's geometry (output must be a multiple of 32);
  set them explicitly only to override

## Links

- Original model + inference code: [huggingface.co/Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate) · [viggle.ai](https://viggle.ai)
- Base model: [huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Comfy-repackaged base VAEs: [huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- Converted ComfyUI weights, quants & LoRAs: [huggingface.co/drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

## Citation

If you use the model in published work, cite the original:

```bibtex
@misc{viggle2026animate,
  title  = {Viggle-Animate: Character Replacement in Video from a Single Repainted Frame},
  author = {Viggle Research},
  year   = {2026},
  url    = {https://huggingface.co/Viggle/Viggle-Animate}
}
```

## License & responsible use

- The **weights** are a Model Derivative of MiniMax H3 — the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3) applies
  to them (read it before redistributing or shipping a product on them). This includes the
  converted/quantized variants linked above.
- This **node pack** is Apache 2.0 (see `LICENSE`).
- The model puts a person into footage they did not shoot; identity comes from the image you
  supply. Do not run it on people who have not consented, and label what you generate as
  AI-generated (see the original repo's intended-use section).
