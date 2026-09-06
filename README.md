# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="731" height="468" alt="Screenshot 2026-09-05 214633" src="https://github.com/user-attachments/assets/1b65c73d-555e-4097-a2de-7ae2f5e6851f" />


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
| **Viggle-Animate Conditioning (H3)** | Builds conditioning + AV latent: video-first reference order, both references nested on the canvas short edge (the driving clip's, unless width/height are overridden) — the layout the finetune was trained with |

Everything else is ComfyUI core: **Load Diffusion Model**, **Load LoRA (Model Only)**,
**ModelSamplingMiniMaxH3** (shift_video 3.0), **KSampler**, **VAE Decode**, **Save Video**.

| cond_vid | ref_img | output |
|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/2529857c-2667-4641-9d2e-5dcb3c03913d" controls muted></video> | <img src="https://github.com/user-attachments/assets/f6adf969-03d5-4a58-bd30-5ec2d0bc604b" width="300"> | <video src="https://github.com/user-attachments/assets/deedde68-80de-47d2-9e1f-7eaa3bc35457" controls muted></video> |



| example_1 | example_2 | example_3 | example_4 |
|:---:|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/c5198b4c-9544-4e5a-bd1e-83ae4477bb0b" controls muted></video> | <video src="https://github.com/user-attachments/assets/afb74de1-3d3d-42ae-9885-1d5b1c6e86af" controls muted></video> | <video src="https://github.com/user-attachments/assets/5ddf1bb1-e744-406b-8b9f-2b729e4ecc4b" controls muted></video> | <video src="https://github.com/user-attachments/assets/8d02389d-67ca-46f0-b9e3-78994d944a90" controls muted></video> |




### euler / beta - 6-steps

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

## Model Links

**diffusion_models** (pick one — pruned is the VRAM-friendly option)

- [minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors) (21 GB)
- [minimax_h3_ref2va_viggle_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_int8_convrot.safetensors) (47 GB)
- [minimax_h3_ref2va_viggle_bf16.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_bf16.safetensors) (66.3 GB, max quality)

**loras** (DMD 4-step accelerator — pick one)

- [viggle_animate_dmd_lora_r64.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors) (0.94 GB, recommended)
- [viggle_animate_dmd_lora.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora.safetensors) (3.8 GB, full rank)

**text_cond**

- [fixed_embed_fwd_anyframe.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors) — precomputed text conditioning, load with **Load Text Conditioning (Viggle)** (no text encoder needed)

**vae**

- [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (low VRAM)
- or [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)

## Model Storage Location

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors
│   ├── 📂 loras/
│   │   └── viggle_animate_dmd_lora_r64.safetensors
│   ├── 📂 text_cond/
│   │   └── fixed_embed_fwd_anyframe.safetensors
│   └── 📂 vae/
│       └── minimax_h3_video_vae_int8_convrot.safetensors
```

## Workflow Notes

- Custom nodes required: [ComfyUI-Viggle-Animate-H3](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3) (Viggle Animate Conditioning + Load Text Conditioning) and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (fast preview).
- **Load Video**: set `frame_load_cap` equal to the conditioning node's `length` (e.g. 124) and `force_rate` to 24.
- Output resolution follows the driving video by default; set the conditioning node's `width`/`height` to override (each axis rounds to 32), or pre-scale the clip with **Scale Image to Total Pixels**. Tested canvas range: **0.4–0.98 MP**.
- Sampler: euler, er_sde, exp_heun_2_x0, lcm, simple, normal, beta, bong_tangent, 4-8 steps, cfg 1.0, ModelSamplingMiniMaxH3 shift 3.0.
- Stacks with **Comfy Kitchen** and **block sparse attention** patches.

## Limitations

- **Identity drift on re-entry**: when the subject leaves the camera view and re-enters, the re-entry settles toward the driving video's original appearance rather than the reference image. The same applies when the subject moves far from the reference pose or makes abrupt large motions (e.g. a backflip) — the further from the still, the weaker the identity hold.

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

## Report Issue

- Weights/conversion issues: [ComfyUI-Viggle-Animate-H3/issues](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3/issues)
