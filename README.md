# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="731" height="468" alt="Screenshot 2026-09-05 214633" src="https://github.com/user-attachments/assets/1b65c73d-555e-4097-a2de-7ae2f5e6851f" />

<br>

ComfyUI nodes for **[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** — a 33.1 B full finetune of MiniMax-H3's `ref2va` transformer for **character replacement in video**: it takes a driving video and a reference still, and re-renders the performer(s) in the clip as the character in the still. Motion, camera, timing, background and lighting come from the video; identity comes from the image.

No text encoder, no prompt: conditioning is one frozen 362-token embedding computed once by the Viggle team with Qwen3-VL (`assets/fixed_prompt.txt`), identical for every render.

The sampler is DMD2-distilled and works with very low step counts. **4–8 steps are recommended, with 6 steps being the sweet spot for speed vs. quality.** The included workflows also contain **manual sigma schedules derived from the upstream sampling formula**, allowing the distilled schedule to be reproduced directly with ComfyUI's existing **ManualSigmas** node.

A 4-step manual schedule contains **4 sigma points including the final `0.0`**, which means only **3 model forward passes** are performed. Likewise, 6 sigma points correspond to 5 forward passes, and 8 sigma points correspond to 7 forward passes.

## Nodes

| Node                                 | What it does                                                                                                                                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Load Text Conditioning (Viggle)**  | Dropdown loader for frozen text conditioning in `models/text_cond/`                                                                                                                                               |
| **Viggle-Animate Conditioning (H3)** | Builds conditioning + AV latent: video-first reference order, both references nested on the canvas short edge (the driving clip's, unless width/height are overridden) — the layout the finetune was trained with |

Everything else is ComfyUI core: **Load Diffusion Model**, **Load LoRA (Model Only)**, **ModelSamplingMiniMaxH3** (shift_video 3.0), **KSampler**, **VAE Decode**, **Save Video**, and optionally **ManualSigmas** when using the supplied manual sigma workflows.

|                                                       cond_vid                                                       |                                                 ref_img                                                 |                                                        output                                                        |
| :------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/user-attachments/assets/2529857c-2667-4641-9d2e-5dcb3c03913d" controls muted></video> | <img src="https://github.com/user-attachments/assets/f6adf969-03d5-4a58-bd30-5ec2d0bc604b" width="300"> | <video src="https://github.com/user-attachments/assets/deedde68-80de-47d2-9e1f-7eaa3bc35457" controls muted></video> |

|                                                       example_1                                                      |                                                       example_2                                                      |                                                       example_3                                                      |                                                       example_4                                                      |
| :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/user-attachments/assets/c5198b4c-9544-4e5a-bd1e-83ae4477bb0b" controls muted></video> | <video src="https://github.com/user-attachments/assets/afb74de1-3d3d-42ae-9885-1d5b1c6e86af" controls muted></video> | <video src="https://github.com/user-attachments/assets/5ddf1bb1-e744-406b-8b9f-2b729e4ecc4b" controls muted></video> | <video src="https://github.com/user-attachments/assets/8d02389d-67ca-46f0-b9e3-78994d944a90" controls muted></video> |

|                                                       example_5                                                      |                                                       example_6                                                      |
| :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/user-attachments/assets/d4533465-705a-4487-8f14-04770c3d84b6" controls muted></video> | <video src="https://github.com/user-attachments/assets/e519cb64-18c3-4fa7-abe3-d5e57fc0b72e" controls muted></video> |

### euler / beta - 6 steps

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

## Model Links

**diffusion_models** (pick one — pruned is the VRAM-friendly option)

* [minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors) (21 GB)
* [minimax_h3_ref2va_viggle_int8_convrot.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_int8_convrot.safetensors) (47 GB)
* [minimax_h3_ref2va_viggle_bf16.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_bf16.safetensors) (66.3 GB, max quality)

**loras** (DMD accelerator — pick one)

* [viggle_animate_dmd_lora_r64.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors) (0.94 GB, recommended)
* [viggle_animate_dmd_lora.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora.safetensors) (3.8 GB, full rank)

**text_cond**

* [fixed_embed_fwd_anyframe.safetensors](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors) — precomputed text conditioning, load with **Load Text Conditioning (Viggle)** (no text encoder needed)

**vae**

* [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors) (3.17 GB, low VRAM)
* or [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) (5.21 GB)

## Model Storage Location

```text
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

* Custom nodes required: [ComfyUI-Viggle-Animate-H3](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3) (Viggle Animate Conditioning + Load Text Conditioning) and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (fast preview).
* **Load Video**: set `frame_load_cap` equal to the conditioning node's `length` (e.g. 124) and `force_rate` to 24.
* Output resolution follows the driving video by default; set the conditioning node's `width`/`height` to override (each axis rounds to 32), or pre-scale the clip with **Scale Image to Total Pixels**. Tested canvas range: **0.4–0.98 MP**.
* Recommended sampling range is **4–8 steps**, with **6 steps recommended as the best speed/quality balance**.
* Tested samplers/schedulers include `euler`, `er_sde`, `exp_heun_2_x0`, `lcm` / `simple`, `normal`, `beta`, and `bong_tangent`, with CFG `1.0` and **ModelSamplingMiniMaxH3** shift `3.0`.
* The included workflows contain both normal scheduler configurations and **manual sigma configurations derived from the upstream formula**.
* Stacks with **Comfy Kitchen** and **block sparse attention** patches.

## Sampling & Manual Sigmas

### Basic Scheduler

For normal ComfyUI scheduler usage, **6–8 steps** are recommended.

Good scheduler choices include:

```text
simple
beta
normal
bong_tangent
```

**6 steps is the recommended sweet spot** for speed vs. quality. You can go down to 4 steps for faster generation or up to 8 steps when you want to favor quality and stability.

### Manual Sigmas

The included workflows also provide manual sigma versions using ComfyUI's existing **ManualSigmas** node.

These sigma values are **derived from the upstream sampling formula** rather than being hand-tuned arbitrary values.

An important distinction is that the number of sigma points includes the terminal `0.0`. Because the final point represents the end of the trajectory, the number of actual model forward passes is one less than the number of sigma values:

```text
4 sigma points = 4 steps = 3 forward passes
6 sigma points = 6 steps = 5 forward passes
8 sigma points = 8 steps = 7 forward passes
```

So when this README refers to a **4-step manual sigma schedule**, it means **4 sigma points including `0.0`**, resulting in **3 actual forward passes**.

Likewise:

* **6-step manual sigmas** = 6 sigma points including `0.0` = **5 forward passes**
* **8-step manual sigmas** = 8 sigma points including `0.0` = **7 forward passes**

### 4-step manual sigmas — 3 forward passes

Use the existing ComfyUI **ManualSigmas** node with:

```text
1.0, 0.8571428571428571, 0.6, 0.0
```

This is the fastest supplied schedule:

```text
4 sigma points
→ 3 intervals
→ 3 model forward passes
```

### 6-step manual sigmas — 5 forward passes

Use:

```text
1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0
```

This is the **recommended speed/quality sweet spot**:

```text
6 sigma points
→ 5 intervals
→ 5 model forward passes
```

### 8-step manual sigmas — 7 forward passes

Use:

```text
1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0
```

This gives the longest of the supplied manual schedules:

```text
8 sigma points
→ 7 intervals
→ 7 model forward passes
```

### Which should I use?

For most renders, start with **6 steps**.

```text
4 steps / 3 forwards → fastest
6 steps / 5 forwards → recommended speed/quality balance
8 steps / 7 forwards → favor quality/stability
```

If using the standard scheduler path, start with **6–8 steps** and try `simple`, `beta`, `normal`, or `bong_tangent`.

If using the manual sigma workflows, use the supplied **4-, 6-, or 8-point sigma schedules** above. These schedules are already included in the workflows.

## Limitations

* **Identity drift on re-entry**: when the subject leaves the camera view and re-enters, the re-entry settles toward the driving video's original appearance rather than the reference image. The same applies when the subject moves far from the reference pose or makes abrupt large motions (e.g. a backflip) — the further from the still, the weaker the identity hold.
* **Lip-sync limitations**: the generated subject does not reliably lip-sync to the conditioning video.
* **Reference image compatibility**: if you cannot get the reference-image character to appear correctly in the output video (identity drift), make the reference image match the pose/stance of the person in the conditioning video as closely as possible (same background). Keep the gen between **0.4–0.6 megapixels** and use **LCM or normal sampling with 6–8 steps**.

## Links

* Original model + inference code: [huggingface.co/Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate) · [viggle.ai](https://viggle.ai)
* Base model: [huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
* Comfy-repackaged base VAEs: [huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
* Converted ComfyUI weights, quants & LoRAs: [huggingface.co/drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)

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

* The **weights** are a Model Derivative of MiniMax H3 — the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3) applies to them (read it before redistributing or shipping a product on them). This includes the converted/quantized variants linked above.
* This **node pack** is Apache 2.0 (see `LICENSE`).
* The model puts a person into footage they did not shoot; identity comes from the image you supply. Do not run it on people who have not consented, and label what you generate as AI-generated (see the original repo's intended-use section).

## Report Issue

* Issues: [ComfyUI-Viggle-Animate-H3/issues](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3/issues)
