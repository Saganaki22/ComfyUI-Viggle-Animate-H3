# ComfyUI-Viggle-Animate-H3

**[English](README.md) | [中文](README_zh.md)**

<img width="731" height="468" alt="Screenshot 2026-09-05 214633" src="https://github.com/user-attachments/assets/1b65c73d-555e-4097-a2de-7ae2f5e6851f" />

<br>

ComfyUI nodes for **[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** — a 33.1 B
full finetune of MiniMax-H3's `ref2va` transformer for **character replacement in video**: it
takes a driving video and a reference still, and re-renders the performer(s) in the clip as the
character in the still. Motion, camera, timing, background and lighting come from the video;
identity comes from the image.

No text encoder, no prompt: conditioning is one frozen 362-token embedding computed once by the
Viggle team with Qwen3-VL (`assets/fixed_prompt.txt`), identical for every render. The sampler is
DMD2-distilled — **4 steps (3 forward passes)**, so a 124-frame shot is fast even on consumer
hardware.

## New in 1.4.0

Added experimental windowed conditioning and the **Viggle Chunked Sampler** for longer clips, with latent carry, chunk reuse and seed overrides for another take. Added custom sigma presets covering upstream-style **4–8 steps** (4, 6 or 8 sigma points), derived from the upstream shift-3 schedule. Choose the preset that fits your use case and speed budget; the longer schedules are experimental and do not guarantee better quality.

## Nodes

| Node | What it does |
|---|---|
| **Load Text Conditioning (Viggle)** | Dropdown loader for frozen text conditioning in `models/text_cond/` |
| **Viggle-Animate Conditioning (H3)** | Builds conditioning + AV latent: video-first reference order, both references nested on the canvas short edge (the driving clip's, unless width/height are overridden) — the layout the finetune was trained with |
| **Viggle-Animate Conditioning (H3, Windowed)** | Splits the driving clip into overlapping windows and builds each chunk's references; outputs `cond_set` for the chunked sampler and `guider_positive` for the guider |
| **Viggle Chunked Sampler** | Samples each window, preserves overlap from the preceding chunk, reuses eligible cached chunks, and decodes the assembled video; outputs `frames` and a readable `chunk_map` |

Model loading and sampling controls use ComfyUI core: **Load Diffusion Model**, **Load LoRA (Model Only)**,
**ModelSamplingMiniMaxH3** (video/audio shifts 3.0), **BasicGuider**, **KSamplerSelect** and **ManualSigmas**.
KJNodes **CustomSigmas** can supply the same schedules below. The chunked sampler decodes internally; connect its `frames` directly to your video-saving node.

| cond_vid | ref_img | output |
|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/2529857c-2667-4641-9d2e-5dcb3c03913d" controls muted></video> | <img src="https://github.com/user-attachments/assets/f6adf969-03d5-4a58-bd30-5ec2d0bc604b" width="300"> | <video src="https://github.com/user-attachments/assets/deedde68-80de-47d2-9e1f-7eaa3bc35457" controls muted></video> |



| example_1 | example_2 | example_3 | example_4 |
|:---:|:---:|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/c5198b4c-9544-4e5a-bd1e-83ae4477bb0b" controls muted></video> | <video src="https://github.com/user-attachments/assets/afb74de1-3d3d-42ae-9885-1d5b1c6e86af" controls muted></video> | <video src="https://github.com/user-attachments/assets/5ddf1bb1-e744-406b-8b9f-2b729e4ecc4b" controls muted></video> | <video src="https://github.com/user-attachments/assets/8d02389d-67ca-46f0-b9e3-78994d944a90" controls muted></video> |


| example_5 | example_6 |
|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/d4533465-705a-4487-8f14-04770c3d84b6" controls muted></video> | <video src="https://github.com/user-attachments/assets/e519cb64-18c3-4fa7-abe3-d5e57fc0b72e" controls muted></video> |


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

- [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors) (3.17 GB, low VRAM)
- or [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) (5.21 GB)

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
- **Load Video**: use `force_rate = 24`. For single-shot conditioning, set `frame_load_cap` equal to `length` (e.g. 124). For windowed conditioning, load the desired full clip (`frame_load_cap = 0` in VHS loads all frames).
- Output resolution follows the driving video by default; set the conditioning node's `width`/`height` to override (each axis rounds to 32), or pre-scale the clip with **Scale Image to Total Pixels**. Tested canvas range: **0.4–0.98 MP**.
- **Upstream sampling baseline:** use **ManualSigmas** with `1.0, 0.8571428571428571, 0.6, 0.0`, **euler**, **BasicGuider** (or CFG 1.0), and the original `viggle_animate_dmd_lora.safetensors` at strength 1.0. Connect ManualSigmas to SamplerCustomAdvanced's or Viggle Chunked Sampler's `sigmas` input. These are four sigma points and **three model evaluations**, matching upstream's “4 steps.” Keep **ModelSamplingMiniMaxH3** at video **3.0**, audio **3.0**; it does not shift the supplied ManualSigmas tensor again. Do not add a separate sigma-transform node afterward.
- Other samplers/schedulers and the rank-64 adapter are experimental alternatives. The bundled example's LCM/bong_tangent eight-step settings differ from the upstream baseline.
- **KJNodes CustomSigmas:** for the four values above, set `interpolate_to_steps` to **3**. Setting it to 4 interpolates through `log(0)` and produces a schedule ending in `0, 0`; Euler returns NaNs, which turn the final video black and contaminate subsequent chunks. The chunked sampler rejects this invalid schedule before rendering.
- Stacks with **Comfy Kitchen** and **block sparse attention** patches.

## Custom sigma presets (4–8 upstream-style steps)

Paste one list into **ManualSigmas** or KJNodes **CustomSigmas**, then connect its `SIGMAS` output to the sampler. These presets use `sigma = 3*t / (1 + 2*t)` with evenly spaced `t` from 1 to 0. The four-point preset matches the upstream baseline; the six- and eight-point presets extend the same pattern.

| Suggested workflow title | Sigma points, including final zero | Sampling updates / KJNodes `interpolate_to_steps` |
|---|---|---|
| **Viggle DMD — 3 Steps (Upstream “4-Step”)** | 4 | **3** |
| **Viggle — 5 Steps (6 Sigma Points)** | 6 | **5** |
| **Viggle — 7 Steps (8 Sigma Points)** | 8 | **7** |

**4 points — fastest baseline:**

```text
1.0, 0.8571428571428571, 0.6, 0.0
```

**6 points — intermediate sampling cost:**

```text
1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0
```

**8 points — more sampling updates:**

```text
1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0
```

Keep **Euler**, **BasicGuider / CFG 1.0**, and model shifts **3.0 / 3.0**. The final `0.0` is required: it is the destination of the last update, not another model evaluation. Do not append another zero or shift these lists again. Use fewer updates for speed; compare the longer presets on your footage before choosing them for quality. Encoding and final decoding still take time regardless of the preset.

## Experimental long clips (`exp`)

Example workflow (Windowed Conditioning + Chunked Sampler): [example_workflows/viggle-animate-h3_workflow_chunked_sampler_exp.json](example_workflows/viggle-animate-h3_workflow_chunked_sampler_exp.json).

Connect **Viggle-Animate Conditioning (H3, Windowed)** to **Viggle Chunked Sampler**. Its `guider_positive` output supplies BasicGuider's conditioning (or CFGGuider's positive). Start with 124-frame chunks and 22-frame overlap. Prior output is preserved in each overlap, and the assembled latent is decoded once. Motion and appearance can still change at joins; use a repainted reference frame from the driving shot and keep the input/output at 24 fps.

Windowed conditioning reuses complete 17-frame encoder blocks from the preceding window when using the standard ComfyUI H3 VAE. Each window's padded tail is still encoded separately. This reduces repeated VAE work without reducing resolution, changing precision or enlarging the encoding window; custom VAE wrappers retain the full-window path. The log reports how many blocks were reused. Higher resolution and longer clips still cost more to encode.

### Chunk seed controls

| Control | Meaning |
|---|---|
| `seed` | Base sampling seed: chunk 1 uses `seed`, chunk 2 uses `seed + 1`, and so on. The chunked sampler uses this value instead of the connected noise node's seed. |
| `rerender_chunk` | **1-based** chunk whose seed you want to override. **0 disables the override**. Choose a chunk number shown in `chunk_map`. |
| `rerender_seed` | Replacement seed for the selected chunk only. It has no effect when `rerender_chunk = 0`; zero itself is a valid seed. |

These controls let you try another take from a troublesome chunk without changing the base seed for the whole clip. For example, with base `seed = 58`, four chunks normally use `58, 59, 60, 61`. Set `rerender_chunk = 2` and `rerender_seed = 123` to use `58, 123, 60, 61`. Chunk 1 can be reused; chunks 2–4 must be regenerated when that override changes, because each receives content from its predecessor. Their original numeric seeds do not make later chunks independent of the changed carry.

Keep the same override to retain that take, or change `rerender_seed` for another. This is a seed override, not a force-refresh button: unchanged settings can reuse cached results. Setting `rerender_chunk` back to 0 restores the base-seed sequence. The `chunk_map` output shows frame ranges, seeds, overlap and `[cached]` / `[rendered]` labels.

### Chaining and rerender limitations

- You cannot change one chunk and keep all later chunks fixed: the overlap is carried forward. Existing overlap from the preceding chunk stays pinned, so rerendering a chunk does not repaint its inherited beginning.
- Reuse is an in-memory optimization, not a saved checkpoint or resume system. Restarting ComfyUI clears it. Cache eviction, changed inputs/model/sampling settings, or oversized entries can require earlier chunks to render again.
- Stock noise and guider objects with inspectable sampler/model settings support reuse. Opaque custom options, callbacks or patches bypass caching and still sample normally; patched workflows may rerender every chunk.
- Chunk caching preserves output precision and is limited to **2 GiB** of CPU tensor storage. Encoded references have a separate **256 MiB / 64-entry** limit. Oversized entries are not cached.
- The assembled latent is decoded again after sampling, even when earlier chunks are reused. Full-clip conditioning, the master latent, final decoding and output frames still need memory; chunking does not make arbitrarily long clips fit in RAM/VRAM.
- Motion, identity and lighting can still change at joins; overlap does not guarantee seamless or stutter-free video. The final window may overlap more than requested, and up to **16 trailing frames** are dropped to fit the `17k+5` frame grid.
- Generated audio is discarded. Connect the driving clip's audio to the video-saving node and match it to the retained video length; use **24 fps** for input and output.
- Invalid sigma schedules and NaN/Inf chunk latents now stop with an actionable error before corrupt output is cached or carried into later chunks.

### Chunk loop nodes (experimental)

Full node-by-node breakdown (sockets, slot order, resume rules, typical session): [docs/experimental_nodes.md](docs/experimental_nodes.md). Tested example workflow: [example_workflows/viggle-animate-h3_workflow_chunked_window_exp.json](example_workflows/viggle-animate-h3_workflow_chunked_window_exp.json).

Four nodes turn the same windowed conditioning into a **graph-expanded loop** with disk checkpoints, so each chunk is decoded and saved through your own nodes while it is produced — no VAE input on the sampler, and a failure mid-run keeps every completed chunk:

| Node | Purpose |
|---|---|
| **Viggle Chunk Loop Start** | Reads the plan from `cond_set`, picks the checkpoint directory, initializes the loop state. |
| **Viggle Sample Chunk** | Samples the current window only. Outputs the chunk's video LATENT for an ordinary VAE Decode, plus the carried state. Checkpoints the latent to disk **before** returning. |
| **Viggle Chunk Loop End** | Waits for this iteration's decode/save branch, then either expands the next chunk or returns the finished collection. |
| **Viggle Assemble Chunk Latents** | Stitches the saved chunks (trimming overlap) into one LATENT for a single final decode. `chunk_number > 0` loads one chunk for inspection; interrupted runs assemble partially. |

```text
Loop Start ─ loop ───────────────────────────────┐
     └ state → Sample Chunk → LATENT → VAE Decode ─┬→ Loop End (images)
                                                   └→ Video Combine → filenames ↗ (after_save)
```

- **The decode/save branch must feed Loop End** — connect VAE Decode's images to `images` and Video Combine's `filenames` output to `after_save`, so a chunk cannot start before the previous one is decoded and saved. (Core SaveWEBM also works: its `images` output goes straight into `images`.)
- Checkpoints land in `output/viggle_chunks/<run_name>/` as safetensors plus a `manifest.json`; writes are atomic, so a crash never leaves a half-valid chunk.
- With `resume` on, re-running the queue restores every chunk whose **graph, models, conditioning, sigmas and per-chunk seed** still match (the checkpoint filename embeds that fingerprint). Changed settings sample new files under new names; old takes stay on disk. `rerender_chunk` / `rerender_seed` work as in the single-pass sampler.
- Decoding each chunk separately means each preview contains overlap context; run the collection through **Viggle Assemble Chunk Latents** → one final VAE Decode for the finished video.
- A run killed mid-way leaves chunks 1…k−1 on disk and the manifest valid: re-queue with the same `run_name` and only the missing chunks sample.

## Limitations

- **Identity drift on re-entry**: when the subject leaves the camera view and re-enters, the re-entry settles toward the driving video's original appearance rather than the reference image. The same applies when the subject moves far from the reference pose or makes abrupt large motions (e.g. a backflip) — the further from the still, the weaker the identity hold.
- **Lip-sync limitations**: the generated subject does not reliably lip-sync to the conditioning video.
- **Reference image compatibility**: if you cannot get the reference-image character to appear correctly in the output video (identity drift), make the reference image match the pose/stance of the person in the conditioning video as closely as possible (same background). Keep the gen between **0.4–0.6 megapixels** and use **LCM or normal sampling with 6–8 steps**.

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

- issues: [ComfyUI-Viggle-Animate-H3/issues](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3/issues)
