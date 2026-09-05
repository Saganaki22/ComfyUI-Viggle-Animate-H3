# ComfyUI-Viggle-Animate-H3

ComfyUI nodes for **Viggle-Animate** — the MiniMax-H3 ref2va full finetune that replaces
the performers in a driving video with the person from a single still image.

No text encoder, no prompt: conditioning comes from a frozen 362-token embedding computed
once by the Viggle team with Qwen3-VL. You supply the driving video (motion, framing,
background, lighting) and a reference still (identity). Nothing in the output comes from
text you write.

## Nodes

| Node | What it does |
|---|---|
| **Load Text Conditioning (Viggle)** | Dropdown loader for frozen text conditioning in `models/text_cond/` |
| **Viggle-Animate Conditioning (H3)** | Builds conditioning + AV latent: video-first reference order, both references nested on the driving clip's short edge (the layout the finetune was trained with) |

Everything else uses ComfyUI core nodes: `UNETLoader`, `LoraLoaderModelOnly`,
`MiniMaxH3SigmaShift` (shift 3), `KSampler` (4 steps), VAE/audio decode, `SaveVideo`.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

Model files (converted checkpoints, LoRAs, frozen text conditioning):
**[huggingface.co/drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI)**

| Repo folder | Goes to | Contents |
|---|---|---|
| `diffusion_models/` | `ComfyUI/models/diffusion_models/` | bf16 / int8_convrot / pruned_int8_convrot transformers |
| `loras/` | `ComfyUI/models/loras/` | DMD LoRA rank 128 (+ r64 / r29 SVD-truncated variants) |
| `text_cond/` | `ComfyUI/models/text_cond/` | `fixed_embed_fwd_anyframe.safetensors` |

Also required (from the base MiniMax-H3 release): the **video VAE** and **audio VAE**.

## Workflow recipe

```
UNETLoader (viggle pruned_int8_convrot) -> LoraLoaderModelOnly (dmd lora) -> MiniMaxH3SigmaShift (3.0)
LoadVideo (24 fps) --+                                          |
LoadImage (still) ---+--> Viggle-Animate Conditioning --+       v
VAELoader -----------+                                KSampler (4 steps)
Load Text Conditioning -----------------------------+
```

## Credits & license

- Original project: [Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)
- Base model: MiniMax-H3 — the weights are under the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- This node pack itself: MIT
