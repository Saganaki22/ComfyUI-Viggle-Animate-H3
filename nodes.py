import math
import os

import torch

import folder_paths
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from comfy_extras import nodes_minimax_h3 as core_h3

CANVAS_MULTIPLE = 32
FPS = 24
MIN_ASPECT, MAX_ASPECT = 1 / 4, 4


def resolve_canvas(aspect_w, aspect_h, short_edge, max_pixels):
    """diffusers resolve_canvas_size: short-edge aim, area cap, round to 32."""
    ratio = aspect_w / aspect_h
    if not MIN_ASPECT <= ratio <= MAX_ASPECT:
        raise ValueError(f"Viggle-Animate: aspect ratio {aspect_w}:{aspect_h} outside 1:4..4:1")
    if ratio >= 1.0:
        w, h = short_edge * ratio, float(short_edge)
    else:
        w, h = float(short_edge), short_edge / ratio
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        w, h = w * s, h * s
    return (max(CANVAS_MULTIPLE, round(h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


class ViggleTextCondLoader:
    """Loads a frozen text-conditioning safetensors from models/text_cond/.

    Viggle-Animate ships one: fixed_embed_fwd_anyframe (362 tokens computed once
    with Qwen3-VL from the fixed prompt, so the text encoder is never needed).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text_cond": (folder_paths.get_filename_list("text_cond"),
                          {"tooltip": "Frozen text conditioning in models/text_cond/ (fixed_embed_fwd_anyframe)."}),
        }}

    RETURN_TYPES = ("TEXT_COND",)
    RETURN_NAMES = ("text_cond",)
    FUNCTION = "load"
    CATEGORY = "loaders/viggle"
    DESCRIPTION = "Load frozen text conditioning (replaces the text encoder entirely)."

    def load(self, text_cond):
        from safetensors.torch import load_file
        path = folder_paths.get_full_path_or_raise("text_cond", text_cond)
        blob = load_file(path)
        return ({"prompt_embeds": blob["prompt_embeds"],
                 "text_token_tags": blob["text_token_tags"]},)


class ViggleAnimateConditioning:
    """Viggle-Animate (MiniMax-H3 ref2va finetune) conditioning.

    Frozen 362-token text embedding replaces the text encoder entirely; the
    driving video supplies motion/framing/background, the still supplies identity.
    References are packed video-first, both nested on the driving clip's short
    edge, matching how the finetune was trained and evaluated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE (from the base model). Encodes the driving clip and the reference still."}),
            "cond_video": ("IMAGE", {"tooltip": "Driving video frames at 24 fps (Load Video node). Supplies motion, camera, background, lighting."}),
            "ref_image": ("IMAGE", {"tooltip": "Single still of the person to place in the video."}),
            "text_cond": ("TEXT_COND", {"tooltip": "From the Load Text Conditioning node."}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                              "tooltip": "Target width. 0 = driving clip's own width (the evaluated configuration)."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                               "tooltip": "Target height. 0 = driving clip's own height."}),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "Frames at 24 fps, snapped to the 17k+5 grid (124 = ~5.2 s)."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/viggle"
    DESCRIPTION = ("Viggle-Animate conditioning: frozen text embed + video-first nested references. "
                   "Pair with MiniMaxH3SigmaShift (shift 3) and 4 sampling steps.")

    def build(self, vae, cond_video, ref_image, text_cond, width, height, length):
        # ---- frozen text conditioning -------------------------------------
        prompt_embeds = text_cond["prompt_embeds"]      # [1, 362, 5120] bf16
        text_token_tags = text_cond["text_token_tags"]  # [362] int64

        # ---- geometry: driving clip rules everything -----------------------
        vh, vw = cond_video.shape[1], cond_video.shape[2]
        short_edge = min(vh, vw)
        max_pixels = short_edge * max(vh, vw)
        tgt_w, tgt_h = (width or vw), (height or vh)
        ch, cw = resolve_canvas(tgt_w, tgt_h, short_edge, max_pixels)

        frame_count, latent_t, audio_t = core_h3.temporal_shape(length)

        # ---- reference 1: the driving video (first in the presentation) ----
        rh, rw = resolve_canvas(vw, vh, short_edge, max_pixels)
        frames = cond_video
        if (vh, vw) != (rh, rw):
            frames = core_h3._resize(frames, rw, rh, "disabled")
        n = min(frames.shape[0], frame_count)
        if n < 5:
            raise ValueError("Viggle-Animate: driving clip needs at least 5 frames (~0.2 s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z_video = vae.encode(frames)
        video_block = {"kind": "video", "latent_t": z_video.shape[2],
                       "latent_h": rh // 16, "latent_w": rw // 16,
                       "ref_audio_t": 0, "latent": z_video, "audio_latent": None}

        # ---- reference 2: the still, nested at the clip's short edge -------
        ih, iw = ref_image.shape[1], ref_image.shape[2]
        scale = short_edge / min(iw, ih)  # upscaling included, no area cap (per the finetune)
        th = max(CANVAS_MULTIPLE, round(ih * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        tw = max(CANVAS_MULTIPLE, round(iw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        img = ref_image[:1] if (ih, iw) == (th, tw) else core_h3._resize(ref_image[:1], tw, th, "disabled")
        z_img = vae.encode(img)
        image_block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z_img}

        # ---- assemble: video first, then picture (the frozen order) --------
        cond = [[prompt_embeds, {"minimax_refs": [video_block, image_block],
                                 "minimax_token_tags": text_token_tags}]]

        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros([1, 24, latent_t, ch // 16, cw // 16],
                        device=comfy.model_management.intermediate_device()),
            torch.zeros([1, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device()),
        ))}
        return (cond, latent)


NODE_CLASS_MAPPINGS = {"ViggleTextCondLoader": ViggleTextCondLoader,
                       "ViggleAnimateConditioning": ViggleAnimateConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleTextCondLoader": "Load Text Conditioning (Viggle)",
                              "ViggleAnimateConditioning": "Viggle-Animate Conditioning (H3)"}
