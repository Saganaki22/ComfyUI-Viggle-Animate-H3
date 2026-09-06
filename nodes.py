import collections
import hashlib
import logging
import math
import os
import weakref

import torch

import folder_paths
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from comfy_extras import nodes_minimax_h3 as core_h3

CANVAS_MULTIPLE = 32
FPS = 24
MIN_ASPECT, MAX_ASPECT = 1 / 4, 4

# Encoded reference latents are deterministic per (content, canvas, frames, VAE);
# caching them lets repeat runs skip the video-VAE encode (the multi-second stall).
# Cap is generous on purpose: a windowed run needs chunks+1 entries, and the total
# bytes self-limit — the windows sum to roughly one full clip's latent, which the
# workflow itself is already holding.
_LATENT_CACHE = collections.OrderedDict()
_CACHE_MAX = 64


def _fingerprint(t, extra):
    """Content key: shape + dtype + strided pixel sample + global checksum.

    The full-tensor sum makes any content change a cache miss; the strided
    byte sample pins down which arrangement produced it. ~0.3 s at 1.4 GB,
    versus the multi-second VAE encode it gates.
    """
    h = hashlib.sha1(repr((tuple(t.shape), str(t.dtype), extra)).encode())
    h.update(t.sum(dtype=torch.float64).item().hex().encode())
    fs = max(1, t.shape[0] // 8)   # <= 9 frames
    hs = max(1, t.shape[1] // 24)  # ~24x24 px per sampled frame
    ws = max(1, t.shape[2] // 24)
    s = t[::fs, ::hs, ::ws, :]
    h.update(s.detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def _cache_get(key, vae):
    ent = _LATENT_CACHE.get(key)
    if ent is None:
        return None
    ref, val = ent
    if ref() is not vae:           # stale id from a freed VAE
        return None
    _LATENT_CACHE.move_to_end(key)
    return val.clone()


def _cache_put(key, vae, val):
    _LATENT_CACHE[key] = (weakref.ref(vae), val.clone())
    _LATENT_CACHE.move_to_end(key)
    while len(_LATENT_CACHE) > _CACHE_MAX:
        _LATENT_CACHE.popitem(last=False)


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
            "cond_video": ("IMAGE", {"tooltip": "Driving video frames at 24 fps (Load Video node). Supplies motion, camera, background, lighting."}),
            "ref_image": ("IMAGE", {"tooltip": "Single still of the person to place in the video."}),
            "text_cond": ("TEXT_COND", {"tooltip": "From the Load Text Conditioning node."}),
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE (from the base model). Encodes the driving clip and the reference still."}),
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
                   "Pair with MiniMaxH3SigmaShift (shift 3) and 4-8 sampling steps.")

    def build(self, cond_video, ref_image, text_cond, vae, width, height, length):
        # ---- frozen text conditioning -------------------------------------
        prompt_embeds = text_cond["prompt_embeds"]      # [1, 362, 5120] bf16
        text_token_tags = text_cond["text_token_tags"]  # [362] int64

        # ---- geometry: clip dims by default; manual w/h sets the canvas ----
        # Reference parity (sample.py): short_edge = min(h, w) of the TARGET,
        # max_pixels = target area; both references lay out on that canvas.
        vh, vw = cond_video.shape[1], cond_video.shape[2]
        tgt_w, tgt_h = (width or vw), (height or vh)
        short_edge = min(tgt_w, tgt_h)
        max_pixels = short_edge * max(tgt_w, tgt_h)
        ch, cw = resolve_canvas(tgt_w, tgt_h, short_edge, max_pixels)

        frame_count, latent_t, audio_t = core_h3.temporal_shape(length)

        # ---- reference 1: the driving video (first in the presentation) ----
        rh, rw = resolve_canvas(vw, vh, short_edge, max_pixels)
        n = min(cond_video.shape[0], frame_count)
        if n < 5:
            raise ValueError("Viggle-Animate: driving clip needs at least 5 frames (~0.2 s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = cond_video[:n]  # truncate before resampling: dropped frames never see lanczos
        vkey = _fingerprint(frames, ("v", n, rw, rh, id(vae)))
        z_video = _cache_get(vkey, vae)
        if z_video is None:
            if (vh, vw) != (rh, rw):
                frames = core_h3._resize(frames, rw, rh, "disabled")
            z_video = vae.encode(frames)
            _cache_put(vkey, vae, z_video)
        video_block = {"kind": "video", "latent_t": z_video.shape[2],
                       "latent_h": rh // 16, "latent_w": rw // 16,
                       "ref_audio_t": 0, "latent": z_video, "audio_latent": None}

        # ---- reference 2: the still, nested at the clip's short edge -------
        ih, iw = ref_image.shape[1], ref_image.shape[2]
        scale = short_edge / min(iw, ih)  # upscaling included, no area cap (per the finetune)
        th = max(CANVAS_MULTIPLE, round(ih * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        tw = max(CANVAS_MULTIPLE, round(iw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ikey = _fingerprint(ref_image[:1], ("i", tw, th, id(vae)))
        z_img = _cache_get(ikey, vae)
        if z_img is None:
            img = ref_image[:1] if (ih, iw) == (th, tw) else core_h3._resize(ref_image[:1], tw, th, "disabled")
            z_img = vae.encode(img)
            _cache_put(ikey, vae, z_img)
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


class ViggleAnimateConditioningWindowed:
    """Windowed Viggle-Animate conditioning for MMH3Tools' Looping Sampler.

    Emits an MMH3_COND_SET: one conditioning entry per chunk, each carrying the
    driving clip's OWN span as its video reference (cut on the looping sampler's
    schedule, so chunk i is conditioned on the footage it renders) plus the still,
    broadcast unchanged to every chunk. The latent output is the whole clip.

    Requires ComfyUI-MMH3Tools. Wire chunk_frames / overlap_frames identically on
    both nodes (MMH3 Chunk Schedule feeds both), and do not wire the sampler's
    prior_av_latent — windows are cut from frame 0 of the driving clip.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_video": ("IMAGE", {"tooltip": "The WHOLE driving clip at 24 fps. Its (grid-snapped) length IS the output length — cut the tail off with Load Video's frame_load_cap if you want shorter."}),
            "ref_image": ("IMAGE", {"tooltip": "Single still of the person. Identity-only conditioning — pose comes from each chunk's own footage, so one still covers every chunk and need not match any frame's pose."}),
            "text_cond": ("TEXT_COND", {"tooltip": "From the Load Text Conditioning node."}),
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE (from the base model)."}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                              "tooltip": "Target width. 0 = driving clip's own width (the evaluated configuration)."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                               "tooltip": "Target height. 0 = driving clip's own height."}),
            "chunk_frames": ("INT", {"default": 192, "min": 0, "max": 3600, "step": 17,
                                     "tooltip": "New content per chunk, frames at 24 fps. 0 = one chunk over the whole clip. MUST equal the looping sampler's chunk_frames."}),
            "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 3600, "step": 17,
                                       "tooltip": "Frames each chunk carries from the previous one. MUST equal the looping sampler's overlap_frames."}),
        }}

    RETURN_TYPES = ("MMH3_COND_SET", "LATENT")
    RETURN_NAMES = ("cond_set", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/viggle"
    DESCRIPTION = ("Viggle-Animate conditioning, windowed: per-chunk driving-video references "
                   "as an MMH3 cond_set for the MiniMax H3 Looping Sampler (MMH3Tools).")

    def build(self, cond_video, ref_image, text_cond, vae, width, height,
              chunk_frames, overlap_frames):
        try:
            from mmh3tools.nodes_windows import _plan
            from mmh3tools.common import frame_at_latent
        except ImportError as e:
            raise ImportError(
                "Viggle-Animate windowed conditioning needs ComfyUI-MMH3Tools installed — "
                "it cuts the driving clip on the looping sampler's own schedule.") from e

        # ---- frozen text conditioning -------------------------------------
        prompt_embeds = text_cond["prompt_embeds"]      # [1, 362, 5120] bf16
        text_token_tags = text_cond["text_token_tags"]  # [362] int64

        # ---- geometry: same canvas rules as the single-pass node ----------
        vh, vw = cond_video.shape[1], cond_video.shape[2]
        tgt_w, tgt_h = (width or vw), (height or vh)
        short_edge = min(tgt_w, tgt_h)
        max_pixels = short_edge * max(tgt_w, tgt_h)
        ch, cw = resolve_canvas(tgt_w, tgt_h, short_edge, max_pixels)
        rh, rw = resolve_canvas(vw, vh, short_edge, max_pixels)

        # ---- total length = the clip itself, snapped DOWN to 17j+5 --------
        total_f = cond_video.shape[0]
        asked_f = total_f
        while total_f % 17 != 5 and total_f > 5:
            total_f -= 1
        if total_f < 5:
            raise ValueError("Viggle-Animate: driving clip needs at least 5 frames (~0.2 s at 24 fps)")
        if total_f != asked_f:
            logging.info("[ViggleAnimateConditioningWindowed] %d frames -> %d on the 17j+5 "
                         "grid, %d dropped from the tail", asked_f, total_f, asked_f - total_f)

        # ---- the looping sampler's own schedule (offset 0: no prior) ------
        cf = int(chunk_frames) if int(chunk_frames) > 0 else total_f
        _length, _overlap, _pf, _pt, windows = _plan(total_f, cf, int(overlap_frames),
                                                     "standard_static")
        spans = [(min(frame_at_latent(w.index_list[0]), total_f - 1),
                  min(frame_at_latent(w.index_list[-1] + 1) - 1, total_f - 1))
                 for w in windows]

        # ---- reference 2 first: the still is encoded ONCE for all chunks --
        ih, iw = ref_image.shape[1], ref_image.shape[2]
        scale = short_edge / min(iw, ih)  # upscaling included, no area cap (per the finetune)
        th = max(CANVAS_MULTIPLE, round(ih * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        tw = max(CANVAS_MULTIPLE, round(iw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ikey = _fingerprint(ref_image[:1], ("i", tw, th, id(vae)))
        z_img = _cache_get(ikey, vae)
        if z_img is None:
            img = ref_image[:1] if (ih, iw) == (th, tw) else core_h3._resize(ref_image[:1], tw, th, "disabled")
            z_img = vae.encode(img)
            _cache_put(ikey, vae, z_img)

        # ---- one cond entry per chunk, each with its own video window -----
        conds, prompts = [], []
        for i, (a, b) in enumerate(spans):
            n = b - a + 1
            while n % 17 != 5:  # grid-valid by construction except the clamped tail window
                n -= 1
            if n < 5:
                raise ValueError(f"Viggle-Animate: chunk {i}'s window (frames {a}-{b}) is "
                                 "under 5 frames after grid snapping — raise chunk_frames or "
                                 "lower overlap_frames.")
            frames = cond_video[a:a + n]
            vkey = _fingerprint(frames, ("v", n, rw, rh, id(vae)))
            z_video = _cache_get(vkey, vae)
            if z_video is None:
                if (vh, vw) != (rh, rw):
                    frames = core_h3._resize(frames, rw, rh, "disabled")
                z_video = vae.encode(frames)
                _cache_put(vkey, vae, z_video)
            video_block = {"kind": "video", "latent_t": z_video.shape[2],
                           "latent_h": rh // 16, "latent_w": rw // 16,
                           "ref_audio_t": 0, "latent": z_video, "audio_latent": None}
            image_block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                           "latent": z_img.clone()}
            conds.append([[prompt_embeds, {"minimax_refs": [video_block, image_block],
                                           "minimax_token_tags": text_token_tags}]])
            prompts.append(f"chunk {i}: frames {a}-{a + n - 1}")
        logging.info("[ViggleAnimateConditioningWindowed] %d chunks over %d frames (%.2fs): %s",
                     len(conds), total_f, total_f / FPS,
                     ", ".join(f"{a}-{b}" for a, b in spans))

        # ---- the whole clip's latent, written back chunk by chunk ---------
        latent_t = core_h3.video_latent_t(total_f)
        audio_t = round(total_f / core_h3.FPS * core_h3.AUDIO_LATENT_FPS)
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros([1, 24, latent_t, ch // 16, cw // 16],
                        device=comfy.model_management.intermediate_device()),
            torch.zeros([1, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device()),
        ))}
        return ({"conds": conds, "prompts": prompts}, latent)


NODE_CLASS_MAPPINGS = {"ViggleTextCondLoader": ViggleTextCondLoader,
                       "ViggleAnimateConditioning": ViggleAnimateConditioning,
                       "ViggleAnimateConditioningWindowed": ViggleAnimateConditioningWindowed}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleTextCondLoader": "Load Text Conditioning (Viggle)",
                              "ViggleAnimateConditioning": "Viggle-Animate Conditioning (H3)",
                              "ViggleAnimateConditioningWindowed": "Viggle-Animate Conditioning (H3, Windowed)"}