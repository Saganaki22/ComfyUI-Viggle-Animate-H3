import collections
import copy
import hashlib
import logging
import math
import os
import weakref

import torch

import folder_paths
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
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


# ---------------------------------------------------------------------------
# windowed (long-clip) tooling: schedule, conditioning, chunked sampler
# ---------------------------------------------------------------------------

def _frame_at_latent(k):
    """First pixel frame covered by latent step k (FRAME_PER_TOKEN = 1,4,4,4,4)."""
    return 17 * (k // 5) + (0, 1, 5, 9, 13)[k % 5]


def _frames_to_latents(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def plan_spans(total_f, chunk_f, overlap_f):
    """Static window schedule in frames.

    Full 17j+5 windows laid at a stride that keeps every start on latent phase
    0; the LAST window starts flush at total-chunk, so every render runs at the
    trained chunk length and the slack becomes extra overlap with the window
    before it. Returns [(first_frame, last_frame), ...] covering 0..total_f-1.
    """
    L = _frames_to_latents(int(chunk_f))
    total_lat = _frames_to_latents(int(total_f))
    if L < 7:
        raise ValueError("Viggle-Animate: chunk_frames must be at least 22 frames (124 recommended).")
    if L >= total_lat:
        return [(0, int(total_f) - 1)]
    O = _frames_to_latents(max(5, int(overlap_f)))
    O = max(2, min(O, L - 5))          # stride stays a multiple of 5 (phase 0)
    stride = L - O
    starts = list(range(0, total_lat - L + 1, stride))
    last = total_lat - L
    if starts[-1] != last:
        starts.append(last)
    return [(_frame_at_latent(s), _frame_at_latent(s + L) - 1) for s in starts]


def _stitch(frames_list, spans, crossfade):
    """Blend chunk renders across their footage overlaps into one clip.

    Both renders of an overlap frame tracked the same driving footage, so the
    blend morphs texture, not motion. Ramp weights are complementary, so every
    blended frame is an exact convex combination of the two renders.
    """
    if len(frames_list) == 1:
        return frames_list[0].float()
    if not int(crossfade):  # hard cut: each chunk owns its frames up to the next start
        parts = [frames_list[i][:spans[i + 1][0] - spans[i][0]]
                 for i in range(len(frames_list) - 1)]
        parts.append(frames_list[-1])
        return torch.cat(parts)
    total_f = spans[-1][1] + 1
    out = torch.zeros(total_f, *frames_list[0].shape[1:], dtype=torch.float32)
    wsum = torch.zeros(total_f, 1, 1, 1, dtype=torch.float32)
    for i, (fr, (a, b)) in enumerate(zip(frames_list, spans)):
        w = torch.ones(fr.shape[0], dtype=torch.float32)
        if i > 0:  # fade in over this window's first frames
            c = min(int(crossfade), fr.shape[0], spans[i - 1][1] - a + 1)
            if c > 0:
                w[:c] = torch.linspace(0.0, 1.0, c + 2)[1:-1]
        if i < len(frames_list) - 1:  # ownership ends where the NEXT window begins:
            # zero past the blend window first, then ramp down across it — the
            # ramps of both chunks occupy the same global frames even when the
            # overlap is wider than the crossfade, so their weights sum to 1
            a_next = spans[i + 1][0]
            local = a_next - a
            w[local:] = 0.0
            c = min(int(crossfade), b - a_next + 1, fr.shape[0] - local)
            if c > 0:
                w[local:local + c] = torch.linspace(1.0, 0.0, c + 2)[1:-1]
        w = w.view(-1, 1, 1, 1)
        out[a:b + 1] += fr.float() * w
        wsum[a:b + 1] += w
    return out / wsum


# Rendered-chunk cache: chunk frames are a pure function of (footage window,
# still, chunk seed, schedule, sampler, cfg, model). Re-queueing with only
# rerender_chunk/rerender_seed changed re-renders exactly that one chunk.
_CHUNK_CACHE = collections.OrderedDict()
_CHUNK_CACHE_MAX_BYTES = 2 * 1024 ** 3


def _sampler_fp(sampler):
    fn = getattr(sampler, "sampler_function", None)
    if fn is not None:
        params = getattr(sampler, "sampler_params", {}) or {}
        return repr((getattr(fn, "__name__", str(fn)),
                     sorted((k, repr(v)) for k, v in params.items())))
    inner = getattr(sampler, "sampler_object", None)
    return repr((type(sampler).__name__, type(inner).__name__ if inner is not None else None))


def _chunk_cache_get(key, model):
    ent = _CHUNK_CACHE.get(key)
    if ent is None:
        return None
    ref, val = ent
    if ref() is not model:         # model/LoRA/patch changed -> stale id
        return None
    _CHUNK_CACHE.move_to_end(key)
    return val.clone()


def _chunk_cache_put(key, model, frames_u8):
    _CHUNK_CACHE[key] = (weakref.ref(model), frames_u8.clone())
    _CHUNK_CACHE.move_to_end(key)
    total = sum(v[1].numel() for v in _CHUNK_CACHE.values())
    while total > _CHUNK_CACHE_MAX_BYTES and len(_CHUNK_CACHE) > 1:
        _, (_, old) = _CHUNK_CACHE.popitem(last=False)
        total -= old.numel()


def _raw_conds(guider):
    """The guider's (positive, negative) in the form set_conds accepts.

    negative is None for a BasicGuider, whose set_conds takes ONE argument.
    """
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    conds = getattr(guider, "original_conds", None) or {}
    return (conds.get("positive"), conds.get("negative"))


def _chunk_guider(guider, positive):
    """The wired guider with its POSITIVE replaced by this chunk's conditioning."""
    new_g = copy.copy(guider)
    # SHALLOW copy shares original_conds; set_conds assigns into it. Rebind
    # before touching it or chunk 0 clobbers the base conditioning.
    new_g.original_conds = dict(getattr(guider, "original_conds", None) or {})
    _, negative = _raw_conds(guider)
    if negative is None:
        new_g.set_conds(positive)
    else:
        new_g.set_conds(positive, negative)
    new_g.raw_conds = (positive, negative)
    return new_g


class ViggleAnimateConditioningWindowed:
    """Windowed Viggle-Animate conditioning for long driving clips.

    Splits the driving clip into overlapping 17j+5 windows (default 124 frames —
    the finetune's evaluated chunk length) and emits one conditioning entry per
    window: the window's OWN footage as the video reference, plus the still,
    broadcast unchanged to every chunk. Pair with Viggle Chunked Sampler, which
    renders each chunk independently and crossfades the overlaps in pixel space.

    One still covers every chunk — identity-only conditioning; pose always comes
    from the footage.
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
            "chunk_frames": ("INT", {"default": 124, "min": 39, "max": 3600, "step": 17,
                                     "tooltip": "Render length per chunk, frames at 24 fps. 124 is the finetune's evaluated operating point — every chunk runs at exactly this length."}),
            "overlap_frames": ("INT", {"default": 22, "min": 5, "max": 3600, "step": 17,
                                       "tooltip": "Frames shared by consecutive windows. Both renders of the overlap are blended, so this is also the default crossfade length."}),
        }}

    RETURN_TYPES = ("VIGGLE_COND_SET",)
    RETURN_NAMES = ("cond_set",)
    FUNCTION = "build"
    CATEGORY = "conditioning/viggle"
    DESCRIPTION = ("Viggle-Animate conditioning, windowed: per-chunk driving-video references "
                   "for the Viggle Chunked Sampler. Long clips in, one still, no flicker.")

    def build(self, cond_video, ref_image, text_cond, vae, width, height,
              chunk_frames, overlap_frames):
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

        spans = plan_spans(total_f, chunk_frames, overlap_frames)

        # ---- the still is encoded ONCE for all chunks ----------------------
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

        # ---- one cond entry per chunk, each with its own footage window ----
        conds, prompts = [], []
        for i, (a, b) in enumerate(spans):
            n = b - a + 1
            while n % 17 != 5:  # grid-valid by construction; kept as a safety net
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
            prompts.append(f"chunk {i + 1}: frames {a}-{a + n - 1}")
        logging.info("[ViggleAnimateConditioningWindowed] %d chunks over %d frames (%.2fs), "
                     "rerender_chunk is 1-based: %s",
                     len(conds), total_f, total_f / FPS,
                     ", ".join(f"{a}-{b}" for a, b in spans))

        return ({"conds": conds, "prompts": prompts, "spans": spans,
                 "total_frames": total_f, "canvas": (ch, cw)},)


class ViggleChunkedSampler:
    """Render a windowed Viggle clip chunk by chunk, crossfaded in pixel space.

    Each chunk is an independent, full-length render — the exact configuration
    the finetune was evaluated at — so nothing is out-of-distribution and there
    is no latent carry to drift. Consecutive chunks share driving footage over
    the overlap; both renders of those frames track the same motion, so the
    pixel-space crossfade blends texture only: no pop, no flicker.

    Chunks are cached on (footage, still, chunk seed, schedule, sampler, cfg,
    model): re-queueing with rerender_chunk/rerender_seed set re-renders exactly
    one chunk and re-blends from cache.

    Takes the same NOISE / GUIDER / SAMPLER / SIGMAS objects as Sampler Custom
    Advanced — the guider's positive is swapped per chunk from the cond_set.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE", {"tooltip": "From a Noise node (e.g. RandomNoise). Its seed is offset per chunk: chunk i renders with seed + i."}),
            "guider": ("GUIDER", {"tooltip": "From BasicGuider / CFGGuider. Only its model, cfg and negative matter — the positive is replaced per chunk from the cond_set."}),
            "sampler": ("SAMPLER", {"tooltip": "From KSamplerSelect or RES4LYF — reused for every chunk."}),
            "sigmas": ("SIGMAS", {"tooltip": "The step schedule (BasicScheduler etc.) — every chunk runs the identical schedule."}),
            "cond_set": ("VIGGLE_COND_SET", {"tooltip": "From Viggle-Animate Conditioning (H3, Windowed)."}),
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE. Decodes each chunk; the model's (silent) audio half is discarded — keep your driving clip's own audio at save time."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "tooltip": "Base seed. Chunk i renders with seed + i, so the chunks vary independently while staying reproducible."}),
            "rerender_chunk": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                       "tooltip": "1-based chunk number to re-render (0 = off). Only that chunk's cache key changes — everything else serves from cache."}),
            "rerender_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                      "tooltip": "Seed for the chunk selected by rerender_chunk. Type a new number for a new take."}),
            "crossfade_frames": ("INT", {"default": 22, "min": 0, "max": 360, "step": 1,
                                         "tooltip": "Blend length across each overlap. Both sides tracked the same footage, so this morphs texture, not motion. Clamped to the overlap."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "chunk_map")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle"
    DESCRIPTION = ("Chunked Viggle-Animate sampler: independent full-length renders, "
                   "crossfaded joins, per-chunk cache and re-render.")

    def sample(self, noise, guider, sampler, sigmas, cond_set, vae, seed,
               rerender_chunk, rerender_seed, crossfade_frames):
        spans = cond_set["spans"]
        conds = cond_set["conds"]
        if len(spans) != len(conds) or not conds:
            raise ValueError("Viggle Chunked Sampler: cond_set spans/conds mismatch.")
        total_f = int(cond_set["total_frames"])
        ch, cw = cond_set["canvas"]
        model = guider.model_patcher

        pbar = comfy.utils.ProgressBar(len(spans))
        frames_list = []
        chunk_map = ["%d chunks, %d frames (%.1fs) at %dx%d — rerender_chunk is 1-based:"
                     % (len(spans), total_f, total_f / FPS, ch, cw)]
        for i, (a, b) in enumerate(spans):
            n = b - a + 1
            seed_i = int(rerender_seed) if int(rerender_chunk) == i + 1 else int(seed) + i
            chunk_map.append("#%d: frames %d-%d (%.1f-%.1fs) seed %d"
                             % (i + 1, a, b, a / FPS, (b + 1) / FPS, seed_i))
            key = self._chunk_key(conds[i], seed_i, ch, cw, n, sigmas, sampler, guider)
            fr = _chunk_cache_get(key, model)
            if fr is None:
                out = self._render_chunk(noise, guider, sampler, sigmas, conds[i],
                                         seed_i, ch, cw, n)
                video = out["samples"]
                if video.is_nested:
                    video = video.unbind()[0]  # the model's audio half is silent; drop it
                images = vae.decode(video)
                if images.dim() == 5:  # combine batches
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                fr = images.clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu()
                _chunk_cache_put(key, model, fr)
            frames_list.append(fr)
            pbar.update(1)

        stitched = _stitch(frames_list, spans, int(crossfade_frames))
        return (stitched / 255.0, "\n".join(chunk_map))

    def _render_chunk(self, noise, guider, sampler, sigmas, cond, seed_i, ch, cw, n):
        lat_t = _frames_to_latents(n)
        samples = comfy.nested_tensor.NestedTensor((
            torch.zeros([1, 24, lat_t, ch // 16, cw // 16],
                        device=comfy.model_management.intermediate_device()),
            torch.zeros([1, 32, 2, round(n / FPS * 40)],
                        device=comfy.model_management.intermediate_device()),
        ))
        samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, samples)
        chunk_latent = {"samples": samples}
        chunk_noise = copy.copy(noise)  # never mutate the cached Noise object
        chunk_noise.seed = seed_i
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out_samples = _chunk_guider(guider, cond).sample(
            chunk_noise.generate_noise(chunk_latent), samples, sampler, sigmas,
            callback=callback, disable_pbar=disable_pbar, seed=seed_i)
        return {"samples": out_samples.to(comfy.model_management.intermediate_device())}

    def _chunk_key(self, cond, seed_i, ch, cw, n, sigmas, sampler, guider):
        refs = cond[0][1]["minimax_refs"]
        h = hashlib.sha1()
        h.update(refs[0]["latent"].detach().cpu().contiguous().numpy().tobytes())
        h.update(refs[1]["latent"].detach().cpu().contiguous().numpy().tobytes())
        h.update(repr((seed_i, ch, cw, n, _sampler_fp(sampler),
                       float(getattr(guider, "cfg", 0) or 0))).encode())
        h.update(sigmas.detach().cpu().float().numpy().tobytes())
        return h.digest()


NODE_CLASS_MAPPINGS = {"ViggleTextCondLoader": ViggleTextCondLoader,
                       "ViggleAnimateConditioning": ViggleAnimateConditioning,
                       "ViggleAnimateConditioningWindowed": ViggleAnimateConditioningWindowed,
                       "ViggleChunkedSampler": ViggleChunkedSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleTextCondLoader": "Load Text Conditioning (Viggle)",
                              "ViggleAnimateConditioning": "Viggle-Animate Conditioning (H3)",
                              "ViggleAnimateConditioningWindowed": "Viggle-Animate Conditioning (H3, Windowed)",
                              "ViggleChunkedSampler": "Viggle Chunked Sampler"}
