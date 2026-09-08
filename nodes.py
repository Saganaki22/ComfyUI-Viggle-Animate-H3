import collections
import copy
import hashlib
import inspect
import logging
import math
import weakref

import torch

import folder_paths
import comfy.ldm.minimax.vae
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import latent_preview
from comfy_execution.utils import get_executing_context
from comfy_extras import nodes_minimax_h3 as core_h3
from comfy_extras import nodes_custom_sampler as core_sampler

def _same_stock_class(cls, stock):
    return (cls is stock or (cls is not None and cls.__qualname__ == stock.__qualname__
                            and inspect.getsourcefile(cls) == inspect.getsourcefile(stock)))


def _send_progress(node_id, text):
    from server import PromptServer
    server = getattr(PromptServer, "instance", None)
    context = get_executing_context()
    if server is not None and server.client_id is not None and context is not None:
        server.send_sync("viggle.chunk_progress", {
            "node_id": node_id, "text": text, "prompt_id": context.prompt_id,
        }, server.client_id)


CANVAS_MULTIPLE = 32
FPS = 24
MIN_ASPECT, MAX_ASPECT = 1 / 4, 4

_LATENT_CACHE = collections.OrderedDict()
_CACHE_MAX = 64
_LATENT_CACHE_MAX_BYTES = 256 * 1024 ** 2


def _fingerprint(t, extra):
    """Hash all content, copying at most a slab at a time for non-CPU inputs."""
    h = hashlib.sha256(repr((tuple(t.shape), str(t.dtype), extra)).encode())
    if t.numel() == 0:
        return h.digest()
    rows = t.detach().reshape(1) if t.ndim == 0 else t.detach()
    row_bytes = math.prod(rows.shape[1:]) * rows.element_size()
    step = max(1, (8 * 1024 ** 2) // max(1, row_bytes))
    for start in range(0, rows.shape[0], step):
        slab = rows[start:start + step].cpu().contiguous().view(torch.uint8)
        h.update(memoryview(slab.numpy()).cast("B"))
    return h.digest()


def _cache_get(key, vae):
    ent = _LATENT_CACHE.get(key)
    if ent is None:
        return None
    ref, val = ent
    if ref() is not vae:
        del _LATENT_CACHE[key]
        return None
    _LATENT_CACHE.move_to_end(key)
    return val.clone()


def _cache_put(key, vae, val):
    _LATENT_CACHE.pop(key, None)
    size = val.numel() * val.element_size()
    if size > _LATENT_CACHE_MAX_BYTES:
        return
    for old_key, (ref, _) in list(_LATENT_CACHE.items()):
        if ref() is None:
            del _LATENT_CACHE[old_key]
    total = sum(t.numel() * t.element_size() for _, t in _LATENT_CACHE.values())
    while _LATENT_CACHE and (total + size > _LATENT_CACHE_MAX_BYTES or len(_LATENT_CACHE) >= _CACHE_MAX):
        _, (_, old) = _LATENT_CACHE.popitem(last=False)
        total -= old.numel() * old.element_size()
    _LATENT_CACHE[key] = (weakref.ref(vae), val.detach().cpu().clone())


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
                               "tooltip": "Maximum frames at 24 fps (124 = ~5.2 s). Shorter inputs automatically use a shorter 17k+5 generation length."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/viggle"
    DESCRIPTION = ("Viggle-Animate conditioning: frozen text embed + video-first nested references. "
                   "Pair with MiniMaxH3SigmaShift (video/audio shift 3). The upstream four-point sigma schedule uses three Euler updates.")

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

        max_frames, _, _ = core_h3.temporal_shape(length)
        n = min(cond_video.shape[0], max_frames)
        if n < 1:
            raise ValueError("Viggle-Animate: driving clip contains no frames")
        frame_count, latent_t, audio_t = core_h3.temporal_shape(_generation_frame_count(n))

        # ---- reference 1: the driving video (first in the presentation) ----
        rh, rw = resolve_canvas(vw, vh, short_edge, max_pixels)
        frames = cond_video[:n]
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


def _generation_frame_count(fc):
    fc = max(5, int(fc))
    return fc + (5 - fc) % 17


def plan_spans(total_f, chunk_f, overlap_f):
    """Static window schedule: frames + latent placement.

    Windows keep each start on latent phase 0 and preserve the overlap stride.
    The final window uses only the remaining latents, including overlap.
    Returns [(first_frame, last_frame, lat_start, lat_count), ...], covering
    every input frame, with the end rounded UP to the 17j+5 frame grid.
    """
    L = _frames_to_latents(int(chunk_f))
    total_f = _generation_frame_count(total_f)
    total_lat = _frames_to_latents(total_f)
    if L < 7:
        raise ValueError("Viggle-Animate: chunk_frames must be at least 22 frames (124 recommended).")
    if L >= total_lat:
        return [(0, int(total_f) - 1, 0, total_lat)]
    O = _frames_to_latents(max(5, int(overlap_f)))
    O = max(2, min(O, L - 5))          # stride stays a multiple of 5 (phase 0)
    stride = L - O
    spans = []
    start = 0
    while True:
        length = min(L, total_lat - start)
        spans.append((_frame_at_latent(start), _frame_at_latent(start + length) - 1, start, length))
        if start + length == total_lat:
            return spans
        start += stride


# Rendered-chunk cache. With carry, chunks are CHAINED: chunk i+1 pins the
# previous chunk's tail, so its output depends on everything before it. Cache
# keys therefore chain: key_i = H(key_{i-1}, footage_i, seed_i, settings).
# Re-rendering chunk k invalidates k..end automatically; 1..k-1 stay cached.
_CHUNK_CACHE = collections.OrderedDict()
_CHUNK_CACHE_MAX_BYTES = 2 * 1024 ** 3


def _hash_cache_value(h, value):
    """Return False for opaque state whose changes cannot be tracked safely."""
    if isinstance(value, torch.Tensor):
        h.update(b"tensor" + _fingerprint(value, ()))
    elif isinstance(value, dict):
        if any(type(k) is not str for k in value):
            return False
        h.update(b"dict[")
        for k in sorted(value):
            if not _hash_cache_value(h, k) or not _hash_cache_value(h, value[k]):
                return False
        h.update(b"]")
    elif type(value) in (list, tuple):
        h.update(type(value).__name__.encode() + b"[")
        for item in value:
            if not _hash_cache_value(h, item):
                return False
        h.update(b"]")
    elif type(value) in (set, frozenset) and not value:
        h.update(b"empty-set")
    elif value is None or type(value) in (bool, int, float, str, bytes):
        encoded = repr((type(value).__name__, value)).encode()
        h.update(str(len(encoded)).encode() + b":" + encoded)
    else:
        return False
    return True


def _chunk_cache_get(key, owners):
    ent = _CHUNK_CACHE.get(key)
    if ent is None:
        return None
    refs, val = ent
    if any(ref() is not owner for ref, owner in zip(refs, owners)):
        del _CHUNK_CACHE[key]
        return None
    _CHUNK_CACHE.move_to_end(key)
    return (val[0].clone(), val[1].clone())


def _chunk_cache_put(key, owners, video, audio):
    if key is None:
        return
    _CHUNK_CACHE.pop(key, None)
    size = sum(t.numel() * t.element_size() for t in (video, audio))
    if size > _CHUNK_CACHE_MAX_BYTES:
        return
    for old_key, (refs, _) in list(_CHUNK_CACHE.items()):
        if any(ref() is None for ref in refs):
            del _CHUNK_CACHE[old_key]
    total = sum(t.numel() * t.element_size() for _, pair in _CHUNK_CACHE.values() for t in pair)
    while _CHUNK_CACHE and total + size > _CHUNK_CACHE_MAX_BYTES:
        _, (_, old) = _CHUNK_CACHE.popitem(last=False)
        total -= sum(t.numel() * t.element_size() for t in old)
    _CHUNK_CACHE[key] = (tuple(weakref.ref(owner) for owner in owners),
                         (video.detach().cpu().clone(), audio.detach().cpu().clone()))


def _chunk_guider(guider, positive):
    """The wired guider with its POSITIVE replaced by this chunk's conditioning."""
    new_g = copy.copy(guider)
    new_g.original_conds = dict(guider.original_conds)
    new_g.inner_set_conds({"positive": positive})
    return new_g


def _validate_sigmas(sigmas):
    if not torch.isfinite(sigmas).all() or (sigmas[:-1] <= 0).any() or (sigmas < 0).any():
        raise ValueError("Viggle Chunked Sampler: sigmas must be finite and positive except for "
                         "the final zero. A zero before the final point causes division by zero. "
                         "For the upstream schedule use ManualSigmas: 1.0, 0.8571428571428571, "
                         "0.6, 0.0. In KJNodes CustomSigmas set interpolate_to_steps to 3, not 4.")


class ViggleAnimateConditioningWindowed:
    """Windowed Viggle-Animate conditioning for long driving clips.

    Splits the driving clip into overlapping 17j+5 windows (default 124 frames —
    the finetune's evaluated chunk length) and emits one conditioning entry per
    window: the window's OWN footage as the video reference, plus the still,
    broadcast unchanged to every chunk. Viggle Chunked Sampler carries prior
    output into each overlap and decodes the assembled latent once.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_video": ("IMAGE", {"tooltip": "The whole driving clip at 24 fps. All loaded frames are used as reference. Generation ends on the next H3 frame-grid boundary, up to 16 frames longer (minimum 5)."}),
            "ref_image": ("IMAGE", {"tooltip": "Reference still shared by all chunks. A repainted frame from the driving shot with matching pose and framing gives the strongest reference."}),
            "text_cond": ("TEXT_COND", {"tooltip": "From the Load Text Conditioning node."}),
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE (from the base model)."}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                              "tooltip": "Target width. 0 = driving clip's own width (the evaluated configuration)."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                               "tooltip": "Target height. 0 = driving clip's own height."}),
            "chunk_frames": ("INT", {"default": 124, "min": 22, "max": 3600, "step": 17,
                                     "tooltip": "Maximum window length on H3's 17k+5 grid. The final window can be shorter. 124 is the usual baseline; shorter windows need visual testing."}),
            "overlap_frames": ("INT", {"default": 22, "min": 5, "max": 3600, "step": 17,
                                       "tooltip": "Overlap carried from the preceding chunk and preserved during sampling. Clamped below the window length so each chunk generates new frames."}),
        }}

    RETURN_TYPES = ("VIGGLE_COND_SET", "CONDITIONING")
    RETURN_NAMES = ("cond_set", "guider_positive")
    FUNCTION = "build"
    CATEGORY = "conditioning/viggle"
    DESCRIPTION = ("Viggle-Animate conditioning, windowed: per-chunk driving-video references "
                   "for the Viggle Chunked Sampler. Carries overlap to improve continuity across long clips.")

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

        # Round the generation extent up, retaining the original reference frames.
        asked_f = cond_video.shape[0]
        if asked_f < 1:
            raise ValueError("Viggle-Animate: driving clip needs at least one frame.")
        total_f = _generation_frame_count(asked_f)
        if total_f != asked_f:
            logging.info("[ViggleAnimateConditioningWindowed] %d frames -> %d on the 17j+5 "
                         "generation grid, %d additional frames to generate; all loaded reference frames retained",
                         asked_f, total_f, total_f - asked_f)

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
        reuse_overlap = (type(vae) is comfy.sd.VAE
                         and type(vae.first_stage_model) is comfy.ldm.minimax.vae.MiniMaxH3VideoVAE
                         and vae.first_stage_model.clip_length == 17
                         and vae.first_stage_model.token_drop == 3)
        conds, prompts = [], []
        reused_blocks = 0
        for i, (a, b, _lat0, latn) in enumerate(spans):
            n = b - a + 1
            frames = cond_video[a:a + n]
            vkey = _fingerprint(frames, ("v", n, rw, rh, id(vae)))
            z_video = _cache_get(vkey, vae)
            if z_video is None:
                prefix = None
                if reuse_overlap and i > 0:
                    previous = conds[-1][0][1]["minimax_refs"][0]["latent"]
                    offset = (a - spans[i - 1][0]) // 17 * 5
                    # Only complete 17-frame / 5-latent blocks are reusable.
                    # The last two latents depend on each window's padded tail.
                    shared = max(0, min(previous.shape[2] - 2 - offset, latn - 2))
                    if shared:
                        prefix = previous[:, :, offset:offset + shared]
                        frames = frames[shared // 5 * 17:]
                        reused_blocks += shared // 5
                if (vh, vw) != (rh, rw):
                    frames = core_h3._resize(frames, rw, rh, "disabled")
                z_video = vae.encode(frames)
                if prefix is not None:
                    z_video = torch.cat((prefix.to(z_video), z_video), dim=2)
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
                     "rerender_chunk is 1-based: %s; reused %d complete VAE blocks",
                     len(conds), total_f, total_f / FPS,
                     ", ".join(f"{a}-{b}" for a, b, _, _ in spans), reused_blocks)

        # guider_positive exists only so the guider's required `positive` socket
        # has a source — the sampler overwrites it per chunk from the cond_set.
        return ({"conds": conds, "prompts": prompts, "spans": spans,
                 "total_frames": total_f, "canvas": (ch, cw)}, conds[0])


class ViggleChunkedSampler:
    """Render a windowed Viggle clip chunk by chunk with LATENT CARRY.

    Chunk i's tail latents are written into a master latent and chunk i+1
    samples with a denoise_mask that PINS the overlap to that content (the
    model sees the carried rows as context at its cond timestep — core #15375).
    The master decodes once at the end. Preserving overlap improves continuity;
    motion and appearance in newly generated frames can still change at joins.

    Chunks are chained: chunk i+1 carries from chunk i, so cache keys chain
    too. Re-rendering chunk k (rerender_chunk + rerender_seed) re-renders
    k..end; earlier chunks can serve from the bounded cache. Opaque custom
    sampling state bypasses the cache.

    Takes the same NOISE / GUIDER / SAMPLER / SIGMAS objects as Sampler Custom
    Advanced — the guider's positive is swapped per chunk from the cond_set.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guider": ("GUIDER", {"tooltip": "From BasicGuider / CFGGuider. Only its model, cfg and negative matter — the positive is replaced per chunk from the cond_set."}),
            "sampler": ("SAMPLER", {"tooltip": "From KSamplerSelect or RES4LYF — reused for every chunk."}),
            "sigmas": ("SIGMAS", {"tooltip": "The step schedule (BasicScheduler etc.) — every chunk runs the identical schedule."}),
            "cond_set": ("VIGGLE_COND_SET", {"tooltip": "From Viggle-Animate Conditioning (H3, Windowed)."}),
            "vae": ("VAE", {"tooltip": "MiniMax-H3 video VAE. Decodes the finished master latent once; the model's (silent) audio half is discarded — keep your driving clip's own audio at save time."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "tooltip": "Base seed. Chunk i renders with seed + i, so the chunks vary independently while staying reproducible."}),
            "rerender_chunk": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                       "tooltip": "1-based chunk number to change (0 = off). Set a new rerender_seed to regenerate it and the following chunks; earlier chunks reuse cache when available."}),
            "rerender_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                      "tooltip": "Seed for the chunk selected by rerender_chunk. Type a new number for a new take."}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "chunk_map")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle"
    DESCRIPTION = ("Chunked Viggle-Animate sampler: latent carry for continuity, "
                   "per-chunk cache and re-render.")

    def sample(self, guider, sampler, sigmas, cond_set, vae, seed,
               rerender_chunk, rerender_seed, dynprompt=None, unique_id=None):
        _validate_sigmas(sigmas)
        progress_node = dynprompt.get_display_node_id(unique_id) if dynprompt is not None else unique_id
        windows = cond_set["spans"]      # (a, b, lat0, latn)
        conds = cond_set["conds"]
        if len(windows) != len(conds) or not conds:
            raise ValueError("Viggle Chunked Sampler: cond_set spans/conds mismatch.")
        total_f = int(cond_set["total_frames"])
        ch, cw = cond_set["canvas"]
        total_lat = _frames_to_latents(total_f)
        total_a = round(total_f / FPS * 40)
        model = guider.model_patcher
        noise = core_sampler.Noise_RandomNoise(int(seed))
        owners = (model, guider, sampler)
        sampling_key = self._sampling_key(noise, guider, sampler)
        dev = comfy.model_management.intermediate_device()

        master_v = torch.zeros([1, 24, total_lat, ch // 16, cw // 16], device=dev)
        master_a = torch.zeros([1, 32, 2, total_a], device=dev)

        pbar = comfy.utils.ProgressBar(len(windows))
        chunk_map = ["%d chunks, %d frames (%.1fs) at %dx%d — rerender_chunk is 1-based:"
                     % (len(windows), total_f, total_f / FPS, ch, cw)]
        if sampling_key is None:
            chunk_map.append("Chunk reuse disabled: custom sampling state cannot be checked.")
        prev_key = b""
        prev_end = None
        for i, (a, b, lat0, latn) in enumerate(windows):
            seed_i = int(rerender_seed) if int(rerender_chunk) == i + 1 else int(seed) + i
            seed_i %= 1 << 64
            carry = 0 if prev_end is None else max(0, prev_end - lat0)
            chunk_map.append("#%d: frames %d-%d (%.1f-%.1fs) seed %d carry %d lat"
                             % (i + 1, a, b, a / FPS, (b + 1) / FPS, seed_i, carry))
            key = self._chunk_key(prev_key, sampling_key, conds[i], seed_i, ch, cw,
                                  (a, b, lat0, latn, carry), sigmas)
            cached = _chunk_cache_get(key, owners)
            _send_progress(progress_node, f"Chunk {i + 1} of {len(windows)}: frames {a}-{b}, seed {seed_i} — "
                           + ("cached" if cached is not None else "sampling"))
            chunk_map[-1] += " [cached]" if cached is not None else " [rendered]"
            if cached is None:
                out_v, out_a = self._render_chunk(noise, guider, sampler, sigmas, conds[i],
                                                  seed_i, ch, cw, a, b, lat0, latn,
                                                  carry, master_v, master_a)
                _chunk_cache_put(key, owners, out_v, out_a)
            else:
                out_v, out_a = cached
                master_v[:, :, lat0:lat0 + latn] = out_v.to(dev)
                a0 = min(round(a / FPS * 40), total_a)
                a1 = min(round((b + 1) / FPS * 40), total_a)
                master_a[:, :, :, a0:a1] = out_a.to(dev)
            prev_key, prev_end = key, lat0 + latn
            pbar.update(1)

        _send_progress(progress_node, "Decoding final video")
        frames = vae.decode(master_v)
        if frames.dim() == 5:  # combine batches
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        _send_progress(progress_node, "Completed — final video decoded; downstream saving may follow")
        return (frames, "\n".join(chunk_map))

    def _render_chunk(self, noise, guider, sampler, sigmas, cond, seed_i,
                      ch, cw, a, b, lat0, latn, carry, master_v, master_a):
        total_a = master_a.shape[-1]
        a0 = min(round(a / FPS * 40), total_a)
        a1 = min(round((b + 1) / FPS * 40), total_a)
        v = master_v[:, :, lat0:lat0 + latn].clone()
        au = master_a[:, :, :, a0:a1].clone()
        out_v, out_a = self._sample_window(noise, guider, sampler, sigmas, cond, seed_i,
                                          ch, cw, a, b, carry, v, au)
        master_v[:, :, lat0:lat0 + latn] = out_v
        master_a[:, :, :, a0:a1] = out_a
        return out_v.cpu(), out_a.cpu()

    def _sample_window(self, noise, guider, sampler, sigmas, cond, seed_i,
                       ch, cw, a, b, carry, v, au):
        latn = v.shape[2]
        samples = comfy.nested_tensor.NestedTensor((v, au))
        samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, samples)
        chunk_latent = {"samples": samples}

        denoise_mask = None
        if carry > 0:  # pin the overlap to the previous chunk's output
            mask_v = torch.ones([1, 1, latn, ch // 16, cw // 16], device=v.device)
            mask_a = torch.ones([1, 1, 1, au.shape[-1]], device=au.device)
            mask_v[:, :, :carry] = 0.0
            denoise_mask = comfy.nested_tensor.NestedTensor((mask_v, mask_a))

        chunk_noise = copy.copy(noise)  # never mutate the cached Noise object
        chunk_noise.seed = seed_i
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out = _chunk_guider(guider, cond).sample(
            chunk_noise.generate_noise(chunk_latent), samples, sampler, sigmas,
            denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar,
            seed=seed_i)
        if out.is_nested:
            out_v, out_a = out.unbind()
        else:
            out_v, out_a = out, None
        for name, latent in (("video", out_v), ("audio", out_a)):
            if latent is not None and not torch.isfinite(latent).all():
                raise RuntimeError(f"Viggle Chunked Sampler: frames {a}-{b} produced NaN/Inf {name} "
                                   "latents. Stopped before caching or carrying them into the next chunk. "
                                   "Check the sigma schedule and model/attention settings.")
        return out_v, out_a

    def _sampling_key(self, noise, guider, sampler):
        # Only the stock implementations have a known state contract. Custom
        # objects still sample normally, but are not safe to reuse across runs.
        if (not any(_same_stock_class(type(noise), cls) for cls in (core_sampler.Noise_RandomNoise, core_sampler.Noise_EmptyNoise))
                or not any(_same_stock_class(type(guider), cls) for cls in (core_sampler.Guider_Basic, comfy.samplers.CFGGuider))
                or type(sampler) is not comfy.samplers.KSAMPLER):
            return None
        model = guider.model_patcher
        if set(model.object_patches) - {"model_sampling"}:
            return None
        other_conds = {name: [{k: v for k, v in entry.items() if k != "uuid"} for entry in entries]
                       for name, entries in guider.original_conds.items() if name != "positive"}
        sampling = model.get_model_object("model_sampling")
        h = hashlib.sha256()
        noise_state = {k: v for k, v in vars(noise).items() if k != "seed"}
        state = (type(noise).__name__, noise_state, guider.cfg, other_conds, sampler.extra_options, sampler.inpaint_options,
                 guider.model_options, vars(sampling), str(model.patches_uuid),
                 model.attachments, model.additional_models, model.callbacks, model.wrappers,
                 model.injections, model.hook_patches, model.weight_wrapper_patches)
        if not _hash_cache_value(h, state):
            return None
        return h.digest()

    def _chunk_key(self, prev_key, sampling_key, cond, seed_i, ch, cw, span, sigmas):
        if prev_key is None or sampling_key is None:
            return None
        h = hashlib.sha256()
        h.update(prev_key)
        h.update(sampling_key)
        if not _hash_cache_value(h, (cond, seed_i, ch, cw, span, sigmas)):
            return None
        return h.digest()


NODE_CLASS_MAPPINGS = {"ViggleTextCondLoader": ViggleTextCondLoader,
                       "ViggleAnimateConditioning": ViggleAnimateConditioning,
                       "ViggleAnimateConditioningWindowed": ViggleAnimateConditioningWindowed,
                       "ViggleChunkedSampler": ViggleChunkedSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleTextCondLoader": "Load Text Conditioning (Viggle)",
                              "ViggleAnimateConditioning": "Viggle-Animate Conditioning (H3)",
                              "ViggleAnimateConditioningWindowed": "Viggle-Animate Conditioning (H3, Windowed)",
                              "ViggleChunkedSampler": "Viggle Chunked Sampler"}
