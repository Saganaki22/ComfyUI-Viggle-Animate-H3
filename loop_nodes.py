"""Experimental graph-expanded chunk sampling with disk checkpoints."""
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import tempfile

import safetensors
import safetensors.torch
import torch

import folder_paths
import nodes as comfy_nodes
import comfy.model_management
from comfy_execution.graph_utils import GraphBuilder, is_link

from comfy_extras import nodes_custom_sampler as core_sampler

from .nodes import FPS, ViggleChunkedSampler, _hash_cache_value, _validate_sigmas, _send_progress, _encode_anchor

CHECKPOINT_VERSION = 1


def _run_dir(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("Viggle: run_name must be 1–64 letters, digits, underscores or hyphens.")
    if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", run_name):
        raise ValueError("Viggle: choose a run_name that is not a reserved Windows filename.")
    root = Path(folder_paths.get_output_directory()).resolve()
    directory = (root / "viggle_chunks" / run_name).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Viggle: checkpoint directory must remain inside ComfyUI's output directory.")
    return directory


def _checkpoint_path(directory, filename):
    if not re.fullmatch(r"chunk_[0-9]+_[0-9a-f]{16}\.latent", filename):
        raise ValueError("Viggle: invalid checkpoint filename.")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Viggle: checkpoint path escapes its run directory.")
    return path


def _atomic_write(path, writer):
    fd, temporary = tempfile.mkstemp(prefix=".viggle-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sampling_graph_signature(dynprompt, unique_id):
    """Stable across node IDs/restarts; include loader file identity and node code."""
    memo = {}
    linked_files = {}
    loader_inputs = {"unet_name": "diffusion_models", "lora_name": "loras",
                     "ckpt_name": "checkpoints", "vae_name": "vae"}

    def describe(node_id):
        memo_key = node_id
        if memo_key in memo:
            return memo[memo_key]
        node = dynprompt.get_node(node_id)
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node["class_type"])
        inputs = {name: [describe(value[0]), value[1]] if is_link(value) else value
                  for name, value in sorted(node.get("inputs", {}).items())}
        files = []
        for name, category in loader_inputs.items():
            value = node.get("inputs", {}).get(name)
            if isinstance(value, str):
                path = folder_paths.get_full_path(category, value)
                if path is not None:
                    stat = os.stat(path)
                    files.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
            elif is_link(value):
                # The graph does not expose the evaluated filename. Track the
                # category's file metadata conservatively, without loading weights
                # or executing arbitrary filename-producing nodes ourselves.
                if category not in linked_files:
                    identities = []
                    for filename in sorted(folder_paths.get_filename_list(category)):
                        path = folder_paths.get_full_path(category, filename)
                        if path is not None:
                            stat = os.stat(path)
                            identities.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
                    linked_files[category] = identities
                files.append((category, linked_files[category]))
        source = inspect.getsourcefile(cls) if cls is not None else None
        code = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
        data = (node["class_type"], inputs, files, code)
        h = hashlib.sha256()
        if not _hash_cache_value(h, data):
            raise ValueError("Viggle: sampling graph contains runtime objects that cannot be recorded for resume.")
        memo[memo_key] = h.hexdigest()
        return memo[memo_key]

    node = dynprompt.get_node(unique_id)
    roots = [node["inputs"][key] for key in ("guider", "sampler", "sigmas")]
    if "vae" in node["inputs"]:
        roots.append(node["inputs"]["vae"])
    data = [(describe(value[0]), value[1]) for value in roots]
    data += [hashlib.sha256(Path(inspect.getsourcefile(core_sampler.Noise_RandomNoise)).read_bytes()).hexdigest()]
    # Changes to our carry/serialization implementation invalidate saved sampling.
    data += [hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             hashlib.sha256(Path(inspect.getsourcefile(ViggleChunkedSampler)).read_bytes()).hexdigest()]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def _read_checkpoint(directory, entry, canvas):
    path = _checkpoint_path(directory, entry["file"])
    a, b, _, latn = entry["span"]
    ch, cw = canvas
    expected = {"latent_tensor": [1, 24, latn, ch // 16, cw // 16],
                "audio": [1, 32, 2, round((b + 1) / FPS * 40) - round(a / FPS * 40)]}
    with safetensors.safe_open(path, framework="pt", device="cpu") as file:
        saved = json.loads(file.metadata()["viggle"])
        if saved != {"version": CHECKPOINT_VERSION, "entry": entry, "canvas": list(canvas)}:
            raise ValueError(f"Viggle: checkpoint metadata mismatch: {path.name}")
        for name, shape in expected.items():
            if file.get_slice(name).get_shape() != shape:
                raise ValueError(f"Viggle: invalid {name} shape in {path.name}")
        video, audio = file.get_tensor("latent_tensor"), file.get_tensor("audio")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError(f"Viggle: NaN/Inf in saved checkpoint {path.name}")
    return video, audio


class ViggleChunkLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_set": ("VIGGLE_COND_SET",),
            "run_name": ("STRING", {"default": "viggle_take_01", "tooltip": "Checkpoints go to output/viggle_chunks/<run_name>. Use a different name for separate projects."}),
            "resume": ("BOOLEAN", {"default": True, "tooltip": "Load matching saved chunks; changed settings get new checkpoints. Completed chunks are decoded/saved again."}),
        }, "optional": {"initial_state": ("VIGGLE_LOOP_STATE", {"forceInput": True})}}

    RETURN_TYPES = ("VIGGLE_LOOP", "VIGGLE_LOOP_STATE", "STRING")
    RETURN_NAMES = ("loop", "state", "status")
    FUNCTION = "start"
    CATEGORY = "sampling/viggle/experimental"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def start(self, cond_set, run_name, resume, initial_state=None):
        spans = cond_set["spans"]
        if not spans or len(spans) != len(cond_set["conds"]):
            raise ValueError("Viggle: conditioning windows are empty or inconsistent.")
        directory = _run_dir(run_name)
        directory.mkdir(parents=True, exist_ok=True)
        if initial_state is None:
            state = {"plan": cond_set, "run_name": run_name, "resume": resume,
                     "index": 0, "entries": [], "previous": None, "graph_signature": None}
        else:
            state = initial_state
        return ("loop", state, f"Chunk {state['index'] + 1} of {len(spans)}")


class ViggleSampleChunk:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = ViggleChunkedSampler.INPUT_TYPES()["required"]
        return {"required": {"state": ("VIGGLE_LOOP_STATE",),
                             **{name: inputs[name] for name in ("guider", "sampler", "sigmas", "seed", "rerender_chunk", "rerender_seed")}},
                "optional": {"vae": ("VAE", {"tooltip": "Connect the H3 VAE for five_frame_anchor continuation."})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_LOOP_STATE", "LATENT", "STRING")
    RETURN_NAMES = ("chunk", "video_latent", "filename_prefix")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle/experimental"
    DESCRIPTION = "Sample and checkpoint one chunk before external VAE decoding. Use between Viggle Chunk Loop Start and End."

    def sample(self, state, guider, sampler, sigmas, seed, rerender_chunk, rerender_seed,
               dynprompt, unique_id, vae=None):
        _validate_sigmas(sigmas)
        plan, i = state["plan"], state["index"]
        continuation = plan.get("continuation", "latent_overlap")
        if continuation == "five_frame_anchor" and vae is None:
            raise ValueError("Viggle: connect the H3 VAE to Sample Chunk for five_frame_anchor continuation.")
        a, b, lat0, latn = plan["spans"][i]
        ch, cw = plan["canvas"]
        seed_i = (int(rerender_seed) if int(rerender_chunk) == i + 1 else int(seed) + i) % (1 << 64)
        signature = state["graph_signature"] or _sampling_graph_signature(dynprompt, unique_id)
        predecessor = state["entries"][-1]["key"] if i else ""
        h = hashlib.sha256()
        values = (CHECKPOINT_VERSION, signature, continuation, predecessor, plan["conds"][i], sigmas,
                  (a, b, lat0, latn), (ch, cw), seed_i)
        if not _hash_cache_value(h, values):
            raise ValueError("Viggle: this conditioning cannot be fingerprinted for checkpoint recovery.")
        key = h.hexdigest()
        entry = {"file": f"chunk_{i + 1:04}_{key[:16]}.latent", "key": key,
                 "previous_key": predecessor, "span": [a, b, lat0, latn], "seed": seed_i}
        directory = _run_dir(state["run_name"])
        path = _checkpoint_path(directory, entry["file"])
        restored = state["resume"] and path.exists()
        status = f"Chunk {i + 1} of {len(plan['spans'])}: frames {a}-{b}, seed {seed_i}"
        logging.info("[ViggleSampleChunk] %s [%s]", status, "restored" if restored else "sampling")
        progress_node = dynprompt.get_display_node_id(unique_id)
        _send_progress(progress_node, status + (" — restoring" if restored else " — sampling"))
        if restored:
            out_v, out_a = _read_checkpoint(directory, entry, plan["canvas"])
        else:
            dev = comfy.model_management.intermediate_device()
            a0, a1 = round(a / FPS * 40), round((b + 1) / FPS * 40)
            v = torch.zeros([1, 24, latn, ch // 16, cw // 16], device=dev)
            au = torch.zeros([1, 32, 2, a1 - a0], device=dev)
            previous = state["previous"]
            carry = 0
            if previous is not None:
                pa, pb, pstart, plen = previous["span"]
                if continuation == "five_frame_anchor":
                    anchor = _encode_anchor(vae, previous["video"].to(dev), a - pa)
                    carry = anchor.shape[2]
                    v[:, :, :carry] = anchor.to(v)
                else:
                    carry = max(0, pstart + plen - lat0)
                    if carry:
                        v[:, :, :carry] = previous["video"][:, :, lat0 - pstart:lat0 - pstart + carry].to(dev)
                    audio_overlap = max(0, round((pb + 1) / FPS * 40) - a0)
                    audio_offset = a0 - round(pa / FPS * 40)
                    if audio_overlap:
                        au[..., :audio_overlap] = previous["audio"][..., audio_offset:audio_offset + audio_overlap].to(dev)
            out_v, out_a = ViggleChunkedSampler()._sample_window(
                core_sampler.Noise_RandomNoise(seed_i), guider, sampler, sigmas, plan["conds"][i], seed_i,
                ch, cw, a, b, carry, v, au)
            out_v, out_a = out_v.detach().cpu().contiguous(), out_a.detach().cpu().contiguous()
            metadata = {"viggle": json.dumps({"version": CHECKPOINT_VERSION, "entry": entry, "canvas": list(plan["canvas"])})}
            tensors = {"latent_tensor": out_v, "audio": out_a, "latent_format_version_0": torch.empty(0)}
            _atomic_write(path, lambda temporary: safetensors.torch.save_file(tensors, temporary, metadata=metadata))
        entries = state["entries"] + [entry]
        collection = {"version": CHECKPOINT_VERSION, "run_name": state["run_name"],
                      "canvas": list(plan["canvas"]), "total_frames": plan["total_frames"],
                      "continuation": continuation, "entries": entries}
        manifest = directory / "manifest.json"
        saved_collection = collection
        if restored and manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if (all(existing.get(k) == collection[k] for k in ("version", "run_name", "canvas", "total_frames", "continuation"))
                    and existing.get("entries", [])[:len(entries)] == entries):
                # Replaying previews must not hide the already completed suffix
                # if decoding or saving fails before the loop reaches it again.
                saved_collection = existing
        _atomic_write(manifest, lambda temporary: Path(temporary).write_text(
            json.dumps(saved_collection, indent=2), encoding="utf-8"))
        next_state = {**state, "index": i + 1, "entries": entries, "graph_signature": signature,
                      "progress_node": progress_node,
                      "previous": {"span": entry["span"], "video": out_v, "audio": out_a}}
        _send_progress(progress_node, status + (" — restored; decoding/saving" if restored else " — checkpoint saved; decoding/saving"))
        prefix = f"viggle_chunks/{state['run_name']}/preview_{i + 1:04}_{key[:8]}"
        return (next_state, {"samples": out_v.clone()}, prefix)


class ViggleChunkLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"loop": ("VIGGLE_LOOP", {"rawLink": True}),
                             "chunk": ("VIGGLE_LOOP_STATE",),
                             "images": ("IMAGE", {"tooltip": "Connect VAE Decode through SaveWEBM (images output) to save each chunk before advancing."})},
                "optional": {"after_save": ("*", {"tooltip": "Optional completion dependency, e.g. VHS Video Combine's filenames output."})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_CHUNKS", "STRING")
    RETURN_NAMES = ("chunks", "status")
    FUNCTION = "finish"
    CATEGORY = "sampling/viggle/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = "Wait for this chunk's decode/save branch, then advance the loop. Connect loop directly from Loop Start."

    def finish(self, loop, chunk, images, dynprompt, unique_id, after_save=None):
        if not torch.isfinite(images).all():
            raise ValueError("Viggle: decoded chunk contains NaN/Inf. Its latent checkpoint is already saved.")
        if "progress_node" in chunk:
            done = chunk["index"] == len(chunk["plan"]["spans"])
            _send_progress(chunk["progress_node"],
                           f"Chunk {chunk['index']} of {len(chunk['plan']['spans'])} — "
                           + ("loop completed (final assembly/decode may follow)" if done else "decode/save finished"))
        if chunk["index"] == len(chunk["plan"]["spans"]):
            collection = {"version": CHECKPOINT_VERSION, "run_name": chunk["run_name"],
                          "canvas": list(chunk["plan"]["canvas"]),
                          "total_frames": chunk["plan"]["total_frames"],
                          "continuation": chunk["plan"].get("continuation", "latent_overlap"), "entries": chunk["entries"]}
            return (collection, f"Completed {chunk['index']} chunks. Saved in {_run_dir(chunk['run_name'])}")

        # Copy only nodes lying between Loop Start and this End, as in ComfyUI's
        # graph-expansion loop example. External loaders/conditioning stay shared.
        children, visited = {}, set()
        pending = [unique_id]
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for value in dynprompt.get_node(node_id).get("inputs", {}).values():
                if is_link(value):
                    children.setdefault(value[0], set()).add(node_id)
                    pending.append(value[0])
        start_id = loop[0]
        if dynprompt.get_node(start_id)["class_type"] != "ViggleChunkLoopStart":
            raise ValueError("Viggle: connect Loop End's loop directly to Viggle Chunk Loop Start.")
        contained, pending = set(), [start_id]
        while pending:
            node_id = pending.pop()
            if node_id not in contained:
                contained.add(node_id)
                pending.extend(children.get(node_id, ()))
        graph = GraphBuilder()
        copies = {}
        for number, node_id in enumerate(sorted(contained)):
            original = dynprompt.get_node(node_id)
            node = graph.node(original["class_type"], "next" if node_id == unique_id else str(number))
            node.set_override_display_id(dynprompt.get_display_node_id(node_id))
            copies[node_id] = node
        for node_id, node in copies.items():
            for name, value in dynprompt.get_node(node_id).get("inputs", {}).items():
                node.set_input(name, copies[value[0]].out(value[1]) if is_link(value) and value[0] in copies else value)
        copies[start_id].set_input("initial_state", dynprompt.get_node(unique_id)["inputs"]["chunk"])
        return {"result": (copies[unique_id].out(0), copies[unique_id].out(1)), "expand": graph.finalize()}


class ViggleAssembleChunkLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run_name": ("STRING", {"default": "viggle_take_01", "tooltip": "Used when chunks is not connected; loads the saved manifest, including interrupted runs."}),
            "chunk_number": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 assembles all completed chunks. A positive number loads that one chunk for inspection."}),
        }, "optional": {"chunks": ("VIGGLE_CHUNKS",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "status")
    FUNCTION = "assemble"
    CATEGORY = "sampling/viggle/experimental"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, run_name, chunk_number, chunks=None):
        if chunks is None:
            chunks = json.loads((_run_dir(run_name) / "manifest.json").read_text(encoding="utf-8"))
        directory = _run_dir(chunks["run_name"])
        entries = chunks["entries"]
        if chunks["version"] != CHECKPOINT_VERSION or not entries:
            raise ValueError("Viggle: unsupported or empty checkpoint manifest.")
        if chunk_number:
            if chunk_number > len(entries):
                raise ValueError(f"Viggle: only {len(entries)} chunks are saved.")
            entry = entries[chunk_number - 1]
            video, _ = _read_checkpoint(directory, entry, chunks["canvas"])
            return ({"samples": video}, f"Chunk {chunk_number}: frames {entry['span'][0]}-{entry['span'][1]} (includes overlap)")
        last = entries[-1]["span"]
        ch, cw = chunks["canvas"]
        master = None
        predecessor, previous_end = "", 0
        for entry in entries:
            a, b, start, count = entry["span"]
            if entry["previous_key"] != predecessor or start > previous_end or start < 0:
                raise ValueError("Viggle: checkpoints do not form a continuous matching chain.")
            video, _ = _read_checkpoint(directory, entry, chunks["canvas"])
            if master is None:
                master = torch.empty((1, 24, last[2] + last[3], ch // 16, cw // 16), dtype=video.dtype, device="cpu")
            local_start = max(0, previous_end - start) if chunks.get("continuation") == "five_frame_anchor" else 0
            master[:, :, start + local_start:start + count] = video[:, :, local_start:]
            predecessor, previous_end = entry["key"], start + count
        complete = last[1] + 1 == chunks["total_frames"]
        return ({"samples": master}, f"{'Complete' if complete else 'Partial'}: {len(entries)} chunks, {last[1] + 1} frames. Decode once for the final video.")


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (ViggleChunkLoopStart, ViggleSampleChunk,
                                                   ViggleChunkLoopEnd, ViggleAssembleChunkLatents)}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleChunkLoopStart": "Viggle Chunk Loop Start",
                            "ViggleSampleChunk": "Viggle Sample Chunk",
                            "ViggleChunkLoopEnd": "Viggle Chunk Loop End",
                            "ViggleAssembleChunkLatents": "Viggle Assemble Chunk Latents"}
