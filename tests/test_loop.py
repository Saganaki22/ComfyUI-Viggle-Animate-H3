"""Executor-level and disk-layer checks for the chunk loop; run with the ComfyUI venv's Python."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))
sys.path.insert(0, str(PACK / "tests"))
sys.argv = [sys.argv[0], "--cpu"]

import types
pkg = types.ModuleType("viggle_pack")
pkg.__path__ = [str(PACK)]
sys.modules["viggle_pack"] = pkg
nodes_spec = importlib.util.spec_from_file_location("viggle_pack.nodes", PACK / "nodes.py")
viggle = importlib.util.module_from_spec(nodes_spec)
sys.modules["viggle_pack.nodes"] = viggle
nodes_spec.loader.exec_module(viggle)
loop_spec = importlib.util.spec_from_file_location("viggle_pack.loop_nodes", PACK / "loop_nodes.py")
loop = importlib.util.module_from_spec(loop_spec)
sys.modules["viggle_pack.loop_nodes"] = loop
loop_spec.loader.exec_module(loop)

import folder_paths
import nodes as comfy_nodes
import comfy.model_patcher
import comfy.model_sampling
from comfy_extras import nodes_custom_sampler as core

from test_chaining import conditioning


def fake_sample_record(record):
    def fake_sample(guider, noise, samples, sampler, sigmas, denoise_mask=None, **kwargs):
        record.append(("sample", kwargs["seed"]))
        outs = samples.unbind()
        eps_all = noise.unbind()
        masks = denoise_mask.unbind() if denoise_mask is not None else [None] * len(outs)
        streams = []
        for z, eps, mask in zip(outs, eps_all, masks):
            output = eps * 0.1 + 1.0001 + z.mean() * 0.25
            if mask is not None:
                output = output * mask + z * (1 - mask)
            streams.append(output)
        return comfy.nested_tensor.NestedTensor(streams)
    return fake_sample


class RecordingVAE:
    def __init__(self, record):
        self.record = record

    def decode(self, pixels):
        self.record.append(("decode", tuple(pixels["samples"].shape if isinstance(pixels, dict) else pixels.shape)))
        return torch.ones(1, 8, 8, 3)


def make_model_patcher():
    module = torch.nn.Module()
    module.model_sampling = comfy.model_sampling.ModelSamplingAV()
    module.model_sampling.set_parameters(shift=3.0, audio_shift=3.0)
    return comfy.model_patcher.ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))


def cond_set_fixture(total=73, chunk=39, overlap=22, canvas=(32, 32)):
    spans = viggle.plan_spans(total, chunk, overlap)
    return {"spans": spans, "conds": [conditioning() for _ in spans],
            "total_frames": total, "canvas": canvas}, len(spans)


class TestModelLoader:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (TestModelLoader.model,)


class TestConditioning:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (conditioning(),)


class TestSigmas:
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (torch.tensor([1.0, 6 / 7, 0.6, 0.0]),)


class TestNoise:
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (core.Noise_RandomNoise(0),)


class TestCondSet:
    RETURN_TYPES = ("VIGGLE_COND_SET",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (TestCondSet.cond_set,)


class TestVAE:
    RETURN_TYPES = ("VAE",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    def go(self):
        return (TestVAE.vae,)


for cls in (TestModelLoader, TestConditioning, TestSigmas, TestNoise, TestCondSet, TestVAE):
    cls.INPUT_TYPES = classmethod(lambda c: {"required": {}})


class LoopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)
        self.record = []
        self.progress = []
        self.model = make_model_patcher()
        TestModelLoader.model = self.model
        TestCondSet.cond_set, self.n_chunks = cond_set_fixture()
        TestVAE.vae = RecordingVAE(self.record)
        self._mappings = patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {
            "TestModelLoader": TestModelLoader, "TestConditioning": TestConditioning,
            "TestSigmas": TestSigmas, "TestNoise": TestNoise, "TestCondSet": TestCondSet,
            "TestVAE": TestVAE, "BasicGuider": core.BasicGuider,
            "KSamplerSelect": core.KSamplerSelect,
            **loop.NODE_CLASS_MAPPINGS})
        self._mappings.start()
        self._out = patch.object(folder_paths, "get_output_directory", lambda: str(self.out_dir))
        self._out.start()

    def tearDown(self):
        self._out.stop()
        self._mappings.stop()
        self._tmp.cleanup()

    def prompt(self, run_name, resume=False, rerender_chunk=0, rerender_seed=0):
        return {
            "1": {"class_type": "TestModelLoader", "inputs": {}},
            "2": {"class_type": "TestConditioning", "inputs": {}},
            "3": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["2", 0]}},
            "4": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "5": {"class_type": "TestSigmas", "inputs": {}},
            "6": {"class_type": "TestNoise", "inputs": {}},
            "7": {"class_type": "TestCondSet", "inputs": {}},
            "8": {"class_type": "ViggleChunkLoopStart",
                  "inputs": {"cond_set": ["7", 0], "run_name": run_name, "resume": resume}},
            "9": {"class_type": "ViggleSampleChunk",
                  "inputs": {"state": ["8", 1], "guider": ["3", 0],
                             "sampler": ["4", 0], "sigmas": ["5", 0], "seed": 42,
                             "rerender_chunk": rerender_chunk, "rerender_seed": rerender_seed}},
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 1], "vae": ["11", 0]}},
            "11": {"class_type": "TestVAE", "inputs": {}},
            "12": {"class_type": "ViggleChunkLoopEnd",
                   "inputs": {"loop": ["8", 0], "chunk": ["9", 0], "images": ["10", 0]}},
        }

    def execute(self, prompt):
        import execution
        from server import PromptServer
        server = type("S", (), {"client_id": None, "send_sync": lambda *a, **k: None})()
        executor = execution.PromptExecutor(
            server, cache_args={"ram": 16.0, "ram_inactive": 16.0, "lru": 0})
        progress_server = types.SimpleNamespace(client_id="test-client", send_sync=lambda *args: self.progress.append(args))
        with patch.object(PromptServer, "instance", progress_server, create=True), \
             patch.object(comfy.model_management, "load_models_gpu", lambda *a, **k: None), \
             patch.object(core.Guider_Basic, "sample", fake_sample_record(self.record)), \
             patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, samples: samples), \
             patch.object(viggle.latent_preview, "prepare_callback", lambda *a, **k: None):
            executor.execute(prompt, "test-prompt", execute_outputs=["12"])
        return executor

    def run_dir(self, name):
        return self.out_dir / "viggle_chunks" / name

    # ---- executor-level ----

    def test_live_progress_targets_original_node_through_all_iterations(self):
        self.assertTrue(self.execute(self.prompt("progress")).success)
        self.assertEqual(len(self.progress), self.n_chunks * 3)
        for event, data, client in self.progress:
            self.assertEqual(event, "viggle.chunk_progress")
            self.assertEqual(data["node_id"], "9")
            self.assertEqual(data["prompt_id"], "test-prompt")
            self.assertEqual(client, "test-client")
        for i in range(self.n_chunks):
            texts = [e[1]["text"] for e in self.progress[i * 3:i * 3 + 3]]
            self.assertIn(f"Chunk {i + 1} of {self.n_chunks}", texts[0])
            self.assertTrue(texts[0].endswith("sampling"))
            self.assertIn("checkpoint saved; decoding/saving", texts[1])
            self.assertIn("loop completed" if i + 1 == self.n_chunks else "decode/save finished", texts[2])

    def test_resume_progress_reports_restoring_without_sampling(self):
        self.assertTrue(self.execute(self.prompt("resume_progress")).success)
        self.progress.clear()
        self.record.clear()
        self.assertTrue(self.execute(self.prompt("resume_progress", resume=True)).success)
        self.assertEqual([e[0] for e in self.record], ["decode"] * self.n_chunks)
        self.assertEqual(len(self.progress), self.n_chunks * 3)
        self.assertTrue(all(e[1]["text"].endswith("restoring") for e in self.progress[::3]))

    def test_failed_decode_does_not_announce_loop_completion(self):
        with patch.object(RecordingVAE, "decode", side_effect=RuntimeError("test decode failure")):
            self.assertFalse(self.execute(self.prompt("failed_progress")).success)
        self.assertEqual(len(self.progress), 2)
        self.assertIn("checkpoint saved", self.progress[-1][1]["text"])

    def test_executor_runs_loop_with_decode_barrier_between_chunks(self):
        executor = self.execute(self.prompt("exec_run"))
        self.assertTrue(executor.success, executor.status_messages)
        files = sorted(p.name for p in self.run_dir("exec_run").glob("chunk_*.latent"))
        self.assertEqual(len(files), self.n_chunks)
        self.assertTrue((self.run_dir("exec_run") / "manifest.json").exists())
        kinds = [event[0] for event in self.record]
        self.assertEqual(kinds, ["sample", "decode"] * self.n_chunks)
        self.assertEqual([e[1] for e in self.record if e[0] == "sample"], [42 + i for i in range(self.n_chunks)])

    def test_resume_restores_every_chunk_without_resampling(self):
        self.execute(self.prompt("resume_run"))
        mtimes = {p.name: p.stat().st_mtime_ns for p in self.run_dir("resume_run").glob("chunk_*.latent")}
        self.record.clear()
        executor = self.execute(self.prompt("resume_run", resume=True))
        self.assertTrue(executor.success, executor.status_messages)
        self.assertEqual([e[0] for e in self.record], ["decode"] * self.n_chunks)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in self.run_dir("resume_run").glob("chunk_*.latent")},
                         mtimes)

    def test_rerender_invalidates_only_from_rerendered_chunk(self):
        self.execute(self.prompt("rerun_run"))
        self.record.clear()
        executor = self.execute(self.prompt("rerun_run", resume=True, rerender_chunk=2, rerender_seed=999))
        self.assertTrue(executor.success, executor.status_messages)
        seeds = [e[1] for e in self.record if e[0] == "sample"]
        self.assertEqual(seeds, [999, 44])
        files = sorted(p.name for p in self.run_dir("rerun_run").glob("chunk_*.latent"))
        self.assertEqual(len(files), self.n_chunks + 2)  # old chunk 2 and 3 kept alongside the new ones

    def test_stock_noise_seed_changes_do_not_resample(self):
        prompt = self.prompt("noise_resume", resume=True)
        prompt["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 100}}
        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {"RandomNoise": core.RandomNoise}):
            self.assertTrue(self.execute(prompt).success)
            self.record.clear()
            prompt["6"]["inputs"]["noise_seed"] = 200
            self.assertTrue(self.execute(prompt).success)
            self.assertEqual([e[0] for e in self.record], ["decode"] * self.n_chunks)

    def test_comfy_loader_noise_alias_resumes_after_seed_change(self):
        # Match load_custom_node's separate import of this core extras file.
        spec = importlib.util.spec_from_file_location("viggle_test_core_alias", core.__file__)
        alias = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: alias}):
            spec.loader.exec_module(alias)
            self.assertIsNot(alias.RandomNoise, core.RandomNoise)
            prompt = self.prompt("alias_noise_resume", resume=True)
            prompt["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 100}}
            with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {"RandomNoise": alias.RandomNoise}):
                self.assertTrue(self.execute(prompt).success)
                self.record.clear()
                prompt["6"]["inputs"]["noise_seed"] = 200
                self.assertTrue(self.execute(prompt).success)
                self.assertEqual([e[0] for e in self.record], ["decode"] * self.n_chunks)

    def test_linked_loader_file_replacement_changes_signature(self):
        from comfy_execution.graph import DynamicPrompt
        prompt = self.prompt("linked_loader")
        prompt["1"]["inputs"]["unet_name"] = ["filename", 0]
        prompt["filename"] = {"class_type": "PrimitiveString", "inputs": {"value": "weights.safetensors"}}
        weights = self.out_dir / "weights.safetensors"
        weights.write_bytes(b"old")
        with patch.object(folder_paths, "get_filename_list", return_value=[weights.name]), \
             patch.object(folder_paths, "get_full_path", return_value=str(weights)):
            first = loop._sampling_graph_signature(DynamicPrompt(prompt), "9")
            weights.write_bytes(b"replacement")
            second = loop._sampling_graph_signature(DynamicPrompt(prompt), "9")
            self.assertNotEqual(first, second)
            prompt["1"]["inputs"]["unet_name"] = weights.name
            first = loop._sampling_graph_signature(DynamicPrompt(prompt), "9")
            weights.write_bytes(b"another replacement")
            self.assertNotEqual(first, loop._sampling_graph_signature(DynamicPrompt(prompt), "9"))

    def test_unconnected_noise_seed_does_not_change_signature(self):
        from comfy_execution.graph import DynamicPrompt
        prompt = self.prompt("custom_noise")
        prompt["6"]["inputs"]["noise_seed"] = 100
        first = loop._sampling_graph_signature(DynamicPrompt(prompt), "9")
        prompt["6"]["inputs"]["noise_seed"] = 200
        self.assertEqual(first, loop._sampling_graph_signature(DynamicPrompt(prompt), "9"))

    def test_failed_resume_preview_preserves_completed_manifest(self):
        self.assertTrue(self.execute(self.prompt("preview_failure")).success)
        path = self.run_dir("preview_failure") / "manifest.json"
        before = json.loads(path.read_text())
        with patch.object(RecordingVAE, "decode", side_effect=RuntimeError("test decode failure")):
            self.assertFalse(self.execute(self.prompt("preview_failure", resume=True)).success)
        self.assertEqual(json.loads(path.read_text()), before)
        loop.ViggleAssembleChunkLatents().assemble("preview_failure", 0)

    def test_rerender_failure_manifest_keeps_only_matching_prefix(self):
        self.assertTrue(self.execute(self.prompt("changed_prefix")).success)
        original_decode = RecordingVAE.decode
        calls = []

        def fail_second(vae, pixels):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("test decode failure")
            return original_decode(vae, pixels)

        with patch.object(RecordingVAE, "decode", fail_second):
            self.assertFalse(self.execute(self.prompt("changed_prefix", resume=True,
                                                     rerender_chunk=2, rerender_seed=999)).success)
        manifest = json.loads((self.run_dir("changed_prefix") / "manifest.json").read_text())
        self.assertEqual([entry["seed"] for entry in manifest["entries"]], [42, 999])
        self.record.clear()
        self.assertTrue(self.execute(self.prompt("changed_prefix", resume=True,
                                                rerender_chunk=2, rerender_seed=999)).success)
        self.assertEqual([e[1] for e in self.record if e[0] == "sample"], [44])

    def test_assemble_full_partial_and_single_from_manifest(self):
        self.execute(self.prompt("assemble_run"))
        node = loop.ViggleAssembleChunkLatents()
        manifest = __import__("json").loads((self.run_dir("assemble_run") / "manifest.json").read_text())
        master, status = node.assemble("assemble_run", 0, None)
        last = manifest["entries"][-1]["span"]
        self.assertEqual(tuple(master["samples"].shape), (1, 24, last[2] + last[3], 2, 2))
        self.assertTrue(status.startswith("Complete" if last[1] + 1 == manifest["total_frames"] else "Partial"))
        single, status = node.assemble("assemble_run", 2, None)
        span = manifest["entries"][1]["span"]
        self.assertEqual(tuple(single["samples"].shape), (1, 24, span[3], 2, 2))
        self.assertIn(f"frames {span[0]}-{span[1]}", status)
        with self.assertRaises(ValueError):
            node.assemble("assemble_run", self.n_chunks + 1, None)

    def test_short_windows_decode_full_length_and_resume_rerender(self):
        # Exercise the real H3 temporal decoder; only the learned spatial
        # decoder is replaced. NaN-filled output detects any unwritten tail.
        cls = viggle.comfy.ldm.minimax.vae.MiniMaxH3VideoVAE
        vae = cls.__new__(cls)
        torch.nn.Module.__init__(vae)
        vae.clip_length, vae.token_drop = 17, 3
        vae.vae_ratio_t, vae.vae_ratio = 4, 16
        vae.frame_pre_padding, vae.tokens_chunk_size = 3, 5
        vae.token_overlap, vae.frame_overlap = 2, 5
        vae.decoder = types.SimpleNamespace(out_channels=3)
        vae._finalize_pixels = lambda x: x
        vae._adaptive_decode = lambda z: torch.ones(z.shape[0], 3, z.shape[2] * 4,
                                                   z.shape[3] * 16, z.shape[4] * 16)
        for frames in (5, 22, 39, 56, 73, 90, 107, 124, 141, 260, 345, 362):
            with self.subTest(frames=frames):
                TestCondSet.cond_set, self.n_chunks = cond_set_fixture(frames, 124, 22)
                name = f"short_{frames}"
                self.assertTrue(self.execute(self.prompt(name)).success)
                assembled, status = loop.ViggleAssembleChunkLatents().assemble(name, 0)
                self.assertTrue(status.startswith("Complete"))
                latent = assembled['samples']
                self.assertEqual(latent.shape[2], viggle._frames_to_latents(frames))
                buffer = torch.full(vae.decode_output_shape(latent.shape), float('nan'))
                decoded = vae.decode_temporal(latent, output_buffer=buffer)
                self.assertEqual(decoded.shape[2], frames)
                self.assertTrue(torch.isfinite(decoded).all())

                # Check that separate loop windows assemble exactly like the
                # single-pass sampler, including the shorter final carry.
                capture = types.SimpleNamespace()
                def capture_decode(value):
                    capture.latent = value.clone()
                    return torch.ones(1, 8, 8, 3)
                capture.decode = capture_decode
                with patch.object(core.Guider_Basic, "sample", fake_sample_record([])), \
                     patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, samples: samples), \
                     patch.object(viggle.latent_preview, "prepare_callback", lambda *a, **k: None):
                    guider = core.Guider_Basic(self.model)
                    guider.set_conds(conditioning())
                    viggle.ViggleChunkedSampler().sample(guider, comfy.samplers.ksampler('euler'),
                        torch.tensor([1.0, 6 / 7, 0.6, 0.0]), TestCondSet.cond_set, capture, 42, 0, 0)
                self.assertTrue(torch.equal(latent, capture.latent))
                self.record.clear()
                self.assertTrue(self.execute(self.prompt(name, resume=True)).success)
                self.assertEqual([e[0] for e in self.record], ['decode'] * self.n_chunks)
                self.record.clear()
                self.assertTrue(self.execute(self.prompt(name, resume=True,
                                                        rerender_chunk=self.n_chunks, rerender_seed=999)).success)
                self.assertEqual([e[1] for e in self.record if e[0] == 'sample'], [999])

    def test_assemble_rejects_broken_chains_and_version(self):
        import json as jsonlib
        import safetensors.torch
        directory = loop._run_dir("broken")
        directory.mkdir(parents=True)
        entries = [
            {"file": "chunk_0001_aaaaaaaaaaaaaaaa.latent", "key": "a", "previous_key": "",
             "span": [0, 38, 0, 12], "seed": 42},
            {"file": "chunk_0002_bbbbbbbbbbbbbbbb.latent", "key": "b", "previous_key": "mismatch",
             "span": [17, 55, 5, 12], "seed": 43}]
        for entry in entries:
            meta = {"viggle": jsonlib.dumps({"version": loop.CHECKPOINT_VERSION, "entry": entry,
                                             "canvas": [32, 32]})}
            loop._atomic_write(directory / entry["file"], lambda t, e=entry: safetensors.torch.save_file(
                {"latent_tensor": torch.zeros(1, 24, 12, 2, 2), "audio": torch.zeros(1, 32, 2, 65),
                 "latent_format_version_0": torch.empty(0)}, t, metadata=meta))
        broken = {"version": loop.CHECKPOINT_VERSION, "run_name": "broken", "canvas": [32, 32],
                  "total_frames": 75, "entries": entries}
        node = loop.ViggleAssembleChunkLatents()
        with self.assertRaises(ValueError):
            node.assemble("x", 0, broken)
        with self.assertRaises(ValueError):
            node.assemble("x", 0, {**broken, "version": 99})

    # ---- disk layer ----

    def test_run_name_and_checkpoint_paths_are_contained(self):
        for bad in ("..", "a/b", "a b", "", "x" * 65, "CON", "aux"):
            with self.assertRaises(ValueError, msg=bad):
                loop._run_dir(bad)
        with self.assertRaises(ValueError):
            loop._checkpoint_path(Path("."), "chunk_1_latent")
        with self.assertRaises(ValueError):
            loop._checkpoint_path(Path("."), "chunk_0001_0123456789abcdef.latent\\..\\..\\x")

    def test_checkpoint_roundtrip_and_rejections(self):
        directory = self.out_dir / "ck"
        directory.mkdir()
        entry = {"file": "chunk_0001_0123456789abcdef.latent", "key": "k", "previous_key": "",
                 "span": [0, 38, 0, 12], "seed": 42}
        video = torch.randn(1, 24, 12, 2, 2)
        audio = torch.randn(1, 32, 2, 65)
        import json as jsonlib
        import safetensors.torch
        meta = {"viggle": jsonlib.dumps({"version": loop.CHECKPOINT_VERSION, "entry": entry,
                                         "canvas": [32, 32]})}
        loop._atomic_write(directory / entry["file"], lambda t: safetensors.torch.save_file(
            {"latent_tensor": video, "audio": audio, "latent_format_version_0": torch.empty(0)}, t,
            metadata=meta))
        back_v, back_a = loop._read_checkpoint(directory, entry, (32, 32))
        self.assertTrue(torch.equal(back_v, video))
        self.assertTrue(torch.equal(back_a, audio))
        with self.assertRaises(ValueError):  # wrong canvas
            loop._read_checkpoint(directory, entry, (64, 64))
        bad = {**entry, "span": [0, 38, 0, 13]}
        with self.assertRaises(ValueError):  # wrong latent length
            loop._read_checkpoint(directory, bad, (32, 32))
        corrupt = {**entry, "file": "chunk_0002_0123456789abcdef.latent"}
        (directory / corrupt["file"]).write_bytes(b"not safetensors")
        with self.assertRaises(Exception):
            loop._read_checkpoint(directory, corrupt, (32, 32))

    def test_loop_start_validation_and_status(self):
        node = loop.ViggleChunkLoopStart()
        cond_set, n = cond_set_fixture()
        result = node.start(cond_set, "status_run", True, None)
        self.assertEqual(result[2], f"Chunk 1 of {n}")
        self.assertEqual(result[1]["index"], 0)
        with self.assertRaises(ValueError):
            node.start({"spans": [], "conds": []}, "status_run", True, None)

    def test_loop_end_expansion_copies_contained_nodes_only(self):
        cond_set, n = cond_set_fixture()
        state = {"plan": cond_set, "run_name": "expand_run", "resume": True, "index": 1,
                 "entries": [], "previous": None, "graph_signature": None}
        dyn = {
            "s": {"class_type": "ViggleChunkLoopStart", "inputs": {"cond_set": ["c", 0],
                                                                  "run_name": "expand_run", "resume": True}},
            "n": {"class_type": "ViggleSampleChunk",
                  "inputs": {"state": ["s", 1], "guider": ["x", 0],
                             "sampler": ["x", 0], "sigmas": ["x", 0], "seed": 42,
                             "rerender_chunk": 0, "rerender_seed": 0}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["n", 1], "vae": ["x", 0]}},
            "e": {"class_type": "ViggleChunkLoopEnd",
                  "inputs": {"loop": ["s", 0], "chunk": ["n", 0], "images": ["d", 0]}},
            "c": {"class_type": "TestCondSet", "inputs": {}},
            "x": {"class_type": "TestNoise", "inputs": {}},
            "o": {"class_type": "ViggleAssembleChunkLatents", "inputs": {"run_name": "expand_run",
                                                                         "chunk_number": 0,
                                                                         "chunks": ["e", 0]}},
        }

        class FakeDynPrompt:
            def get_node(self, node_id):
                return dyn[node_id]

            def get_display_node_id(self, node_id):
                return node_id

        end = loop.ViggleChunkLoopEnd()
        images = torch.ones(1, 8, 8, 3)
        out = end.finish(loop=["s", 0], chunk=state, images=images,
                         dynprompt=FakeDynPrompt(), unique_id="e")
        self.assertIn("expand", out)
        expanded = out["expand"]
        copied = {info["class_type"] for info in expanded.values()}
        self.assertEqual(copied, {"ViggleChunkLoopStart", "ViggleSampleChunk", "VAEDecode",
                                  "ViggleChunkLoopEnd"})
        self.assertNotIn("ViggleAssembleChunkLatents", copied)  # downstream of End: not in the loop
        end_links = out["result"]
        self.assertEqual(len(end_links), 2)
        self.assertEqual({link[0] for link in end_links}, {end_links[0][0]})
        self.assertEqual([link[1] for link in end_links], [0, 1])
        self.assertEqual(expanded[end_links[0][0]]["class_type"], "ViggleChunkLoopEnd")
        start_copy = [info for info in expanded.values() if info["class_type"] == "ViggleChunkLoopStart"][0]
        self.assertIn("initial_state", start_copy["inputs"])

    def test_loop_end_done_path_returns_collection(self):
        cond_set, n = cond_set_fixture()
        state = {"plan": cond_set, "run_name": "done_run", "resume": False, "index": n,
                 "entries": [{"file": "f", "key": "k", "previous_key": "", "span": [0, 38, 0, 12],
                              "seed": 42}],
                 "previous": None, "graph_signature": None}

        class FakeDynPrompt:
            def get_node(self, node_id):
                raise AssertionError("must not expand when done")

        end = loop.ViggleChunkLoopEnd()
        collection, status = end.finish(["s", 0], state, torch.ones(1, 8, 8, 3),
                                        dynprompt=FakeDynPrompt(), unique_id="e")
        self.assertEqual(collection["run_name"], "done_run")
        self.assertIn("Completed", status)
        with self.assertRaises(ValueError):
            end.finish(["s", 0], state, torch.full((1, 8, 8, 3), float("nan")),
                       dynprompt=FakeDynPrompt(), unique_id="e")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
