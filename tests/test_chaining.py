"""CPU regression checks; run with the ComfyUI venv's Python."""
import gc
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))
sys.argv = [sys.argv[0], "--cpu"]
spec = importlib.util.spec_from_file_location("viggle_nodes", PACK / "nodes.py")
viggle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viggle)

import comfy.model_patcher
import comfy.model_sampling
from comfy.k_diffusion.sampling import sample_euler
from comfy_extras import nodes_custom_sampler as core

torch.set_num_threads(2)


def conditioning(value=0.0):
    return [[torch.full((1, 3, 4), value, dtype=torch.bfloat16), {
        "minimax_refs": [{"latent": torch.full((1, 24, 37, 2, 2), value)},
                         {"latent": torch.zeros(1, 24, 1, 2, 2)}],
        "minimax_token_tags": torch.zeros(3, dtype=torch.int64),
    }]]


class DecodeVAE:
    def decode(self, latent):
        return latent.clone()


class ChainingTests(unittest.TestCase):
    def setUp(self):
        viggle._CHUNK_CACHE.clear()
        viggle._LATENT_CACHE.clear()
        module = torch.nn.Module()
        module.model_sampling = comfy.model_sampling.ModelSamplingAV()
        module.model_sampling.set_parameters(shift=3.0, audio_shift=3.0)
        self.model = comfy.model_patcher.ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))
        self.guider = core.Guider_Basic(self.model)
        self.guider.set_conds(conditioning())
        self.noise = core.Noise_RandomNoise(0)
        self.sampler = comfy.samplers.ksampler("euler")
        self.sigmas = torch.tensor([1.0, 6 / 7, 0.6, 0.0])
        self.node = viggle.ViggleChunkedSampler()
        self.owners = (self.model, self.noise, self.guider, self.sampler)

    def run_key(self):
        return self.node._sampling_key(self.noise, self.guider, self.sampler)

    def chunk_key(self, cond=None, previous=b"", span=(0, 123, 0, 37, 0)):
        return self.node._chunk_key(previous, self.run_key(), conditioning() if cond is None else cond,
                                    0, 32, 32, span, self.sigmas)

    def test_basic_and_cfg_preserve_original_conditions(self):
        for guider in (self.guider, comfy.samplers.CFGGuider(self.model)):
            if type(guider) is comfy.samplers.CFGGuider:
                guider.set_conds(conditioning(), conditioning(2))
                guider.set_cfg(2.5)
            original = dict(guider.original_conds)
            replacement = conditioning(1)
            updated = viggle._chunk_guider(guider, replacement)
            self.assertIs(updated.original_conds["positive"][0]["cross_attn"], replacement[0][0])
            self.assertIs(guider.original_conds["positive"], original["positive"])
            self.assertEqual(updated.cfg, guider.cfg)
            if "negative" in original:
                self.assertIs(updated.original_conds["negative"], original["negative"])

    def test_full_content_fingerprint(self):
        a = torch.zeros(1, 48, 48, 3)
        b = a.clone()
        a[0, 1, 1, 0] = 1
        b[0, 1, 3, 0] = 1
        self.assertNotEqual(viggle._fingerprint(a, ()), viggle._fingerprint(b, ()))
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.arange(24).reshape(2, 3, 4).to(dtype).transpose(1, 2)
            self.assertEqual(viggle._fingerprint(x, ()), viggle._fingerprint(x.contiguous(), ()))
        viggle._fingerprint(torch.tensor(1.0), ())
        viggle._fingerprint(torch.empty(0), ())
        viggle._fingerprint(torch.empty(1, 0, 3), ())

    def test_conditioning_and_schedule_invalidation(self):
        original = self.chunk_key()
        self.assertIsNotNone(original)
        for field in ("text", "tags", "reference"):
            cond = conditioning()
            if field == "text":
                cond[0][0] += 1
            elif field == "tags":
                cond[0][1]["minimax_token_tags"][0] += 1
            else:
                cond[0][1]["minimax_refs"][0]["latent"] += 1
            self.assertNotEqual(original, self.chunk_key(cond))
        self.assertNotEqual(original, self.chunk_key(previous=b"changed"))
        self.assertNotEqual(original, self.chunk_key(span=(17, 140, 5, 37, 7)))
        self.sigmas[1] = 0.8
        self.assertNotEqual(original, self.chunk_key())

    def test_sampler_model_and_noise_invalidation(self):
        original = self.run_key()
        self.sampler.extra_options["eta"] = 0.5
        self.assertNotEqual(original, self.run_key())
        self.sampler.extra_options.clear()
        self.sampler.inpaint_options["random"] = True
        self.assertNotEqual(original, self.run_key())
        self.sampler.inpaint_options.clear()
        self.guider.cfg = 1.5
        self.assertNotEqual(original, self.run_key())
        self.guider.cfg = 1.0
        self.model.model_options["transformer_options"]["test_setting"] = 1
        self.assertNotEqual(original, self.run_key())
        self.model.model_options["transformer_options"].clear()
        self.model.model.model_sampling.set_parameters(shift=4, audio_shift=3)
        self.assertNotEqual(original, self.run_key())
        self.model.model.model_sampling.set_parameters(shift=3, audio_shift=3)
        original = self.run_key()
        self.noise.seed = 10
        self.assertEqual(original, self.run_key())  # Per-chunk seed overrides this value.

    def test_negative_invalidation(self):
        self.guider = comfy.samplers.CFGGuider(self.model)
        self.guider.set_conds(conditioning(), conditioning())
        first = self.run_key()
        self.assertIsNotNone(first)
        self.guider.set_conds(conditioning(), conditioning())
        self.assertEqual(first, self.run_key())  # New internal UUIDs are not conditioning changes.
        self.guider.set_conds(conditioning(), conditioning(1))
        self.assertNotEqual(first, self.run_key())

    def test_opaque_state_bypasses_cache(self):
        self.sampler.extra_options["custom_callback"] = lambda x: x
        self.assertIsNone(self.run_key())
        self.sampler.extra_options.clear()
        self.model.attachments["custom_state"] = object()
        self.assertIsNone(self.run_key())
        self.model.attachments.clear()
        class CustomNoise(core.Noise_RandomNoise):
            pass
        self.noise = CustomNoise(0)
        self.assertIsNone(self.run_key())
        self.assertIsNone(self.chunk_key(previous=None))
        viggle._chunk_cache_put(None, self.owners, torch.ones(1), torch.ones(1))
        self.assertFalse(viggle._CHUNK_CACHE)

    def test_chunk_cache_byte_budget_precision_and_ownership(self):
        video = torch.full((5,), 1.0001)
        audio = torch.full((1,), 2.0001)
        with patch.object(viggle, "_CHUNK_CACHE_MAX_BYTES", 32):
            for key in (b"first", b"second"):
                viggle._chunk_cache_put(key, self.owners, video, audio)
            self.assertEqual(list(viggle._CHUNK_CACHE), [b"second"])
            cached = viggle._chunk_cache_get(b"second", self.owners)
            self.assertTrue(torch.equal(cached[0], video))
            cached[0].zero_()
            self.assertTrue(torch.equal(viggle._chunk_cache_get(b"second", self.owners)[0], video))
            viggle._chunk_cache_put(b"large", self.owners, torch.zeros(100), audio)
            self.assertNotIn(b"large", viggle._CHUNK_CACHE)
            other_noise = core.Noise_RandomNoise(0)
            self.assertIsNone(viggle._chunk_cache_get(b"second", (self.model, other_noise, self.guider, self.sampler)))

    def test_reference_cache_budget_and_copy(self):
        vae = DecodeVAE()
        with patch.object(viggle, "_LATENT_CACHE_MAX_BYTES", 24):
            viggle._cache_put(b"first", vae, torch.ones(4))
            viggle._cache_put(b"second", vae, torch.ones(4))
            self.assertEqual(list(viggle._LATENT_CACHE), [b"second"])
            viggle._cache_get(b"second", vae).zero_()
            self.assertTrue(torch.equal(viggle._cache_get(b"second", vae), torch.ones(4)))
            viggle._cache_put(b"large", vae, torch.ones(7))
            self.assertNotIn(b"large", viggle._LATENT_CACHE)
            del vae
            gc.collect()
            other = DecodeVAE()
            viggle._cache_put(b"third", other, torch.ones(4))
            self.assertEqual(list(viggle._LATENT_CACHE), [b"third"])

    def test_cold_cached_and_suffix_rerender_carry(self):
        spans = viggle.plan_spans(260, 124, 22)
        cond_set = {"spans": spans, "conds": [conditioning() for _ in spans],
                    "total_frames": 260, "canvas": (32, 32)}
        calls = []
        def fake_sample(guider, noise, samples, sampler, sigmas, denoise_mask=None, **kwargs):
            calls.append((kwargs["seed"], samples.unbind()[0].clone()))
            streams = []
            for i, (z, eps) in enumerate(zip(samples.unbind(), noise.unbind())):
                output = eps * 0.1 + 1.0001 + z.mean() * 0.25
                if denoise_mask is not None:
                    mask = denoise_mask.unbind()[i]
                    output = output * mask + z * (1 - mask)
                streams.append(output)
            return comfy.nested_tensor.NestedTensor(streams)
        def run(chunk=0, seed=0):
            return self.node.sample(self.guider, self.sampler, self.sigmas,
                                    cond_set, DecodeVAE(), 10, chunk, seed)[0]
        with patch.object(core.Guider_Basic, "sample", fake_sample), \
             patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, samples: samples), \
             patch.object(viggle.latent_preview, "prepare_callback", lambda *args: None), \
             patch.object(viggle, "_send_progress") as progress:
            cold = run()
            self.assertEqual(len(calls), 3)
            texts = [call.args[1] for call in progress.call_args_list]
            self.assertEqual(len(texts), 5)
            self.assertTrue(all("sampling" in text for text in texts[:3]))
            self.assertEqual(texts[3], "Decoding final video")
            self.assertTrue(texts[4].startswith("Completed"))
            progress.reset_mock()
            self.assertGreater(calls[1][1].abs().sum().item(), 0)
            self.assertTrue(torch.equal(cold, run()))
            self.assertTrue(all("cached" in call.args[1] for call in progress.call_args_list[:3]))
            self.assertEqual(len(calls), 3)
            suffix_cached = run(2, 123)
            self.assertEqual([x[0] for x in calls], [10, 11, 12, 123, 12])
            viggle._CHUNK_CACHE.clear()
            suffix_fresh = run(2, 123)
            self.assertTrue(torch.equal(suffix_cached, suffix_fresh))

    def test_decode_trims_padding_and_rejects_missing_source_frames(self):
        plan = {"spans": [(0, 21, 0, 7)], "conds": [conditioning()],
                "total_frames": 22, "source_frames": 20, "canvas": (32, 32)}
        pixels = torch.arange(22).reshape(22, 1, 1, 1).expand(-1, 2, 2, 3)
        with patch.object(self.node, "_render_chunk", return_value=(torch.zeros(1), torch.zeros(1))), \
             patch.object(viggle, "_chunk_cache_get", return_value=None), \
             patch.object(DecodeVAE, "decode", return_value=pixels) as decode:
            frames, _ = self.node.sample(self.guider, self.sampler, self.sigmas,
                                         plan, DecodeVAE(), 10, 0, 0)
            self.assertTrue(torch.equal(frames, pixels[:20]))
            decode.return_value = pixels[:19]
            with self.assertRaisesRegex(ValueError, "shorter than the source"):
                self.node.sample(self.guider, self.sampler, self.sigmas,
                                 plan, DecodeVAE(), 10, 0, 0)

    def test_five_frame_anchor_offsets_assembly_cache_and_rerender(self):
        class AnchorVAE:
            def __init__(self):
                self.encoded = []
                self.decoded = []

            def decode(self, video):
                self.decoded.append(video.clone())
                n = viggle._frame_at_latent(video.shape[2])
                return (torch.arange(n).float() + video[0, 0, 0, 0, 0] * 1000).reshape(n, 1, 1, 1).expand(-1, 32, 32, 3)

            def encode(self, frames):
                self.encoded.append(frames.clone())
                return torch.full((1, 24, 2, 2, 2), frames[0, 0, 0, 0].item() + 100)

        vae = AnchorVAE()
        spans = viggle.plan_spans(322, 124, 5)
        plan = {"spans": spans, "conds": [conditioning() for _ in spans],
                "total_frames": 328, "source_frames": 322, "canvas": (32, 32),
                "continuation": "five_frame_anchor"}
        calls = []
        def sample(noise, guider, sampler, sigmas, cond, seed, ch, cw, a, b, carry, v, au):
            calls.append((seed, carry, v.clone(), au.clone()))
            # Deliberately change even overlap values to verify assembly keeps
            # the accepted prefix, including the final window's warm-up region.
            return torch.full_like(v, seed), torch.full_like(au, seed)

        with patch.object(self.node, "_sample_window", side_effect=sample):
            frames, _ = self.node.sample(self.guider, self.sampler, self.sigmas, plan, vae, 10, 0, 0)
            self.assertEqual(len(frames), 322)
            self.assertEqual([c[:2] for c in calls], [(10, 0), (11, 2), (12, 2)])
            self.assertTrue(torch.equal(vae.encoded[0][:, 0, 0, 0], torch.arange(119, 124) + 10000))
            self.assertTrue(torch.equal(vae.encoded[1][:, 0, 0, 0], torch.arange(85, 90) + 11000))
            for call, encoded in zip(calls[1:], vae.encoded):
                self.assertTrue((call[2][:, :, :2] == encoded[0, 0, 0, 0] + 100).all())
                self.assertEqual(call[2][:, :, 2:].count_nonzero(), 0)
                self.assertEqual(call[3].count_nonzero(), 0)
            master = vae.decoded[-1]
            self.assertTrue((master[:, :, :37] == 10).all())
            self.assertTrue((master[:, :, 37:72] == 11).all())
            self.assertTrue((master[:, :, 72:] == 12).all())
            self.node.sample(self.guider, self.sampler, self.sigmas, plan, vae, 10, 0, 0)
            self.assertEqual(len(calls), 3)
            self.node.sample(self.guider, self.sampler, self.sigmas, plan, vae, 10, 2, 99)
            self.assertEqual([c[0] for c in calls], [10, 11, 12, 99, 12])
            self.assertTrue((vae.decoded[-1][:, :, :37] == 10).all())
            self.assertTrue((vae.decoded[-1][:, :, 37:72] == 99).all())
            self.node.sample(self.guider, self.sampler, self.sigmas, plan, AnchorVAE(), 10, 0, 0)
            self.assertEqual([c[0] for c in calls[-3:]], [10, 11, 12])

    def test_anchor_rejects_short_decode_and_wrong_encoder(self):
        vae = type("VAE", (), {})()
        vae.decode = lambda video: torch.zeros(4, 32, 32, 3)
        with self.assertRaisesRegex(ValueError, "five finite"):
            viggle._encode_anchor(vae, torch.zeros(1), 0)
        vae.decode = lambda video: torch.zeros(5, 32, 32, 3)
        vae.encode = lambda frames: torch.zeros(1, 24, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "two finite"):
            viggle._encode_anchor(vae, torch.zeros(1), 0)

    def test_comfy_core_alias_cache_and_suffix_rerender(self):
        spec = importlib.util.spec_from_file_location("viggle_chaining_core_alias", core.__file__)
        alias = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: alias}):
            spec.loader.exec_module(alias)
            self.noise = alias.Noise_RandomNoise(0)
            self.guider = alias.Guider_Basic(self.model)
            self.guider.set_conds(conditioning())
            original = self.run_key()
            self.assertIsNotNone(original)
            self.noise.seed = 999
            self.assertEqual(original, self.run_key())
            with patch.object(sys.modules[__name__], "core", alias):
                self.test_cold_cached_and_suffix_rerender_carry()

    def test_frame_schedule_coverage(self):
        for total in range(5, 1800, 17):
            for chunk in (22, 39, 56, 124, 243, 362):
                for overlap in (5, 22, 39, 124, 362):
                    spans = viggle.plan_spans(total, chunk, overlap)
                    self.assertEqual((spans[0][0], spans[-1][1]), (0, total - 1))
                    for a, b, start, length in spans:
                        self.assertEqual((a % 17, (b - a + 1) % 17, start % 5), (0, 5, 0))
                    for left, right in zip(spans, spans[1:]):
                        self.assertLessEqual(right[0], left[1] + 1)
                        self.assertGreater(right[1], left[1])
                        self.assertEqual(right[3], left[3])

    def test_full_final_window_preserves_source_ending(self):
        self.assertEqual(viggle.plan_spans(192, 124, 22),
                         [(0, 123, 0, 37), (68, 191, 20, 37)])
        self.assertEqual(viggle.plan_spans(322, 124, 5),
                         [(0, 123, 0, 37), (119, 242, 35, 37), (204, 327, 60, 37)])
        for total in range(1, 400):
            spans = viggle.plan_spans(total, 124, 22)
            self.assertEqual(spans[-1][1] + 1, viggle._generation_frame_count(total))
            self.assertGreaterEqual(spans[-1][1] + 1, total)
            self.assertLessEqual(spans[-1][1] + 1 - total, 16)
        self.assertEqual(viggle.plan_spans(362, 124, 22),
                         [(0, 123, 0, 37), (102, 225, 30, 37),
                          (204, 327, 60, 37), (238, 361, 70, 37)])
        self.assertEqual(viggle.plan_spans(345, 124, 22)[-1], (221, 344, 65, 37))
        self.assertEqual(viggle.plan_spans(124, 124, 22), [(0, 123, 0, 37)])

    def test_conditioning_keeps_source_and_pads_only_grid_tail(self):
        class RecordingEncoder:
            def __init__(self):
                self.inputs = []

            def encode(self, frames):
                self.inputs.append(frames.clone())
                return torch.zeros(1, 24, viggle._frames_to_latents(len(frames)), 2, 2)

        text = {"prompt_embeds": conditioning()[0][0], "text_token_tags": torch.zeros(3, dtype=torch.int64)}
        for total in (1, 4, 5, 21, 22, 38, 39, 56, 73, 90, 107, 123, 124, 125, 289, 345, 361, 362):
            with self.subTest(total=total):
                viggle._LATENT_CACHE.clear()
                vae = RecordingEncoder()
                video = torch.arange(total, dtype=torch.float32)[:, None, None, None].expand(-1, 32, 32, 3)
                plan = viggle.ViggleAnimateConditioningWindowed().build(
                    video, video[:1], text, vae, 0, 0, 124, 22)[0]
                seen = set()
                for span, encoded in zip(plan['spans'], vae.inputs[1:]):
                    a, b, _, _ = span
                    expected = torch.arange(a, b + 1).clamp(max=total - 1).float()
                    self.assertTrue(torch.equal(encoded[:, 0, 0, 0], expected))
                    seen.update(encoded[:, 0, 0, 0].tolist())
                self.assertEqual(seen, set(range(total)))
                self.assertEqual(plan['total_frames'], viggle._generation_frame_count(total))
                self.assertEqual(plan['source_frames'], total)
                self.assertEqual(plan['continuation'], 'five_frame_anchor')
                self.assertEqual(plan['spans'], viggle.plan_spans(total, 124, 5))

    def test_single_shot_length_is_a_maximum_without_repeated_reference_frames(self):
        class RecordingEncoder:
            def __init__(self):
                self.inputs = []

            def encode(self, frames):
                self.inputs.append(frames.clone())
                return torch.zeros(1, 24, viggle._frames_to_latents(
                    viggle._generation_frame_count(len(frames))), 2, 2)

        text = {"prompt_embeds": conditioning()[0][0], "text_token_tags": torch.zeros(3, dtype=torch.int64)}
        for total in (1, 4, 5, 22, 39, 55, 56, 57, 73, 90, 107, 123, 124, 125, 362):
            for maximum in (56, 124):
                with self.subTest(total=total, maximum=maximum):
                    viggle._LATENT_CACHE.clear()
                    vae = RecordingEncoder()
                    video = torch.arange(total).float()[:, None, None, None].expand(-1, 32, 32, 3)
                    _, latent = viggle.ViggleAnimateConditioning().build(
                        video, video[:1], text, vae, 0, 0, maximum)
                    used = min(total, maximum)
                    self.assertTrue(torch.equal(vae.inputs[0], video[:used]))
                    generated = viggle._generation_frame_count(used)
                    self.assertEqual(latent['samples'].tensors[0].shape[2], viggle._frames_to_latents(generated))
                    self.assertLessEqual(generated, maximum)

    def test_invalid_sigmas_stop_before_sampling(self):
        for values in ([1, 0.89, 0.72, 0, 0], [1, float("nan"), 0],
                       [1, float("inf"), 0], [1, -0.1, 0]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "interpolate_to_steps to 3"):
                self.node.sample(self.guider, self.sampler, torch.tensor(values),
                                 {}, DecodeVAE(), 0, 0, 0)
        self.assertFalse(viggle._CHUNK_CACHE)

    def test_windowed_encode_reuses_only_complete_temporal_blocks(self):
        # Use the real VAE wrapper, temporal padding, normalization and slicing.
        # Replace the heavy spatial network with a deterministic, clip-dependent
        # encoder, recording the exact inputs that the real network would see.
        class WrappedVAE(comfy.sd.VAE):
            pass

        def make_vae(wrapper):
            vae = wrapper.__new__(wrapper)
            network = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE.__new__(comfy.ldm.minimax.vae.MiniMaxH3VideoVAE)
            torch.nn.Module.__init__(network)
            network.clip_length, network.token_drop = 17, 3
            network.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1, 1)
            network.pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1, 1)
            network.latents_mean = torch.linspace(-1, 1, 24)
            network.latents_std = torch.linspace(0.5, 1.5, 24)
            seen = []
            def encode_spatial(x):
                seen.append(x.clone())
                z = torch.nn.functional.adaptive_avg_pool3d(x.mean(dim=1, keepdim=True),
                                                            (5, x.shape[-2] // 16, x.shape[-1] // 16))
                return (z + x.mean()).repeat(1, 48, 1, 1, 1)
            network._adaptive_encode = encode_spatial
            vae.first_stage_model = network
            vae.crop_input, vae.output_channels = False, 3
            vae.latent_dim, vae.not_video = 3, False
            vae.device = vae.output_device = torch.device("cpu")
            vae.vae_dtype = torch.float16
            vae.disable_offload = False
            vae.format_encoded = None
            vae.memory_used_encode = lambda *args: 1
            vae.process_input = lambda pixels: pixels * 2 - 1
            vae.patcher = self.model
            return vae, seen

        text = {"prompt_embeds": conditioning()[0][0], "text_token_tags": torch.zeros(3, dtype=torch.int64)}
        generator = torch.Generator().manual_seed(123)
        video = torch.rand(362, 32, 32, 3, generator=generator)
        still = torch.rand(1, 32, 32, 3, generator=generator)
        node = viggle.ViggleAnimateConditioningWindowed()
        cases = [(1, 124, 22, 0), (4, 124, 22, 0), (5, 124, 22, 0),
                 (22, 22, 5, 0), (39, 39, 22, 0), (56, 56, 22, 0),
                 (73, 73, 22, 0), (90, 90, 22, 0), (107, 107, 22, 0),
                 (123, 124, 22, 0), (124, 124, 22, 0), (125, 124, 22, 0),
                 (361, 124, 22, 0), (362, 124, 22, 0),
                 (289, 124, 22, 0), (289, 124, 5, 0), (345, 124, 22, 0), (345, 124, 5, 0), (345, 124, 39, 0),
                 (260, 56, 39, 0), (345, 124, 22, 64)]
        with patch.object(comfy.model_management, "load_models_gpu"):
            for total, chunk, overlap, size in cases:
                with self.subTest(total=total, chunk=chunk, overlap=overlap, size=size):
                    viggle._LATENT_CACHE.clear()
                    old_vae, old_calls = make_vae(WrappedVAE)
                    new_vae, new_calls = make_vae(comfy.sd.VAE)
                    args = (video[:total], still, text)
                    old = node.build(*args, old_vae, size, size, chunk, overlap, "latent_overlap")[0]
                    new = node.build(*args, new_vae, size, size, chunk, overlap, "latent_overlap")[0]
                    self.assertEqual(old["spans"], new["spans"])
                    for plan in (old, new):
                        for span, cond in zip(plan["spans"], plan["conds"]):
                            self.assertEqual(cond[0][1]["minimax_refs"][0]["latent"].shape[2], span[3])
                    for left, right in zip(old["conds"], new["conds"]):
                        for a, b in zip(left[0][1]["minimax_refs"], right[0][1]["minimax_refs"]):
                            self.assertTrue(torch.equal(a["latent"], b["latent"]))
                    self.assertLessEqual(len(new_calls), len(old_calls))
                    for encoded in new_calls:
                        self.assertTrue(any(torch.equal(encoded, original) for original in old_calls))
                    if (total, chunk, overlap, size) == (345, 124, 22, 0):
                        self.assertEqual((len(old_calls) - 1, len(new_calls) - 1), (32, 24))
                    before = len(new_calls)
                    warm = node.build(*args, new_vae, size, size, chunk, overlap, "latent_overlap")[0]
                    self.assertEqual(before, len(new_calls))
                    for cold_cond, warm_cond in zip(new["conds"], warm["conds"]):
                        self.assertTrue(torch.equal(cold_cond[0][1]["minimax_refs"][0]["latent"],
                                                    warm_cond[0][1]["minimax_refs"][0]["latent"]))

    def test_euler_duplicate_zero_corrupts_output_after_finite_preview(self):
        previews = []
        def model(x, sigma, **kwargs):
            return torch.full_like(x, 0.25)
        bad = torch.tensor([1.0, 0.890819907, 0.717137158, 0.0, 0.0])
        output = sample_euler(model, torch.ones(1), bad, disable=True,
                              callback=lambda state: previews.append(state["denoised"].clone()))
        self.assertTrue(all(torch.isfinite(p).all() for p in previews))
        self.assertTrue(torch.isnan(output).all())
        fixed = sample_euler(model, torch.ones(1), self.sigmas, disable=True)
        self.assertTrue(torch.equal(fixed, torch.full((1,), 0.25)))

    def test_nonfinite_chunk_is_not_cached_or_carried(self):
        spans = viggle.plan_spans(260, 124, 22)
        cond_set = {"spans": spans, "conds": [conditioning() for _ in spans],
                    "total_frames": 260, "canvas": (32, 32)}
        for stream in (0, 1):
            def broken_sample(guider, noise, samples, *args, **kwargs):
                parts = [z.clone() for z in samples.unbind()]
                parts[stream].fill_(float("nan"))
                return comfy.nested_tensor.NestedTensor(parts)
            with patch.object(core.Guider_Basic, "sample", autospec=True, side_effect=broken_sample) as sample, \
                 patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, samples: samples), \
                 patch.object(viggle.latent_preview, "prepare_callback", lambda *args: None), \
                 self.assertRaisesRegex(RuntimeError, "frames 0-123 produced NaN/Inf"):
                self.node.sample(self.guider, self.sampler, self.sigmas,
                                 cond_set, DecodeVAE(), 10, 0, 0)
            self.assertEqual(sample.call_count, 1)
            self.assertFalse(viggle._CHUNK_CACHE)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
