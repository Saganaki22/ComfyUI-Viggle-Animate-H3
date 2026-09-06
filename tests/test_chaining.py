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
        self.noise.seed = 10
        self.assertNotEqual(original, self.run_key())

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
            return self.node.sample(self.noise, self.guider, self.sampler, self.sigmas,
                                    cond_set, DecodeVAE(), 10, chunk, seed)[0]
        with patch.object(core.Guider_Basic, "sample", fake_sample), \
             patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, samples: samples), \
             patch.object(viggle.latent_preview, "prepare_callback", lambda *args: None):
            cold = run()
            self.assertEqual(len(calls), 3)
            self.assertGreater(calls[1][1].abs().sum().item(), 0)
            self.assertTrue(torch.equal(cold, run()))
            self.assertEqual(len(calls), 3)
            suffix_cached = run(2, 123)
            self.assertEqual([x[0] for x in calls], [10, 11, 12, 123, 12])
            viggle._CHUNK_CACHE.clear()
            suffix_fresh = run(2, 123)
            self.assertTrue(torch.equal(suffix_cached, suffix_fresh))

    def test_frame_schedule_coverage(self):
        for total in range(5, 1800, 17):
            for chunk in (39, 56, 124, 243, 362):
                for overlap in (5, 22, 39, 124, 362):
                    spans = viggle.plan_spans(total, chunk, overlap)
                    self.assertEqual((spans[0][0], spans[-1][1]), (0, total - 1))
                    for a, b, start, length in spans:
                        self.assertEqual((a % 17, (b - a + 1) % 17, start % 5), (0, 5, 0))
                    for left, right in zip(spans, spans[1:]):
                        self.assertLessEqual(right[0], left[1] + 1)

    def test_invalid_sigmas_stop_before_sampling(self):
        for values in ([1, 0.89, 0.72, 0, 0], [1, float("nan"), 0],
                       [1, float("inf"), 0], [1, -0.1, 0]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "interpolate_to_steps to 3"):
                self.node.sample(self.noise, self.guider, self.sampler, torch.tensor(values),
                                 {}, DecodeVAE(), 0, 0, 0)
        self.assertFalse(viggle._CHUNK_CACHE)

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
                self.node.sample(self.noise, self.guider, self.sampler, self.sigmas,
                                 cond_set, DecodeVAE(), 10, 0, 0)
            self.assertEqual(sample.call_count, 1)
            self.assertFalse(viggle._CHUNK_CACHE)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
