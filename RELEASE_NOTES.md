# v1.3.2 — Fix long-video ending drift

Fix a final-window conditioning mismatch introduced by rounding source lengths up to the H3 frame grid. For example, a 289-frame source could supply 32 reference latents against 37 target latents in the last window. Padding the reference before VAE encoding aligns those lengths; the maintainer confirmed the fix on the affected clip.

- Restore full-length, end-aligned final windows.
- Add five-frame decoded/re-encoded continuation anchors, enabled by default. `latent_overlap` remains available for comparison.
- Preserve accepted output when final windows overlap earlier chunks.
- Trim the Chunked Sampler output to the loaded source frame count.
- Update example workflows, English/Chinese documentation, cache and checkpoint handling.

**After updating:** restart ComfyUI and refresh the browser. For older advanced loop workflows, connect the H3 VAE to Sample Chunk when using `five_frame_anchor`. Code updates invalidate automatic checkpoint reuse; saved latent files remain readable. The advanced loop's external decode still includes grid padding.

Validation: 42 Python regression tests covering reference encoding, anchor placement, assembly, caching and loop recovery. Motion quality can still vary by clip.
