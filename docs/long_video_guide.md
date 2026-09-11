# Long video generation — workflow and node guide

[README](../README.md) · [中文指南](long_video_guide_zh.md)

This guide covers the
long-clip path for Viggle-Animate: a driving clip of any length in, per-chunk
sampling with the finetune's evaluated 124-frame window, previous output carried
through five decoded/re-encoded anchor frames, plus one final decode.

Two ways to run it:

| | Single-pass sampler | Loop nodes |
|---|---|---|
| Viggle nodes, excluding shared text loader | 2 (Windowed Conditioning + Chunked Sampler) | 5 (Windowed Conditioning + Start / Sample Chunk / Loop End / Assemble) |
| VAE | Sampler decodes the assembled latent itself | Yours — VAE Decode sits in the loop body |
| Per-chunk output | Nothing until the whole clip finishes | Each chunk is decoded/saved before the next one samples |
| Failure mid-run | Everything after restart is resampled | Finished chunks are on disk and restore on re-queue |
| Rerender one chunk | `rerender_chunk` / `rerender_seed`, in-memory cache | Same inputs, but reuse is durable disk checkpoints |

```text
Single-pass:  Windowed Conditioning ──cond_set──▶ Chunked Sampler ──frames──▶ save

Loop:  Start ──state──▶ Sample Chunk ──video_latent──▶ VAE Decode ─┬─▶ Video Combine ──filenames──┐
            ▲                          └────────────── images ──────────────────────── Loop End
            └────────────────────── chunk/state ───────────────────────────────────────────┘
                                                                        Loop End ──chunks──▶ Assemble ──▶ final VAE Decode ──▶ save
```

**Which one should I use?**

- **Single-pass** when the run is short enough to lose: a 2–4 chunk clip, you just
  want the video out the end, and you prefer the smaller graph (2 nodes, one
  decode, no disk writes, nothing to clean up). Chunk reuse still works, but only
  in memory — restart ComfyUI and it's gone.
- **Loop** when the run is long or expensive (many chunks, hours of sampling), you
  want each chunk decoded/saved while it renders, you need crash-resume, you want
  to keep several takes of a chunk on disk, or you want your own encoder settings
  (Video Combine format/audio) applied per chunk. Costs a 4-node graph and a
  checkpoint folder per run.

---

## Viggle-Animate Conditioning (H3, Windowed)

`ViggleAnimateConditioningWindowed` — the planner. It never samples; it decides
where the chunks are and what each of them conditions on.

**Inputs**

| Input | Meaning |
|---|---|
| `cond_video` | The whole driving clip at 24 fps. Every loaded reference frame is retained; off-grid lengths generate up to 16 additional frames to reach the next `17k+5` boundary (minimum 5). No copied output frames are appended. |
| `ref_image` | Reference still shared by every chunk. A repainted frame from the driving shot with matching pose/framing conditions best. |
| `text_cond` | From the Load Text Conditioning node. |
| `vae` | MiniMax-H3 video VAE (the base model's). Used to encode each window's footage and the still. |
| `width` / `height` | `0` = the driving clip's own size (the evaluated configuration). Any other value rescales the canvas for every chunk. |
| `chunk_frames` | Maximum window length (default 124; minimum configurable 22). Short clips use fewer frames; longer clips keep full windows, including the final window. |
| `continuation` | `five_frame_anchor` (default): five decoded/re-encoded frames, two pinned latents, 119-frame stride with 124-frame windows. `latent_overlap`: previous raw-latent carry for comparison. |
| `overlap_frames` | Used only in `latent_overlap` mode. Frames carried from the preceding window and pinned (default 22). Clamped below the window length so every new window makes progress. The final window shifts back to end at the generation boundary, increasing its overlap. |

**Outputs**

| Slot | Name | Type | Goes to |
|---|---|---|---|
| 0 | `cond_set` | `VIGGLE_COND_SET` | Sampler / Loop Start |
| 1 | `guider_positive` | `CONDITIONING` | BasicGuider's `conditioning` (or CFGGuider's positive) |

`guider_positive` exists only so the guider's required socket has a valid source —
the samplers replace it per chunk with that chunk's own conditioning. Connect it
anyway; it is what makes the graph validate.

**Notes**

- Windows are `17j+5` frames (124 = 7·17+5). The first window starts at frame 0,
  the last ends at the last frame; every chunk's own footage becomes its video
  reference — chunk 3 is not conditioned on frames it cannot see.
- With the standard ComfyUI H3 VAE, complete 17-frame encoder blocks are reused
  from the preceding window's encoding; unseen blocks and each window's padded tail
  are encoded fresh. The log reports how many blocks were reused. Custom VAE wrappers get the
  full-window path.
- The log line also prints the full window plan with 1-based chunk numbers.

In `five_frame_anchor` mode, 322 source frames use windows **0–123, 119–242, 204–327**
on a 328-frame generation extent. The final overlap is larger, but only two latents
are pinned; already accepted output is preserved. The Chunked Sampler trims to 322 frames.

In `latent_overlap` mode, for 362 loaded frames with 124-frame windows and 22-frame overlap, expect
**0–123, 102–225, 204–327, 238–361**. Every window is **124 frames**.
For 345 frames, the final window is **221–344**.
Inputs under 124 frames use a single shorter window on the same grid.

For 361 loaded frames, generation extends to 362, while conditioning receives all
361 original frames plus one repeated final reference frame to match the generation
grid before VAE encoding. This keeps reference and target temporal latent counts aligned.
The repeated frame is conditioning padding, not an appended output frame.
The Chunked Sampler trims the decoded padding back to 361 frames. The loop workflow
returns latents and its external decode still includes grid padding.
This preserves the loaded tail, but cannot recover a source frame lost by the loader.

---

## Viggle Chunked Sampler (single-pass)

Its read-only live_progress box shows each chunk sampling or being reused from memory, then final decoding and completion. The chunk_map output remains the detailed report. This sampler does not save disk checkpoints.

`ViggleChunkedSampler` — samples every chunk inside one node, carries the
previous chunk's output through a five-frame VAE anchor (or raw overlap in
`latent_overlap` mode), assembles the master latent, and decodes at the end.

**Inputs**

| Input | Meaning |
|---|---|
| `guider` / `sampler` / `sigmas` | Your existing sampling stack (SamplerCustomAdvanced-style sockets). Stock objects support chunk reuse; opaque custom ones bypass it and rerender everything. |
| `cond_set` | From Windowed Conditioning. |
| `vae` | Decodes/re-encodes continuation anchors and decodes the assembled video. |
| `seed` | Base seed: chunk 1 uses `seed`, chunk 2 `seed + 1`, and so on. Standard noise is generated internally; remove old noise connections when updating. |
| `rerender_chunk` | 1-based chunk whose seed you want to override; `0` disables the override. |
| `rerender_seed` | Replacement seed for that chunk only. Zero itself is a valid seed. |

**Outputs**

| Slot | Name | Type |
|---|---|---|
| 0 | `frames` | `IMAGE` — the finished clip, decoded once |
| 1 | `chunk_map` | `STRING` — per-chunk report: frame range, seconds, effective seed, carry, `[cached]`/`[rendered]` |

**How carry works.** In `five_frame_anchor`, five decoded frames at the next
window start are re-encoded and only their two H3 temporal latents are pinned.
The rest of the new window starts empty; previously accepted positions are never
overwritten during assembly. In `latent_overlap`, the whole overlapping latent
section is copied and pinned. In both modes,
chunks are chained: rerendering chunk k changes what k+1..end receive, so those
resample; 1..k−1 come from cache. Changing a chunk's seed therefore never leaves
later chunks independent of the change. Caching preserves output precision and is
limited to 2 GiB of CPU tensor storage; restarting ComfyUI clears it.

---

## Viggle Chunk Loop Start

`ViggleChunkLoopStart` — reads the plan and opens the loop.

| Input | Meaning |
|---|---|
| `cond_set` | From Windowed Conditioning. Its window count is the loop count — you never set it manually. |
| `run_name` | Automatically creates `output/viggle_chunks/<run_name>/` if missing. 1–64 letters/digits/`_`/`-`; avoid reserved Windows names. Use a new name for a new take; reuse it to resume. |
| `resume` | On queue, restore matching saved chunks instead of resampling them (matching = the checkpoint fingerprint still agrees — see below). |
| `initial_state` *(optional, link-only)* | Do not set by hand — Loop End wires the previous iteration's state here when it expands. |

**Outputs:** 0 `loop` (`VIGGLE_LOOP`) · 1 `state` (`VIGGLE_LOOP_STATE`) · 2 `status` (`STRING`, "Chunk 1 of N").

Connect `loop` directly into Loop End's `loop` — the pair is found through that wire.

---

## Viggle Sample Chunk

`ViggleSampleChunk` — samples exactly one window. No VAE input: the chunk's
latent is checkpointed to disk **before** this node returns, and decoding is
your business in the loop body.

| Input | Meaning |
|---|---|
| `state` | From Loop Start's `state` (slot 1). |
| `guider` / `sampler` / `sigmas` | Your sampling stack. Connect the H3 VAE to Sample Chunk for `five_frame_anchor` mode. The updated
example includes this connection; older loop workflows need this additional wire.

Sample Chunk creates standard random noise internally from its seed; no noise connection is needed. |
| `seed` | Base seed; chunk N uses `seed + N − 1`. |
| `rerender_chunk` / `rerender_seed` | Same semantics as single-pass: override one chunk's seed; that chunk and everything after resamples, everything before restores from disk. |

**Outputs**

| Slot | Name | Type | Goes to |
|---|---|---|---|
| 0 | `chunk` | `VIGGLE_LOOP_STATE` | Loop End's `chunk` |
| 1 | `video_latent` | `LATENT` | VAE Decode in the loop body |
| 2 | `filename_prefix` | `STRING` | Handy as the save node's filename prefix (contains run + chunk number) |

Slot 2 is the filename prefix. Sample Chunk shows status directly in its live display; it has no status output port.

Sample Chunk also has a read-only **live_progress** display. It updates the same
visible node across loop iterations: sampling/restoring → checkpoint saved/restored
→ decoding/saving → loop completed. No Show Text connection is required for this
display. Loop completion does not mean the downstream final decode/save is finished.
Restart ComfyUI and refresh the browser after installing the frontend extension.
The STRING status outputs remain ordinary execution outputs, not live displays.

Each iteration writes `chunk_NNNN_<fingerprint>.latent` plus an updated
`manifest.json` atomically before returning. The fingerprint covers the sampling
graph (loaders' files, node code), this chunk's conditioning, the sigmas, the
window, canvas, effective seed and the predecessor chunk's fingerprint — so a
changed setting silently becomes a new checkpoint file and old takes stay on disk.

Connect the H3 VAE to Sample Chunk for `five_frame_anchor` mode. The updated
example includes this connection; older loop workflows need this additional wire.

Sample Chunk creates standard random noise internally from the effective chunk seed.
It has no external noise input. When updating an older workflow, remove its old noise wire.
For linked loader filenames, the fingerprint conservatively includes file paths,
sizes and modification times across that model category; replacing another file in
the same category can therefore also invalidate reuse. Weight contents are not hashed.
Code updates invalidate automatic resume too; existing latent files remain readable.

---

## Viggle Chunk Loop End

`ViggleChunkLoopEnd` — the loop's gatekeeper and the execution barrier.

| Input | Meaning |
|---|---|
| `loop` | Directly from Loop Start (slot 0). Raw link — it identifies the loop to expand. |
| `chunk` | From Sample Chunk (slot 0). Carries the state to the next iteration. |
| `images` | The decoded chunk — VAE Decode's output. This wire is what forces decode to finish before anything else happens. |
| `after_save` *(optional)* | Any downstream completion token, e.g. Video Combine's `filenames` output. Guarantees the save finished before the next chunk starts. |

**Outputs:** 0 `chunks` (`VIGGLE_CHUNKS`) · 1 `status` (`STRING`).

If chunks remain, it clones the whole loop body (every node between Start and End,
including your decode/save branch) as the next iteration and feeds this chunk's
state into the clone's Loop Start. When the last chunk is done it returns the
collection instead. It is an OUTPUT_NODE — that is normal; the collection still
feeds Assemble. It also refuses to continue if the decoded chunk contains NaN/Inf
(its latent checkpoint is already saved, so a matching re-queue retries decoding
without resampling). Replaying a matching saved prefix preserves the longer
manifest, even if a preview fails; a changed chunk starts a new manifest suffix.

---

## Viggle Assemble Chunk Latents

`ViggleAssembleChunkLatents` — stitches saved chunks into one latent, trimming
the overlap so later windows only contribute their new frames.

| Input | Meaning |
|---|---|
| `chunks` | From Loop End (slot 0). Leave unlinked to load from disk instead. |
| `run_name` | Used when `chunks` is unlinked: reads `output/viggle_chunks/<run_name>/manifest.json` — including **interrupted** runs. |
| `chunk_number` | `0` = assemble all completed chunks (the final video). `> 0` = load exactly that one chunk, overlap included, for inspection or a single-chunk decode. |

**Outputs:** 0 `video_latent` (`LATENT`) → final VAE Decode · 1 `status` (`STRING`, "Complete/Partial: N chunks, F frames").

The chain of checkpoints is validated during assembly (each chunk must continue
its predecessor's fingerprint). A run killed mid-way assembles partially — you
get frames up to the last finished chunk.

---

## Typical session

1. Queue. Chunks 1..N sample, each decoded/saved as it completes.
2. Crash / cancel / OOM at chunk 3? Fix the cause, queue again with the same
   `run_name` and `resume` on: saved matching chunks restore and decode again.
   If chunk 3 had already checkpointed before its decode/save failed, it restores too.
3. Don't like chunk 2's motion? `rerender_chunk = 2`, `rerender_seed = <new>`,
   queue: chunk 1 restores, 2..N resample with the new carry.
4. Happy with the take? Assemble (already in the graph) → final decode → save.
   Delete the run folder when you're done with it; old `rerender` checkpoints
   accumulate there.

Rules of thumb: keep input and output at 24 fps; keep the sigma/guider/sampler
stack unchanged between queues you want resumed; scene cuts belong in separate
runs with appropriate references. Joins can still show motion/identity changes —
overlap carry preserves content, it does not force seamless motion.

ComfyUI's execution cache may retain each decoded chunk until the queue finishes.
Chunked decoding does not guarantee constant host RAM: one 124-frame 1024×576
float32 RGB preview is about 0.82 GiB. The loop does not clear ComfyUI's shared cache.
For durable preview videos, enable your save node's `save_output`; temporary
previews are separate from the durable latent checkpoints.

## First long-generation setup

Use the [loop example](../example_workflows/viggle-animate-h3_workflow-long-video-advanced.json)
for recovery and per-chunk previews, or the [single-pass example](../example_workflows/viggle-animate-h3_workflow-chunked-sampler.json)
for a compact workflow. The loop example uses VHS video nodes; install VideoHelperSuite
and any other missing nodes reported when loading the example.

1. Load the desired driving clip at 24 fps (`force_rate = 24`; VHS `frame_load_cap = 0`
   loads the whole clip). Choose a matching reference still and the model/VAE/text files
   described in the README. Start with 124-frame windows and `continuation = five_frame_anchor`.
2. Connect Windowed Conditioning's `cond_set` to Start and `guider_positive` to the guider.
   Keep one Start, one Sample Chunk and one End; the window count controls repetition.
3. Connect Start `state` → Sample Chunk `state`, Start `loop` → End `loop`, and
   Sample Chunk `chunk` → End `chunk`. Leave Start `initial_state` disconnected.
4. Connect Sample Chunk `video_latent` → VAE Decode → End `images`.
   To save each preview, also feed those images to Video Combine, connect its
   `filenames` → End `after_save`, and Sample Chunk `filename_prefix` → Video Combine's
   filename prefix (convert that widget to an input if needed).
5. Enable Video Combine `save_output` for durable videos; the supplied example uses
   temporary previews by default. Latent checkpoints are always written independently.
6. Connect End `chunks` → Assemble `chunks`, set `chunk_number = 0`, and connect its
   LATENT → final VAE Decode → final save node. Save at 24 fps. Use the driving audio
   for the final video and trim it to the retained frame count; chunk previews contain
   overlap, so concatenating preview files is not the final assembly method.
7. Set a new `run_name`, `resume = true`, a fixed base `seed`, and `rerender_chunk = 0`.
   The supplied loop workflow has seed control set to `randomize`; change it to
   `fixed` before testing cancellation/resume, or each queue gets a different seed.
   Choose the [fast, balanced or quality-focused sigma presets](../README.md#custom-sigma-presets-4-6-or-8-points).
   Use **4 sigma points / 3 Euler updates for speed**, **6 points / 5 updates for
   balance**, or **8 points / 7 updates for quality (may over-sharpen)**. With
   BasicGuider / CFG 1.0, each Euler update is one model forward pass. Set KJNodes
   `interpolate_to_steps` to **3, 5 or 7**, respectively, and retain the final zero.
   Conditioning/VAE encoding happens before chunk sampling, so the live box may still
   say “Waiting” while that work runs.

## Where status appears

| Display/output | Purpose |
|---|---|
| Sample Chunk `live_progress` | Updates the same visible sampler through sampling/restoring, decode/save and loop completion; no wire required |
| Start `status` | Ordinary STRING describing that iteration's chunk number |
| Sample Chunk `filename_prefix` | A save filename, not a status display; includes run name, chunk index and fingerprint |
| End `status` | Loop completion and checkpoint location; final downstream decoding can still be running |
| Assemble `status` | Complete/partial chain and frame count, or selected single-chunk information |

Connect STRING outputs to Show Text when useful. A Show Text attached to an expanded
loop may show only one iteration; use `live_progress` for live updates. Progress is
sent to the browser that queued the run; it is not a persistent log recovered on refresh.

## Quick recovery test and common questions

Try a 226-frame clip at the default settings (two windows).
Cancel while chunk 2 is sampling, then requeue unchanged with `resume` on.
Chunk 1 restores and decodes again; chunk 2 samples unless its checkpoint was already
saved. After completion, queue unchanged to check that both restore. Then change
`rerender_chunk = 2` and `rerender_seed` to test that only the suffix resamples.

- **New run folder?** Start creates it automatically. Assemble only reads an existing
  folder when `chunks` is disconnected. Use a separate name per project/take and avoid
  concurrent runs writing the same folder.
- **Recover without rerunning the loop?** Disconnect Assemble `chunks`, enter the saved
  `run_name`, and use `chunk_number = 0` for all manifested chunks or a positive number
  for one chunk. Missing/invalid checkpoints produce an error instead of silent substitution.
- **Why did everything resample?** Check `resume`, run name, base seed, conditioning,
  sampling settings and model files. Code updates also change checkpoint fingerprints.
- **Does rerender repaint the whole chunk?** Its five-frame anchor stays pinned (the full overlap in `latent_overlap` mode). Changing
  chunk k changes the carry into k+1 onward; unchanged overrides can reuse a matching take.
- **Why does the end deform or stutter?** Carry does not guarantee identity or seamless
  motion. Review the driving motion/reference and individual chunks. The final window
  now stays full length; test motion and identity quality locally. More sigma points do not guarantee an improvement.
- **Why is memory/encoding still high?** The full input and conditioning still occupy memory,
  the execution cache can retain decoded previews, and final assembly needs a full latent
  and decode. Use shorter shots or a smaller canvas when those stages exceed available memory.
