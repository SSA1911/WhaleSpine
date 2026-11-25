# Findings — standalone DFlash2 draft heads on RustingStrata, at 131k context

## What this is about

[RustingStrata](https://github.com/RustingStudio/RustingStrata) is a pure-Rust inference engine for GGUF models, built for mixture-of-experts (MoE) models that do not fit in VRAM. In an MoE model, each layer holds many small "expert" networks and routes every token to a few of them; the ones that do not fit in VRAM run on the CPU while the GPU works, which is why speed depends on memory behaviour as much as on compute.

This document records an investigation into **speculative decoding** on RustingStrata v1.3.1 with **Qwen3.8-Flash-Next** at long context, and a bug fix it produced (upstream PR: *dflash: fix draft-feature capture on hyper-connection models*).

Some terms used throughout:

- **Speculative decoding**: a small "drafter" model guesses several tokens ahead; the big model checks all of them in one pass and keeps the ones it agrees with. Faster when the drafter guesses well.
- **DFlash2**: a family of small standalone drafter models, published per target model on Hugging Face. Loaded in RustingStrata with `--draft-model`.
- **MTP**: multi-token prediction heads built into the target model itself — RustingStrata's other speculative path (`--mtp`).
- **Acceptance rate**: the fraction of drafter tokens the target's greedy choice agrees with. The bench `--json-out` records it as `drafts {made, accepted}`.
- **Hyper-connections (hc)**: in qwen4exp-family models (Qwen3.8-Flash-Next among them), each position's hidden state is carried as several parallel variants ("streams") that are summed later in the layer. RustingStrata's GPU forward keeps them interleaved in one buffer. The bug behind the PR is that the draft-feature capture read a single stream instead of the streams' mean.
- **Expert hit / resident**: how many of the routed experts were already in VRAM ("hit") and how many fit at once ("resident"). Misses stream over PCIe.
- **Worker busy**: microseconds per token the CPU worker spends streaming and running experts. Lower is better.

## Environment (every number below comes from this setup)

| | |
|---|---|
| OS | Windows 11 Pro 26H2 (build 10.0.26300) |
| GPU | NVIDIA RTX 4070 SUPER 12 GB, driver 617.14 |
| CPU / RAM | AMD Ryzen 9 7900X3D, 32 GB DDR5 |
| Model storage | local fixed drive (Z:) |
| Engine | RustingStrata (upstream `main`): sections 1–3 measured on v1.3.1 @ `4b3f44b`; the PR and the serving comparison (section 3.5) re-run on v1.3.4 @ `a02a259`, which also fixed Windows working-set accounting relevant to expert streaming. CUDA kernels compiled at run time by NVRTC — no CUDA build step |
| Compiler | rustc 1.100.0 (0ef1bc386, 2026-11-09), LLVM 23.1.5; MSVC build via `--target x86_64-pc-windows-msvc`, `RUSTFLAGS="-C target-cpu=x86-64-v3"` |
| Model | Qwen3.8-Flash-Next GSQ-RCO Q2_0 — 66.4 GB across two shards (61.9 GiB), qwen4exp A3B MoE, 48 layers, 512 experts, 10 active per token |
| Head | incoai/Qwen3.8-Flash-Next-DFlash2, safetensors `d6723f2a…`, converted with the in-repo converter, GGUF SHA-256 `c00080c5…` |

The model is larger than system RAM, so non-resident experts page from disk and page-cache state is a live variable — the reason the headline measurement is an interleaved same-session pair (section 2).

Unless stated, every benchmark is: context 131,072 · prefill prompt 98,304 tokens · 128 generated tokens · KV cache q8 · chunk 4096 · CUDA backend · greedy decoding.

## Approach

1. Baseline the upstream release binary with no drafting.
2. Reproduce a local-build gap, isolate its cause (backend → CPU flags → toolchain), because every later number depends on a trusted local build.
3. Fix the draft-head capture bug so the DFlash2 path works at all.
4. Measure the fix with a same-session interleaved A/B (alternating single-rep runs of both arms, so page cache and profile state are shared).
5. Sweep the flags that plausibly interact (`--kv`, `--spec-draft-n-max`).
6. Cross-check in generate mode (chat-style output, not a bench continuation) — which changed the practical picture (section 3).

Every claim carries its recorded command line, JSON output and binary SHA-256; the harness scripts and evidence logs ship in this repo (`bench/`, `artifacts/`).

## 1. The Windows toolchain cliff (fixed the approach, not the engine)

The first local build measured ~21 t/s against the release binary's 34.4 t/s — a 32% gap that turned out to be our build, not the engine. On a Windows host where Visual Studio was not yet installed, we had set the Rust toolchain to GNU manually (`rustup default stable-gnu`) before installing the VS C++ Build Tools; upstream's README already instructs installing the Build Tools first. Isolation:

| build (v1.3.1 source, 131k) | decode | CPU worker busy/token |
|---|--:|--:|
| upstream release binary (MSVC) | 34.4 (31.4–35.9) | 16.8 ms |
| local GNU, `target-cpu=native` | 21.8 | ~32 ms |
| local GNU, `target-cpu=x86-64-v3` | 23.0 (21.6–23.8) | 31.3 ms |
| local MSVC, `target-cpu=x86-64-v3` | 33.7 (30.4–35.5) | 17.4 ms |

`target-cpu` is not the cause (GNU at v3 is still 23.0). Both binaries came from the same rustc 1.100.0 / LLVM 23.1.5 — the MSVC exe is a cross-build with `--target` — so the difference is the target triple: C runtime, linker, runtime. CPU worker-busy time (1.80× worse on GNU) locates the gap in the worker path; the mechanism inside it is not isolated further. CUDA needs no build step either way (NVRTC compiles at run time).

## 2. The fix, and what it is worth

**The bug.** The drafter's input features are the target model's hidden states, captured at the head's `target_layers`. On hyper-connection targets the capture source holds the hc streams interleaved per position, but both capture sites copied with a single-stream stride, so capture row *r* read stream `r mod hc_n` of position `r / hc_n`. A drafter is trained to consume the streams' mean (llama.cpp fixed the identical bug the same way for this architecture, ggml-org/llama.cpp#30927) and never saw inputs it was trained on. Same defect in both the decode-step path (`src/gpu_model.rs`) and the prefill path (`src/gpu_model/prefill.rs`).

**The fix.** Collapse the hc streams to their plain mean first (RustingStrata's existing `hc_mix` with a saturated gate — sigmoid(12) ≈ 1 − 6·10⁻⁶, and the residual scale is absorbed by the following bias-free `fc` + `hidden_norm`), then copy. The prefill copy is tiled to `TMAX` blocks because `hc_mix`'s gate buffer indexes per position. Targets without hyper-connections take the unchanged path, and the mean scratch buffer is only allocated when needed. Two files, +18/−1 and +22/−2, no API or format changes. The patch's own prefill cost at 8k (reps 1): 125.8 → 109.9 tok/s (≈ −13%). The EAGLE3 capture paths (`gpu_model.rs:1975`, `prefill.rs:376`) carry the same single-stream copy with no hc guard — untouched here; flagged upstream as a follow-up.

**Before / after, measured honestly.** With a working head and a calibrated profile, unpatched v1.3.1 accepts **44.0%** of drafts on an 8k-context bench run (`DFlash drafts accepted 74/168`). With the patch: **100%** (`96/96`; 5-rep 131k run: `480/480`). (An earlier draft of this document claimed "0% before": those runs used a head we remapped by hand from a llama.cpp-format GGUF, with two conversion bugs in our script. Passing the Hugging Face directory to the engine and letting it convert natively was always correct.)

**Throughput, same-session interleaved pair** (both arms alternate rep-by-rep in one session on one binary, so page cache and profile state are shared):

| arm | decode | prefill | acceptance |
|---|--:|--:|--:|
| no draft (KV q8) | 30.8 t/s (30.1–31.7, 5 reps) | 583.3 t/s | – |
| DFlash2 Q8_0 head + this fix | 47.7 t/s (43.7–49.9, 5 reps) | 441.6 t/s (−24%) | 480/480 |

**+55% decode with this head at 131k context**, paid for with 24% slower prefill.

Two honesty notes. First, the same configurations measured in *different sessions* spread 50.0–56.3 t/s (≈12% drift) on the DFlash2 arm; cross-session A/Bs are why our first headline (+88%) and a later one (+67%) were both wrong in different directions — the interleaved pair is the number we stand behind. Second, `--kv q4` (a smaller KV cache) added ~+8.5% for speculative runs (61.0 t/s, 3 reps), but a no-draft q4 control at 131k thrashed (no completed rep in 108 minutes; a 32k probe ran fine at 29.9 t/s), so quote q4 as a speculative-only configuration, and note its mechanism is unresolved — the no-draft q4 control's plan print shows q4 *raises* resident experts (3,552 vs 2,960), while the spec q4 arm itself ran 2,160 resident (vs 1,104 at q8 spec) — the two arms run different memory plans, which is part of why the mechanism is hard to pin. `--spec-draft-n-max` does not reach the DFlash2 path at all (it populates only the ngram config; `main.rs:501–517`) — deltas attributed to it were noise.

**On the current base (v1.3.4).** The PR was rebased onto upstream v1.3.4 (`dflash-hc-mean` @ `32bde88`, diff +40/−3) and the acceptance check re-run: unpatched stock v1.3.4 accepts **72/168 (42.9%)** on the same 8k-context bench run — the bug is still live upstream — and the patched build accepts **96/96 (100%)**. Patched 8k prefill 125.5 vs 125.8 unpatched (reps 1): the ≈−13% prefill cost measured on v1.3.1 does not show on v1.3.4 (single rep, treat as unmeasured). One engine fact worth recording for anyone extending this: the DFlash2 drafter's attention cache is allocated for the full context (`src/gpu_model.rs`, `kv_words(ctx * df_kv)`) although the head's attention window is 2,048 tokens (`sliding_window` in its config) — about 1.4 GB wasted at 131k context, part of why only 1,104 experts stay resident with the drafter attached.

## 3. The bench-vs-generate caveat (the practically important part)

`bench` decodes a raw greedy continuation of the prefill prompt; `generate` writes a chat-style reply. On our bench prompt (a repeated instruction file) those are near-copy tasks with a narrow, repeating expert set — which flatters speculative decoding.

One deep generate-mode run (~90k-token prompt, 1,700 repeats of a 239-byte instruction file, 131k context, all three arms):

| arm (generate mode) | decode | expert hit | resident experts |
|---|--:|--:|--:|
| no draft | 14.6 t/s | 30.3% | 2,960 |
| DFlash2 | 11.4 t/s (−22%) | 16.7% | 1,104 |
| MTP | 18.6 t/s (+27%) | 26.9% | 2,254 |

(The MTP arm requested 4 draft tokens; the engine verifies at most 3 draft tokens per step on every backend — `verify_rows()` is a constant 4, `src/backend.rs:488`, `k = k.min(verify_rows() - 1)`, `src/gpu_model.rs:715`. The same cap applies to DFlash2's default block of 7, so `--mtp N` is the actual draft-length knob for both.)

Readings from this run, all single-run/n=1:

- Bench decode overstates serving decode even *without* speculative decoding: bench-mode no-draft ran 33.7 t/s vs generate-mode 14.6 on comparable content. Whatever the bench table says about throughput, treat it as a bench-regime number.
- The DFlash2 arm lost in generate mode while winning in bench mode on the same file (bench arm: 52.6 t/s, 100% acceptance, 37.6% hit). What differs between the modes in our logs is acceptance (100% vs 69.0%) and hit rate (40.5% vs 16.7%), not residency — the bench DFlash2 arm also ran 1,104 resident experts and won. We cannot yet explain the regime boundary; the arms also ran different memory plans (2,960/2,254/1,104 resident), so CPU/GPU expert placement is a confound.
- On this measurement, MTP is the only positive generate-mode lever (+27%), tracking worker-busy amortization (33.6 vs 51.9 ms) rather than hit rate.
- This ranking is not stable across depth. Our identity-check runs — invalid as identity checks (a file path was passed as the prompt) but still valid shallow-depth throughput samples — flipped it: no-draft 17.43, DFlash2 18.70 (+7%), MTP 12.97 (−26%) on CUDA. Every arm here is n=1; treat both directions as unsettled until the deep diverse-text comparison is unblocked (section 5).

Do not generalize the generate-mode ranking from one prompt on repetitive content.

## 3.5 Serving-mode comparison (chat, real text, end-to-end — the decisive test)

The bench-vs-generate gap above motivated a comparison on the real serving path. Arms: no-draft, MTP with 1/2/3 drafts per step, DFlash2 with 1/2/3. Prompts: three depths (≈1.9k / 8.7k / 24.3k tokens) of genuinely mixed Rust source plus a summarizing instruction, sent through the OpenAI-compatible `serve` endpoint (chat template), 512-token reply budget, greedy. One request per freshly started server process, so the server's prefix cache cannot skip prefill between cells. Patched v1.3.4 build, CUDA, RTX 4070 SUPER.

| arm | 2k tok | 8k tok | 24k tok |
|---|--:|--:|--:|
| no draft | 58.8 s | 126.6 s | 282.5 s |
| MTP best (3 drafts) | 54.1 (−8%) | 118.0 (−7%) | 291.2 (+3%) |
| DFlash2 best (1 draft) | 54.5 (−7%) | 121.1 (−4%) | 317.0 (+12%) |

Single rep per cell; the full grid is in `bench/e2_cells.jsonl` (21 cells, every drafter length recorded). Reading:

- No drafter configuration beats no-draft beyond single-run noise at any depth. At 24k every drafter arm is *slower* than no-draft (DFlash2 worst, +6–12%): the drafter's VRAM cost and (for DFlash2) its prefill capture cost outweigh the drafting gain on real reasoning-style workloads.
- DFlash2 does not beat MTP at any depth. Per the decision rule fixed before the run — tuned DFlash2 must beat tuned MTP end-to-end on at least 2 of 3 prompts by more than run spread — the DFlash2 serving investigation **stops here**. A planned deeper-context (32k vs 131k) run was skipped as moot: at 24k the drafter's wasted full-context cache is already small and DFlash2 still loses, so the engine change (sizing the drafter cache to its 2,048-token window) cannot flip this verdict, though the ~1.4 GB waste at 131k remains a real memory issue worth reporting upstream on its own merits.
- MTP at 3 drafts is the only arm that trends positive (−8%/−7%), and only at shallow depth; at 24k it is +3%. Treat as "no measurable serving win" rather than a recommendation.

## 4. Output identity

Speculative decoding here does not reproduce undrafted greedy output byte-for-byte. Elimination evidence, all on the same ~90k-token prompt, temp 0: no-draft output is stable *within* a backend but forks *between* backends (CUDA vs Vulkan, first differing byte 24, 0-indexed) — the cleanest evidence that floating-point accumulation order alone flips near-tie argmax choices. DFlash2 diverges on both CUDA and Vulkan; the native MTP path, which this fix does not touch, also diverges on CUDA (fork at byte 67) — no Vulkan MTP run was made. The arms run different memory plans, so CPU/GPU expert placement is a further confound; we did not measure logit margins and claim consistency with near-tie flips, not a proven mechanism. Every divergent output we inspected was a coherent, on-task continuation.

Two practical notes follow. Acceptance percentages must always state their mode (bench arms on synthetic prompts here: 100%; the generate-mode identity runs: 69.0% for DFlash2, 82.8% for MTP). And v1.3.1's `generate --help` says "greedy output unchanged" for `--mtp` — if that is meant as *behavioral equivalence*, our data does not contradict it; if a reader takes it as bit-exactness, it does not hold (documented as a suggested help-text clarification, filed after the PR).

## 5. Open items

- **Immediate EOS at depth — resolved; not an engine bug.** `generate --prompt` (raw text, no chat template) on long code dumps made the model continue the pasted code (2k/4k of source) or end its turn immediately (≥8k; appending an instruction did not help; the boundary sits between 4k and 8k tokens of source text). Through the chat template (`serve`, OpenAI endpoint) the same 16k-token prompt is answered normally (`finish: stop`, correct content). Two things had made this look like a bug: raw-generate prompting gives the model nothing to reply to, and the model is a *reasoning* model — its reply text arrives in `reasoning_content`, so an empty `content` field only means the token budget ran out mid-reasoning. Deep diverse-text comparisons are unblocked via the chat path (section 3.5 used it).
- `--spec-type ngram-*` hybrid cascade on top of DFlash2: untested. `--threads`, `--chunk`: untested.
- Sampled (non-greedy) speculative acceptance: not measured; every number here is greedy.
- An earlier suspected regression in commit `bbd57c2` was retracted — it was the GNU-toolchain confound, not the commit.

## 6. Error history (kept deliberately)

| error | how it was caught | what changed |
|---|---|---|
| Local build 32% slower than the release binary | matched-pair vs release binary; toolchain ladder | build with the MSVC target; never compare builds across toolchains |
| First headline "+88%" | MSVC discriminator exposed both arms were GNU | corrected; later re-corrected to +55% by the interleaved pair |
| Cross-session drift (~12%) mistaken for signal | timestamps of the per-config runs | same-session interleaved A/B is now the standard |
| A calibration profile written without the drafter attached | prefill 364 → 41.5 t/s on the record | calibrate with the drafter attached; never mix profiles between arms |
| Two runs contaminated by an orphaned GPU process | GPU process list after a kill | verify the GPU is empty before every arm |
| A real-prompt identity check passed a file *path* as the prompt | prompt token count in the log (`--prompt <path>` sends the path as text) | `--prompt -` with stdin; check `N tok` before reading any other number |
| "0% → 100%" headline | clean before-arm on stock binary + correct head | honest claim: 44% → 100% |

## 7. Reproduction

Everything is scripted: `bench/repro_acceptance.sh` (the 8k before/after check), `bench/run_s7_chain.sh` / `run_s7b_chain.sh` (flag sweeps), `bench/run_s8_bind.sh` and `bench/run_s9_conditions.sh` (before arm, escape test, interleaved pair), `bench/e2_sweep.sh` (the serving-mode comparison of section 3.5, one fresh server per cell). `bench/logs/` ships the generate-mode evidence with local paths redacted; `artifacts/` ships binary SHA-256s, the rustc version, head provenance and a calibration snapshot.

`cargo test --release` (with `STRATA_MODEL` set): 112 passed, 1 failed. The failure (`standalone_external_draft_model_spec_step_matches_greedy`, Windows error 1224 "user-mapped section open") reproduces identically on unpatched v1.3.1 — a pre-existing Windows mmap flake in the test, not a change from the fix.

## Conclusion

The capture bug is real and the fix is small: standalone DFlash2 heads went from ~44% to 100% bench acceptance on a hyper-connection target, verified on both v1.3.1 and v1.3.4 (llama.cpp fixed the identical bug the same way). But in serving mode on real text (chat, end-to-end, section 3.5), no drafter configuration beat plain no-draft decoding beyond noise on this machine, and DFlash2 was the worst arm at depth — so the practical verdict for this hardware is: run no-draft, use the PR to make the DFlash2 path *correct* for those who evaluate it, and treat any bench-mode speedup as a bench-regime result. Whether speculative decoding pays off on other hardware or workloads is still workload-dependent; measure on the serving path before enabling a drafter.
