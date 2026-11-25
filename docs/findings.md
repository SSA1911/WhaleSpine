# Findings — DFlash2 speculative decoding on RustingStrata (Qwen3.8-Flash-Next, 131k ctx)

Machine: B-01, Windows 11, RTX 4070 SUPER 12 GB, Ryzen 9 7900X3D, 32 GB RAM, model on a local fixed drive.
Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (66.4 GB total across two shards = 61.9 GiB, qwen4exp A3B MoE, 48 layers, 512 experts, 10 active). Larger than system RAM (32 GB), so non-resident experts page from disk and page-cache state is a live variable - hence the interleaved pairs in section 2.
Unless stated: ctx 131072 / pp 98304 / tg 128 / reps 3 / q8 KV / chunk 4096 / CUDA. All claims carry their recorded command line and binary SHA-256 in the artifacts.

## 1. The Windows toolchain cliff

Our host toolchain was GNU (set manually via `rustup default stable-gnu` before Visual Studio was installed; upstream's README already instructs installing the VS C++ Build Tools first). The worker path on that toolchain is markedly slower:

| arm (v1.3.1 source, 131k) | decode t/s | worker busy |
|---|--:|--:|
| upstream release binary (MSVC) | 34.4 (31.4-35.9) | 16.8 ms |
| local GNU, target-cpu=native | 21.8 | ~32 ms |
| local GNU, target-cpu=x86-64-v3 | 23.0 (21.6-23.8) | 31.3 ms |
| local MSVC, target-cpu=x86-64-v3 | 33.7 (30.4-35.5) | 17.4 ms |

`target-cpu` is not the cause (GNU v3 still 23.0). Both binaries came from the same rustc 1.100.0 / LLVM 23.1.5 (host triple x86_64-pc-windows-gnu; the MSVC exe is a cross-build via `--target`), so only the target triple differs - C runtime, linker, runtime. Worker-busy time locates the gap in the worker path; the mechanism inside it is not isolated to codegen specifically. Worker-busy ratio GNU/MSVC = 31,290/17,414 us = 1.80x; decode ratio = 1.47x. CUDA needs no build step (NVRTC runtime compilation). Build recipe in the README.

Consequence: every measurement taken on a GNU local build is ~32% low. Ours were. The "+88%" headline from our first matched A/B was an artifact of both arms being on the broken toolchain; corrected below.

## 2. The hc-gather fix (the PR)

DFlash2 heads loaded through the standalone draft path accepted ~0% of drafts in our initial runs; the clean measurement (stock v1.3.1 release exe + converted head + calibrated profile, 8k smoke) gives 44.0%. Three stacked data problems produced that one symptom; this fork's patch fixes the engine-side one:

1. hc-gather logic in gpu_model (this patch)
2. metadata convention mismatch: incoai's downloadable GGUF is llama.cpp-format and carries 1-based `dflash.target_layers` while the engine indexes 0-based - a defect in OUR remap of that file (dflash_to_rs_remap.py), not in the in-repo converter, which produced the working c00080c5 head
3. data-offset alignment bug, also in our remap script (not 'key alignment')

After the patch, on v1.3.1 + MSVC + CUDA, same-build matched pair:

| arm | decode t/s | prefill t/s | acceptance |
|---|--:|--:|--:|
| no-draft (kv q8) | 30.8 (30.1-31.7, 5 reps) | 583.3 | - |
| DFlash2 Q8_0 head (kv q8) | 47.7 (43.7-49.9, 5 reps) | 441.6 (-24%) | 480/480 |

Same-session interleaved pair (rep-by-rep alternation, one binary, exe SHA-256 efaefe50..., commit 1728400): +55%. Cross-session runs of the same configs spread 50.0-56.3 (about 12% drift), which is why the interleaved pair is the headline. Earlier measurements: 33.7 no-draft (single 10:01 run), 50.0/56.2/56.3 DFlash2 across sessions.

`--kv q4` alone adds +8.5% for spec runs (61.0, 53.3-65.2). Stacked with other levers it measured 62.2 (55.4-65.8), but those deltas are unseparated at 3 reps. It is a spec-only lever at this depth: the no-draft q4 control thrashes at 131k (a 3-rep arm produced no completed rep in 108 minutes; a 32k probe runs fine at 29.9 t/s). Note the plan print shows q4 raises resident experts (3,552 vs 2,960 at q8), so the thrash mechanism is unresolved - do not attribute it to residency collapse. `--spec-draft-n-max` does not reach the DFlash2 path (it populates only the ngram config; verified in source, main.rs:501-517) - deltas attributed to it are noise.

Head provenance: GGUF SHA-256 c00080c5..., source safetensors d6723f2a... (incoai/Qwen3.8-Flash-Next-DFlash2), converted with the in-repo converter (STRATA_DFLASH_Q8=1), config confirms 0-based target_layers [5,15,25,35,45], block_size 8.

## 3. Bench-mode vs generate-mode (the important caveat)

Bench numbers above are synthetic. The picture changes in generate mode. One deep generate-mode run (a ~90k-token prompt built as 1,700 repeats of a 239-byte instruction file, 131k ctx):

| arm (generate mode, 131k ctx) | decode | expert hit | resident experts |
|---|--:|--:|--:|
| no-draft | 14.6 t/s | 30.3% | 2,960 |
| DFlash2 | 11.4 t/s (-22%) | 16.7% | 1,104 |
| MTP(4) | 18.6 t/s (+27%) | 26.9% | 2,254 |

Bench mode decodes a raw greedy continuation of the prefill prompt (main.rs bench loop), which on a repeated prompt is a near-copy task - a narrow, repeating expert set (65% hit), not a uniform distribution. A bench-mode DFlash2 arm on that same file scored 52.6 t/s / 100% acceptance (37.6% hit); no bench no-draft or MTP arm was run on that file, so there is no cross-mode ranking on identical content - the bench-vs-generate difference is itself the finding. What differs between the modes in our logs: acceptance (100% vs 69.0%) and hit rate (40.5% vs 16.7%), not residency (the bench DFlash2 arm also ran 1,104 resident experts and won). The arms run different memory plans (2,960 / 2,254 / 1,104 resident), so CPU/GPU expert placement is a confound. The MTP win tracks worker-busy amortization (33.6 vs 51.9 ms) rather than hit rate. Separately and arguably more important: generate-mode no-draft ran 14.6 t/s vs bench-mode 33.7 - bench decode overstates serving decode ~2.3x on this run even with no spec decoding. On this measurement the only positive generate-mode lever is MTP; DFlash2 is a net negative there. Prefill: the tiled-file prompt prefills at 573 (no-draft) vs 677 synthetic; genuinely diverse source text prefills far slower (89 t/s, -87%) but never produced decode output (section 5).

Single prompt, single run, n=1, repetitive content - treat as directional until replicated on genuinely diverse text (blocked by the EOS issue below). Every headline number in section 2 should be read as "synthetic-prompt bench regime".

## 4. Output identity

Speculative decoding on this engine does not reproduce token-by-token greedy output byte-for-byte. Elimination evidence: no-draft is stable within a backend but forks between backends (CUDA vs Vulkan first differing byte 24, 0-indexed) - the cleanest evidence for floating-point non-associativity. DFlash2 diverges on both CUDA and Vulkan; native MTP(4), which touches none of the hc-gather code, diverges on CUDA (fork byte 67 vs no-draft) - no Vulkan MTP run was made. Consistent with batched-vs-scalar accumulation differences flipping near-tie argmax decisions; we did not measure logit margins, so we claim consistency, not proof. All observed divergent outputs were coherent, on-task continuations. Note: v1.3.1's `generate --help` says "greedy output unchanged" for --mtp; on a real ~90k-token prompt the MTP(4) output diverged from no-draft within the first tokens (whole-output divergence, coherent on both sides). If the help text means behavioral equivalence rather than bit-identity, this is a documentation-accuracy point rather than a correctness bug.

Acceptance numbers need care: bench arms measure drafts made/accepted over synthetic prompts (100% here); the real-prompt identity runs measured 69.0% (DFlash2) and 82.8% (MTP). Always report which.

## 5. Open items

- Immediate-EOS: `generate` on ~42k+ token diverse prompts emits EOS at step 1 (exit 0, gen 0) on all paths including no-draft, while the same prompt tail at 1.5k tokens and a repetitive 90k prompt generate normally. Escape test executed: stripping all 9 `<|...|>` token-like literals (15 chat-role markers included) from the 42k prompt and rerunning no-draft at ctx 65536 still produced gen 0 - the literals are ruled out; the trigger is depth/content, not the chat markers. The shared-calibration-state suspect was also ruled out: the calibration JSON is not rewritten by runs (the .f32 expert profile is). Unresolved.
- `--spec-type ngram-*` hybrid cascade on top of DFlash2: untested.
- `--threads`, `--chunk`: untested.
- The `bbd57c2` "regression" we suspected earlier was retracted: it was the GNU/Vulkan toolchain confound, not the commit.

## 6. Error history (kept deliberately)

- Toolchain confound (section 1) inflated our first spec-decode headline to +88%; the MSVC discriminator cut it to +67%, and the same-session interleaved pair settled it at +55%. Retire cross-session numbers whenever drift exceeds the effect.
- A calibration profile written without the drafter attached misconfigures DFlash2 runs (source record: prefill 364 -> 41.5; the decode 40.1 -> 34.8 pair was on the GNU/Vulkan build). Do not mix calibration profiles between arms.
- Two benchmark runs were contaminated by an orphaned GPU process that survived its parent's kill; always verify the GPU is empty before an arm.
- The 100%-acceptance figure covered only synthetic bench prompts; the first real-prompt identity check accidentally used a 27-token file-path string as the prompt ("--prompt <path>" passes the path as text), which is why the deep reruns exist.

## 7. Reproduction

`bench/run_s7_chain.sh`, `bench/run_s7b_chain.sh`, `bench/run_s8_bind.sh`, `bench/run_s9_conditions.sh`, `bench/repro_acceptance.sh` record every command. `artifacts/` ships binary hashes + rustc version, head provenance and a calibration snapshot; `bench/logs/` ships the generate-mode evidence (paths redacted). The calibration snapshot predates the generate-mode runs; the expert profile is rewritten per run, so profile state is a live variable - the interleaved pairs control for it.
