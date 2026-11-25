# Findings — DFlash2 speculative decoding on RustingStrata (Qwen3.8-Flash-Next, 131k ctx)

Machine: B-01, Windows 11, RTX 4070 SUPER 12 GB, Ryzen 9 7900X3D, 32 GB RAM, model on a local fixed drive.
Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (37.6 GB, qwen4exp A3B MoE, 48 layers, 512 experts, 10 active).
Unless stated: ctx 131072 / pp 98304 / tg 128 / reps 3 / q8 KV / chunk 4096 / CUDA. All claims carry their recorded command line and binary SHA-256 in the artifacts.

## 1. The Windows toolchain cliff

Local builds on a Windows host without Visual Studio default to the GNU toolchain (rustup auto-selects it). GNU codegen of the host-expert matvec loop is markedly slower:

| arm (v1.3.1 source, 131k) | decode t/s | worker busy |
|---|--:|--:|
| upstream release binary (MSVC) | 34.4 (31.4-35.9) | 16.8 ms |
| local GNU, target-cpu=native | 21.8 | ~32 ms |
| local GNU, target-cpu=x86-64-v3 | 23.0 (21.6-23.8) | 31.3 ms |
| local MSVC, target-cpu=x86-64-v3 | 33.7 (30.4-35.5) | 17.4 ms |

`target-cpu` is not the cause (GNU v3 still 23.0); the toolchain is. MSVC build reproduces the release binary within noise. Worker-busy ratio GNU/MSVC = 31,290/17,414 us = 1.80x. CUDA needs no build step (NVRTC runtime compilation). Build recipe in the README.

Consequence: every measurement taken on a GNU local build is ~32% low. Ours were. The "+88%" headline from our first matched A/B was an artifact of both arms being on the broken toolchain; corrected below.

## 2. The hc-gather fix (the PR)

DFlash2 heads loaded through the standalone draft path accepted 0% of drafts. Three independent data problems produced that one symptom; this fork's patch fixes the gather logic (the engine-side one):

1. hc-gather logic in gpu_model (this patch)
2. 1-based vs 0-based target_layers in the upstream dflash loader (converter-side; heads must ship `dflash.target_layers = [5,15,25,35,45]` 0-based for this model)
3. remapper key alignment (converter-side)

After the patch, on v1.3.1 + MSVC + CUDA, same-build matched pair:

| arm | decode t/s | acceptance |
|---|--:|--:|
| no-draft (kv q8) | 33.7 (30.4-35.5) | - |
| DFlash2 Q8_0 head (kv q8) | 56.3 (50.5-58.7, 5 reps) | 480/480 |

Bind run on the rebuilt binary from the pushed commit (exe SHA-256 11f547b4...); the pre-rebase build measured 56.2 (49.0-59.0) - within noise.

`--kv q4` adds +8.5% on top for spec runs (62.2, 55.4-65.8). It is a spec-only lever at this depth: the no-draft q4 control thrashes at 131k (expert-cache residency collapse; a 3-rep arm produced no completed rep in 108 minutes; a 32k probe runs fine at 29.9 t/s). `--spec-draft-n-max` does not reach the DFlash2 path (it populates only the ngram config; verified in source, main.rs:501-517) - deltas attributed to it are noise.

Head provenance: GGUF SHA-256 c00080c5..., source safetensors d6723f2a... (incoai/Qwen3.8-Flash-Next-DFlash2), converted with the in-repo converter (STRATA_DFLASH_Q8=1), config confirms 0-based target_layers [5,15,25,35,45], block_size 8.

## 3. Synthetic vs real prompts (the important caveat)

The bench numbers above use synthetic prefill prompts. On real text the picture changes materially:

| arm (real ~90k-token prompt, 131k ctx) | decode | expert hit | resident experts |
|---|--:|--:|--:|
| no-draft | 14.6 t/s | 30.3% | 2,960 |
| DFlash2 | 11.4 t/s (-22%) | 16.7% | 1,104 |
| MTP(4) | 18.6 t/s (+27%) | 26.9% | 2,254 |

Synthetic prompts route experts uniformly, so the cache pins a hot set (65% hit) and spec decode wins big. Real diverse text does not: hit rates collapse, and the DFlash2 head's own VRAM cuts resident experts from 2,960 to 1,104, halving the hit rate again. On real text at working depth the only positive lever we measured is MTP; DFlash2 is a net negative. Real-prompt prefill also collapses (89 t/s vs 677 on synthetic).

Single prompt, single run, n=1 - treat as directional until replicated. But the direction is consistent with the mechanism, and every headline number in section 2 should be read as "synthetic-prompt regime".

## 4. Output identity

Speculative decoding on this engine does not reproduce token-by-token greedy output byte-for-byte. Elimination evidence: no-draft is stable within a backend; DFlash2 diverges on both CUDA and Vulkan (not the CUDA batched kernels); native MTP(4), which touches none of the hc-gather code, diverges too. Consistent with batched-vs-scalar accumulation differences flipping near-tie argmax decisions; we did not measure logit margins, so we claim consistency, not proof. All observed divergent outputs were coherent, on-task continuations. Note: v1.3.1's `generate --help` says "greedy output unchanged" for --mtp; our counterexample contradicts that claim at depth (27-token prompt, generate).

Acceptance numbers need care: bench arms measure drafts made/accepted over synthetic prompts (100% here); the real-prompt identity runs measured 69.0% (DFlash2) and 82.8% (MTP). Always report which.

## 5. Open items

- Immediate-EOS: `generate` on ~42k+ token DIVERSE prompts emits EOS at step 1 (exit 0, gen 0) on all paths including no-draft, while the same prompt tail at 1.5k tokens and a repetitive 90k prompt generate normally. Unresolved; suspects are model behavior on this content at depth or generate budget state tied to the shared calibration file (which every run rewrites with its own ctx - check `strata-*.json`).
- `--spec-type ngram-*` hybrid cascade on top of DFlash2: untested.
- `--threads`, `--chunk`: untested.
- The `bbd57c2` "regression" we suspected earlier was retracted: it was the GNU/Vulkan toolchain confound, not the commit.

## 6. Error history (kept deliberately)

- Toolchain confound (section 1) inflated our first spec-decode headline to +88%; corrected to +67% matched-KV after the MSVC discriminator.
- A calibration profile written without the drafter attached misconfigures DFlash2 runs (decode 40.1 -> 34.8, prefill 677 -> 41.5). Do not mix calibration profiles between arms.
- Two benchmark runs were contaminated by an orphaned GPU process that survived its parent's kill; always verify the GPU is empty before an arm.
- The 100%-acceptance figure covered only synthetic bench prompts; the first real-prompt identity check accidentally used a 27-token file-path string as the prompt ("--prompt <path>" passes the path as text), which is why the deep reruns exist.

## 7. Reproduction

`bench/run_s7_chain.sh`, `bench/run_s7b_chain.sh`, `bench/run_s8_bind.sh` record every command. Artifacts (binary hashes, calibration snapshots, provenance) ship in the repo under `artifacts/`.
