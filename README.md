# WhaleSpine

A fork of [RustingStudio/RustingStrata](https://github.com/RustingStudio/RustingStrata) — a pure-Rust inference engine for GGUF models, built for mixture-of-experts (MoE) models that do not fit in VRAM (experts that do not fit run on the CPU while the GPU works). All credit for RustingStrata itself goes to Vasyl Trefilov; this fork carries the modifications below under the same [RustingStrata License 1.0](https://github.com/RustingStudio/RustingStrata/blob/main/LICENSE.md).

## What this fork changes

**1. A fix for standalone DFlash2 draft heads on hyper-connection models** (upstream PR: *dflash: fix draft-feature capture on hyper-connection models*).

Speculative decoding runs a small "drafter" model a few tokens ahead and lets the big model check all of them in one pass; it only helps if the drafter's guesses match what the big model would have said (the "acceptance rate"). DFlash2 is a family of small drafter models published per target model. On models from the qwen4exp family (such as Qwen3.8-Flash-Next) — which carry their hidden state as several parallel "hyper-connection" streams per position — RustingStrata v1.3.1 fed the drafter the wrong data: the feature capture read one stream instead of the streams' average, so a well-trained head had only 44% of its drafts accepted on an 8k-context bench run (with the patch: 100%). The fix averages the streams before the capture, in both the decode-step and prefill paths. Two files, +18/−1 and +22/−2.

After the fix: 100% acceptance (480/480 at 131k), and on a same-session interleaved A/B, decode went from 30.8 to 47.7 tok/s at 131k context (+55%), at a 24% prefill cost. Details, caveats and the full evidence chain in [docs/findings.md](docs/findings.md).

**2. Windows build notes.** The shipped release binary is MSVC and fine. A local build made before installing Visual Studio can end up on the GNU Rust toolchain, whose worker path was markedly slower at long context in our measurements (see findings, section 1) — upstream's README already documents installing the VS C++ Build Tools first, and the fix is to follow that.

## Build (Windows, MSVC)

```
rustup target add x86_64-pc-windows-msvc
# VS 2022 Build Tools with the "Desktop development with C++" workload (upstream README requirement)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-pc-windows-msvc
# --target is only needed while the default toolchain is GNU; after `rustup default stable-msvc`, plain `cargo build --release` suffices.
# binary: target/x86_64-pc-windows-msvc/release/rustingstrata.exe
```

CUDA needs no build step: kernels compile at run time via NVRTC. The NVRTC DLLs (`nvrtc64_130_0.dll`, `nvrtc-builtins64_130.dll`) come from the CUDA toolkit; place them next to the exe or on `PATH` (see the upstream README's runtime notes).

## Measured results

Everything below: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (66.4 GB, MoE — 48 layers, 512 experts, 10 active per token), RustingStrata v1.3.1, greedy decoding, 131,072-token context with a 98,304-token prefill prompt, CUDA backend. Hardware/OS: Windows 11 Pro 26H2, RTX 4070 SUPER 12 GB (driver 617.14), Ryzen 9 7900X3D, 32 GB RAM, model on a local fixed drive. The model is larger than system RAM, so non-resident experts stream from RAM/disk — which is why the numbers are memory-behaviour-sensitive and why the headline A/B is interleaved in one session.

| arm | decode (tok/s) | notes |
|---|--:|---|
| upstream v1.3.1 release binary, no draft | 34.4 | reference |
| this fork, no draft | 30.8 (30.1–31.7, 5 reps, interleaved) | within noise of the release binary |
| this fork + DFlash2 Q8_0 head | 47.7 (43.7–49.9, 5 reps, interleaved) | acceptance 480/480; prefill 441.6 vs 583.3 (−24%) |
| + `--kv q4` (3 reps, separate run) | 61.0 | speculative-only at this depth; see findings §2 |

**Caveats that matter (details in docs/findings.md):**

- `bench` decodes a raw greedy continuation of a repeated prefill prompt — a near-copy task that flatters speculative decoding (100% acceptance, 65% expert hit). In generate mode on one deep ~90k-token prompt, no-draft ran ~14.6 tok/s (vs 33.7 in bench mode), MTP ~18.6 (+27%), DFlash2 ~11.4 (−22%), acceptance 69.0% — while a shallow generate-mode run flipped that ranking (no-draft 17.4, DFlash2 18.7, MTP 13.0). Every arm is a single run; treat all of it as directional. At depth, prefill scales with `--chunk` (experts re-stream at chunk boundaries): on the §3.6 replay prefill, `--chunk 8192` cut the one-time cold prefill 40% vs the 4096 default (693.6 s vs 1,154–1,193 s), and 16384 exceeds the VRAM margin at 131k on 12 GB — use `--chunk 8192` for long-context serving. Two controlled comparisons on real text (findings §3.5: 21 cells at 2k–24k; §3.6: an 8-turn agent-session replay at 100k–111k context with a persistent server) found no drafter configuration that beats plain no-draft decoding on this machine — at 100k+ context MTP(1) is ~5% slower end-to-end, and the prefix cache makes long sessions practical (one ~19-minute cold prefill per ~100k-token base, then 36–79 s per turn). Measure your own workload before enabling a drafter.
- Speculative decoding here does not reproduce undrafted greedy output byte-for-byte; even no-draft output forks between CUDA and Vulkan (near-tie argmax flips). Outputs stay coherent.
- Every number is greedy decoding; sampled decoding was not measured.

## Repo layout

- `docs/findings.md` — the full investigation: approach, results, the bench-vs-generate caveat, output-identity evidence, open items, error history, reproduction
- `AGENTS.md` — conventions for AI agents working in this repo
- `bench/` — the harness scripts behind every number above, plus the evidence logs (paths redacted)
- `artifacts/` — binary hashes, compiler version, head provenance, a calibration snapshot

## Status

The fix is prepared as an upstream PR (branch `dflash-hc-mean`, single commit on upstream `main`). Numbers were produced on one machine and one model; treat single-run values as indicative until the PR's repro steps are followed.
