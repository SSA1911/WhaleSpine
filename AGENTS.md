# AGENTS.md

Guidance for coding agents working in this repository.

## What this fork changes vs upstream

One patch on top of upstream v1.3.1 (+2 serve commits): the hc-gather path in `src/gpu_model.rs` and `src/gpu_model/prefill.rs`, fixing draft acceptance for standalone DFlash2 heads. Nothing else. When in doubt, diff against upstream `main`.

## Rules

- Do not reformat or refactor upstream code. Diffs must contain only the change being made.
- Every performance claim needs a matched pair: same build, same flags, same prompt, differing only in the variable under test. Never compare numbers across toolchains.
- Record provenance for any binary used as evidence: exe SHA-256, `rustc -vV`, git commit. No number without provenance is publishable.
- Benchmarks: `--reps 3` minimum, report min/max, pin `--backend cuda` (silent Vulkan fallback changes results), verify no other GPU process is running before starting.
- On Windows, build with `--target x86_64-pc-windows-msvc`. GNU-target builds are ~30% slower at depth and will contaminate any comparison.
- `--spec-draft-n-max` does not affect the DFlash2 path (it only populates the ngram config). Do not use it as a DFlash2 tuning lever.
- Long prompts are ~3 bytes/token for dense code: check `prompt N tok` in the log before interpreting generation results. `generate --prompt -` reads stdin; a 330 KB prompt is ~117k tokens.
- Generate can stop early (immediate EOS) and still exit 0; check the output file is non-empty before trusting a run.

## Verification workflow

1. Reproduce the claim from its recorded command line.
2. Compare against the matched baseline, not against a different build.
3. If a finding contradicts an earlier one, correct the earlier artifact in place and note the correction. Do not silently drop it.

## Commit style

Conventional Commits (`fix:`, `bench:`, `docs:`, `serve:` - the last matches upstream's convention). Keep commits single-purpose.
