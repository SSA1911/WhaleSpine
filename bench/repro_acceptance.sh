#!/usr/bin/env bash
# Regression repro for the hc-gather patch (docs/findings.md section 2).
# Pre-patch: standalone DFlash2 heads accept ~0% of drafts.
# Post-patch: this smoke run must exceed 90% acceptance, else exit 1.
# Usage: bash bench/repro_acceptance.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
MODEL="${MODEL:-Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}"
DRAFT="${DRAFT:-Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf}"

"$BIN" bench --gpu --backend cuda \
  --model "$MODEL" --draft-model "$DRAFT" \
  --ctx 8192 --pp 4096 --tg 128 --reps 1 \
  --kv q8 --chunk 4096 \
  --json-out bench/repro_acceptance.json 2>&1

python3 - <<'PY'
import json, sys
d = json.load(open("bench/repro_acceptance.json"))
dr = d.get("drafts") or d["results"][0].get("drafts", {})
made, acc = dr.get("made", 0), dr.get("accepted", 0)
rate = acc / made if made else 0.0
print(f"hc-gather regression check: acceptance {acc}/{made} = {rate:.1%}")
sys.exit(0 if rate > 0.9 else 1)
PY
