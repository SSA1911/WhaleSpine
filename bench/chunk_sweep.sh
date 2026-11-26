#!/usr/bin/env bash
# Chunk sweep at depth: cold prefill of the same ~100,488-token real-text prompt
# (replay turn 1) at --chunk 2048 / 8192 / 16384, fresh server per cell.
# Baseline chunk 4096 comes from replay_cells.jsonl turn1 (three independent reps:
# 1153.7 / 1192.6 s no-draft, 1206.4 / 1255.5 s mtp — MTP does not touch prefill much).
# Decision rule (b) follow-up: prefill scales with 1/chunk if experts stream once per chunk.
set -u
cd "$(dirname "$0")/.."
EXE="Z:/ai-models/rs-main/target/x86_64-pc-windows-msvc/release/rustingstrata.exe"
MODEL="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
PORT=8206
if netstat -ano 2>/dev/null | grep -q ":$PORT .*LISTENING"; then echo "PORT-BUSY $PORT"; exit 1; fi
: > bench/chunk_sweep.jsonl
BASE=$(python -c "import pathlib; t=pathlib.Path('bench/replay_base_ctx.txt').read_text(encoding='utf-8'); import json; print(json.dumps('You are assisting an engineer in the repository shown above. Reply to the LATEST user message only.\n\n' + t + '\n\n(Turn 1) List every file shown above and, in one line each, what it is responsible for.'))")

for chunk in 2048 8192 16384; do
  LOG="bench/chunk_serve_$chunk.log"; : > "$LOG"
  "$EXE" serve --gpu --backend cuda --model "$MODEL" \
    --addr 127.0.0.1:$PORT --ctx 131072 --chunk "$chunk" > "$LOG" 2>&1 &
  ok=0
  for i in $(seq 1 90); do sleep 5; grep -q "listening" "$LOG" 2>/dev/null && { ok=1; break; }; done
  if [ "$ok" != 1 ]; then echo "SERVER-FAILED chunk=$chunk"; kill_server=1; fi
  if [ "$ok" = 1 ]; then
    sleep 3
    python - "$chunk" "$BASE" <<'PY'
import sys, json, time, urllib.request
chunk, msg = sys.argv[1], json.loads(sys.argv[2])
req = urllib.request.Request("http://127.0.0.1:8206/v1/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": msg}], "max_tokens": 128}).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time(); row = {"chunk": chunk}
try:
    r = json.loads(urllib.request.urlopen(req, timeout=2400).read())
    u = r["usage"]
    row.update({"elapsed_s": round(time.time() - t0, 1),
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "finish": r["choices"][0].get("finish_reason")})
except Exception as e:
    row.update({"error": str(e)[:200], "elapsed_s": round(time.time() - t0, 1)})
open('bench/chunk_sweep.jsonl', 'a').write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)
PY
  fi
  powershell -NoProfile -Command "Get-Process rustingstrata -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  sleep 5
done
echo CHUNK-SWEEP-DONE
