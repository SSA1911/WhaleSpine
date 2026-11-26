#!/usr/bin/env bash
# Agent-session replay: persistent server per phase (prefix cache ACTIVE), ~110k-token
# real project context, 8 scripted turns (tool-result/web chunks), max_tokens 512.
# Phases interleaved: nd1 -> mtp1 -> mtp2 -> nd2 (fresh server each phase).
# Engine: stock upstream v1.3.4 build (rs-main @ a02a259) — no fork patch.
# Pre-registered decision rules (Opus consult 2026-11-26):
#  (a) cache misses dominate turns 2+  (b) prefill >= 50% of turn time
#  (c) decode-bound + disk >300MB/s    (d) MTP >= +8% both warm reps -> enable MTP
set -u
cd "$(dirname "$0")/.."
EXE="Z:/ai-models/rs-main/target/x86_64-pc-windows-msvc/release/rustingstrata.exe"
MODEL="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
MTP="Z:/ai-models/mtp-heads/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
PORT=8205
# guard: never start if the port is already bound (leftover server or a second run)
if netstat -ano 2>/dev/null | grep -q ":$PORT .*LISTENING"; then echo "PORT-BUSY $PORT"; exit 1; fi
: > bench/replay_cells.jsonl

phase_server() { # $1=phase $2=armargs
  local LOG="bench/replay_serve_$1.log"
  : > "$LOG"
  "$EXE" serve --gpu --backend cuda --model "$MODEL" $2 \
    --addr 127.0.0.1:$PORT --ctx 131072 --chunk 4096 > "$LOG" 2>&1 &
  for i in $(seq 1 90); do
    sleep 5; grep -q "listening" "$LOG" 2>/dev/null && return 0
  done
  echo "SERVER-FAILED $1"; return 1
}
kill_server() {
  powershell -NoProfile -Command "Get-Process rustingstrata -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  sleep 5
}

for phase in nd1 mtp1 mtp2 nd2; do
  case $phase in
    nd*) ARMARGS="" ;;
    mtp*) ARMARGS="--mtp-model $MTP --mtp 1" ;;
  esac
  echo "== phase $phase server start $(date +%T)"
  if ! phase_server "$phase" "$ARMARGS"; then kill_server; continue; fi
  sleep 3
  python - "$phase" <<'PY'
import sys, os, json, time, pathlib, urllib.request
phase = sys.argv[1]
arm = 'nodraft' if phase.startswith('nd') else 'mtp1'
BASE = pathlib.Path('bench/replay_base_ctx.txt').read_text(encoding='utf-8')
INSTR = "You are assisting an engineer in the repository shown above. Reply to the LATEST user message only."
turns = [("turn1", BASE + "\n\n(Turn 1) List every file shown above and, in one line each, what it is responsible for.")]
for i in range(2, 9):
    turns.append((f"turn{i}", pathlib.Path(f'bench/replay_turn_{i}.txt').read_text(encoding='utf-8')))
hist = []
FIXED = "Acknowledged. (fixed reply for prompt-identity across arms)"
for tname, ttext in turns:
    if tname == 'turn1':
        hist.append({"role": "user", "content": INSTR + "\n\n" + ttext})
    else:
        hist.append({"role": "user", "content": ttext})
    logf = f"bench/replay_serve_{phase}.log"
    off = os.path.getsize(logf)
    req = urllib.request.Request("http://127.0.0.1:8205/v1/chat/completions",
        data=json.dumps({"messages": hist, "max_tokens": 512}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); row = {"phase": phase, "arm": arm, "turn": tname}
    try:
        # cold prefill of ~100k real-text tokens runs ~93 tok/s (E2 24k cells) -> ~1100 s; allow 2400
        r = json.loads(urllib.request.urlopen(req, timeout=2400).read())
        ch = r["choices"][0]; u = r["usage"]
        row.update({"elapsed_s": round(time.time() - t0, 1),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "finish": ch.get("finish_reason"),
                    "content_chars": len(ch["message"].get("content") or ""),
                    "reasoning_chars": len(ch["message"].get("reasoning_content") or "")})
    except Exception as e:
        row.update({"error": str(e)[:200], "elapsed_s": round(time.time() - t0, 1)})
    hist.append({"role": "assistant", "content": FIXED})
    with open(logf, encoding='utf-8', errors='replace') as f:
        f.seek(off)
        timing = [l.strip()[:180] for l in f.read().splitlines()
                  if ('prefill' in l.lower() or 'decode' in l.lower() or 'cache' in l.lower())]
    row["log_lines"] = timing[:10]
    open('bench/replay_cells.jsonl', 'a').write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in row if k != 'log_lines'}), flush=True)
    if 'error' in row: break
PY
  kill_server
done
echo REPLAY-DONE
