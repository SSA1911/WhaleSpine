#!/usr/bin/env bash
# Q7 experiment (Astra consult 2026-11-26): does corrected DFlash2 shorten warm turn
# time on real 100k+ agent traffic, scoring ANSWER-CARRYING turns (max_tokens 2048,
# not the replay's 512 which truncated reasoning)? Same binary as E2 (patched 32bde88),
# same base context and turn construction as bench/replay_run.sh.
# Arms interleaved nd1 -> df1 -> df2 -> nd2, persistent server per phase (prefix cache
# part of the measurement), chunk 4096 = replay baseline for comparability.
# Per phase: turn1 (max_tokens 1, unscored, establishes ~100.5k-token prefix) ->
# turn2 (scored) -> turn6 (scored), fixed assistant replies between, byte-identical
# prompts across arms. Kill threshold (pre-registered): DFlash2 must beat no-draft by
# >=10% on BOTH turns in BOTH passes, with content_chars > 0, else the fork line closes.
set -u
cd "$(dirname "$0")/.."
EXE=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
MODEL="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
DF="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
PORT=8207
if netstat -ano 2>/dev/null | grep -q ":$PORT .*LISTENING"; then echo "PORT-BUSY $PORT"; exit 1; fi
: > bench/dflash_warm_cells.jsonl

kill_server() {
  powershell -NoProfile -Command "Get-Process rustingstrata -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
  sleep 5
}

for phase in nd1 df1 df2 nd2; do
  case $phase in
    nd*) ARMARGS="" ;;
    df*) ARMARGS="--draft-model $DF --mtp 1" ;;
  esac
  LOG="bench/dflash_warm_serve_$phase.log"; : > "$LOG"
  echo "== phase $phase server start $(date +%T)"
  "$EXE" serve --gpu --backend cuda --model "$MODEL" $ARMARGS \
    --addr 127.0.0.1:$PORT --ctx 131072 --chunk 4096 --kv q8 > "$LOG" 2>&1 &
  # v3: --kv q8 added on ALL phases (v1/v2 burned on the same f16-KV plan — the DFlash2
  # drafter's full-context KV + head + scratch add ~4.2 GB and left 0 MB free at f16 KV).
  ok=0
  for i in $(seq 1 90); do sleep 5; grep -q "listening" "$LOG" 2>/dev/null && { ok=1; break; }; done
  if [ "$ok" != 1 ]; then echo "SERVER-FAILED $phase"; kill_server; continue; fi
  sleep 3
  python - "$phase" <<'PY'
import sys, os, json, time, pathlib, urllib.request
phase = sys.argv[1]
arm = 'nodraft' if phase.startswith('nd') else 'dflash2'
BASE = pathlib.Path('bench/replay_base_ctx.txt').read_text(encoding='utf-8')
INSTR = "You are assisting an engineer in the repository shown above. Reply to the LATEST user message only."
turns = [("turn1", BASE + "\n\n(Turn 1) List every file shown above and, in one line each, what it is responsible for.")]
for i in (2, 6):
    turns.append((f"turn{i}", pathlib.Path(f'bench/replay_turn_{i}.txt').read_text(encoding='utf-8')))
hist = []
FIXED = "Acknowledged. (fixed reply for prompt-identity across arms)"
for tname, ttext in turns:
    if tname == 'turn1':
        hist.append({"role": "user", "content": INSTR + "\n\n" + ttext})
    else:
        hist.append({"role": "user", "content": ttext})
    mt = 1 if tname == 'turn1' else 2048
    logf = f"bench/dflash_warm_serve_{phase}.log"
    off = os.path.getsize(logf)
    req = urllib.request.Request("http://127.0.0.1:8207/v1/chat/completions",
        data=json.dumps({"messages": hist, "max_tokens": mt}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); row = {"phase": phase, "arm": arm, "turn": tname}
    try:
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
                  if ('prefill' in l.lower() or 'decode' in l.lower() or 'cache' in l.lower()
                      or 'accept' in l.lower() or 'draft' in l.lower())]
    row["log_lines"] = timing[:10]
    open('bench/dflash_warm_cells.jsonl', 'a').write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in row if k != 'log_lines'}), flush=True)
    if 'error' in row: break
PY
  kill_server
done
echo Q7-DONE
