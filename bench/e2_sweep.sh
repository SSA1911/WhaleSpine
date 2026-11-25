#!/usr/bin/env bash
# E2: chat-mode draft-length sweep, fresh serve per cell (no prefix-cache reuse).
# arms: nodraft | mtp1 | mtp2 | mtp3 | df1 | df2 | df3  x prompts 2k/8k/24k
set -u
cd "$(dirname "$0")/.."
EXE=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
MODEL="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
MTP="Z:/ai-models/mtp-heads/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
DF="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
PORT=8203
OUT=bench/e2_cells.jsonl
: > "$OUT"
for arm in nodraft mtp1 mtp2 mtp3 df1 df2 df3; do
  case $arm in
    nodraft) ARMARGS="" ;;
    mtp*) ARMARGS="--mtp-model $MTP --mtp ${arm#mtp}" ;;
    df*)  ARMARGS="--draft-model $DF --mtp ${arm#df}" ;;
  esac
  for p in 2k 8k 24k; do
    echo "== cell $arm $p $(date +%T)"
    LOG="bench/e2_serve_${arm}_${p}.log"
    "$EXE" serve --gpu --backend cuda --model "$MODEL" $ARMARGS \
      --addr 127.0.0.1:$PORT --ctx 65536 --chunk 4096 > "$LOG" 2>&1 &
    SPID=$!
    for i in $(seq 1 60); do
      sleep 5; grep -q "listening" "$LOG" 2>/dev/null && break
    done
    sleep 3
    python - "$arm" "$p" <<'PY'
import json, sys, pathlib, urllib.request, time
arm, p = sys.argv[1], sys.argv[2]
content = pathlib.Path(f"bench/e2_prompt_{p}.txt").read_text(encoding="utf-8")
body = {"messages":[{"role":"user","content":content}], "max_tokens":512}
req = urllib.request.Request("http://127.0.0.1:8203/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0=time.time()
try:
    r = json.loads(urllib.request.urlopen(req, timeout=560).read())
    ch=r["choices"][0]; u=r["usage"]
    row={"arm":arm,"prompt":p,"elapsed_s":round(time.time()-t0,1),
         "prompt_tokens":u.get("prompt_tokens"),"completion_tokens":u.get("completion_tokens"),
         "finish":ch.get("finish_reason"),
         "content_chars":len(ch["message"]["content"] or ""),
         "reasoning_chars":len(ch["message"]["reasoning_content"] or "")}
except Exception as e:
    row={"arm":arm,"prompt":p,"error":str(e)[:200],"elapsed_s":round(time.time()-t0,1)}
open("bench/e2_cells.jsonl","a").write(json.dumps(row)+"\n")
print(row)
PY
    kill $SPID 2>/dev/null
    sleep 5
    powershell -NoProfile -Command "Get-Process rustingstrata -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
    sleep 5
  done
done
echo E2-DONE
