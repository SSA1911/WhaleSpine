#!/bin/bash
# S9: Opus xhigh audit conditions C1 (before arm), C7 (EOS escape test), C2 (interleaved same-session pair).
cd /z/ai-models/rs-v131wt
M="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
D="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
STOCK="Z:/ai-models/rs-v131/rustingstrata-v131.exe"
B=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe

echo "=== C1 before-arm START $(date +%H:%M:%S) stock=$(sha256sum $STOCK | cut -c1-8)"
"$STOCK" bench --gpu --backend cuda --model "$M" --draft-model "$D" \
  --ctx 8192 --pp 4096 --tg 128 --reps 1 --kv q8 --chunk 4096 \
  --json-out s9_before_stock_8k.json > s9_before_stock.log 2>&1
echo "C1 acceptance: $(grep 'DFlash drafts accepted' s9_before_stock.log)"
echo "=== C1 EXIT=$? $(date +%H:%M:%S)"

echo "=== C7 escape START $(date +%H:%M:%S)"
"$B" generate --gpu --backend cuda --model "$M" --kv q8 --chunk 4096 --ctx 65536 -n 128 \
  --prompt - < s9_escape_prompt.txt > s9_escape_out.txt 2> s9_escape.log
echo "C7 exit=$? bytes=$(wc -c < s9_escape_out.txt)"
grep -E "accepted|tok" s9_escape.log | head -2
echo "=== C7 EXIT=$? $(date +%H:%M:%S)"

echo "=== C2 interleaved pair START $(date +%H:%M:%S) exe=$(sha256sum $B | cut -c1-8)"
for i in 1 2 3 4 5; do
  echo "--- rep $i no-draft $(date +%H:%M:%S)"
  "$B" bench --gpu --backend cuda --model "$M" \
    --ctx 131072 --pp 98304 --tg 128 --reps 1 --kv q8 --chunk 4096 \
    --json-out "s9_nodraft_rep$i.json" > "s9_nodraft_rep$i.log" 2>&1
  echo "--- rep $i dflash $(date +%H:%M:%S)"
  "$B" bench --gpu --backend cuda --model "$M" --draft-model "$D" \
    --ctx 131072 --pp 98304 --tg 128 --reps 1 --kv q8 --chunk 4096 \
    --json-out "s9_dflash_rep$i.json" > "s9_dflash_rep$i.log" 2>&1
done
echo "=== C2 EXIT=$? $(date +%H:%M:%S)"
echo S9_ALL_DONE
