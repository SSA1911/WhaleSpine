#!/bin/bash
cd /z/ai-models/rs-v131wt
B=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
M="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
D="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
echo "=== bind START $(date +%H:%M:%S) exe=$(sha256sum $B | cut -c1-16)"
$B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 5 --kv q8 --chunk 4096 --json-out s8_bind_dflash_131k.json > s8_bind.log 2>&1
echo "=== bind EXIT=$? $(date +%H:%M:%S)"
python -c "import json; d=json.load(open('s8_bind_dflash_131k.json')); r=d['results'][0]; print('BIND:', r['decode_tok_s']['mean'], f\"({r['decode_tok_s']['min']}-{r['decode_tok_s']['max']})\", '| drafts', d.get('drafts'))"
echo BIND_DONE
