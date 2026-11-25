#!/bin/bash
cd /z/ai-models/rs-v131wt
B=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
M="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
D="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
P="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/bench_suite/p1_code_rep.txt"
run(){ name=$1; shift; echo "=== $name START $(date +%H:%M:%S)"; "$@" > "$name.log" 2>&1; echo "=== $name EXIT=$? $(date +%H:%M:%S)"; }
run s7b_best       $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 3 --kv q4 --chunk 4096 --spec-draft-n-max 8 --json-out s7b_best.json
run s7b_realprompt $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 3 --kv q8 --chunk 4096 --prompt-file "$P" --json-out s7b_realprompt.json
echo ALL_S7B_DONE
