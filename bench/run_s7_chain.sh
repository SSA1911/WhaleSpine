#!/bin/bash
cd /z/ai-models/rs-v131wt
B=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
M="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
D="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
P="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/bench_suite/prompts/p1.txt"
run(){ name=$1; shift; echo "=== $name START $(date +%H:%M:%S)"; "$@" > "$name.log" 2>&1; echo "=== $name EXIT=$? $(date +%H:%M:%S)"; }
run s7_nmax8      $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 3 --kv q8 --chunk 4096 --spec-draft-n-max 8 --json-out s7_nmax8.json
run s7_kvq4       $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 3 --kv q4 --chunk 4096 --json-out s7_kvq4.json
run s7_32k        $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 32768 --pp 24576 --tg 128 --reps 3 --kv q8 --chunk 4096 --json-out s7_32k.json
run s7_reps5      $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 5 --kv q8 --chunk 4096 --json-out s7_reps5.json
run s7_realprompt $B bench --gpu --backend cuda --model "$M" --draft-model "$D" --ctx 131072 --pp 98304 --tg 128 --reps 3 --kv q8 --chunk 4096 --prompt-file "$P" --json-out s7_realprompt.json
echo ALL_S7_DONE
