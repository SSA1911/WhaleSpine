#!/bin/bash
cd /z/ai-models/rs-v131wt
B=./target/x86_64-pc-windows-msvc/release/rustingstrata.exe
M="Z:/ai-models/gguf/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
D="Z:/ai-models/mtp-heads/dflash2-st/model-dflash-q8_0.gguf"
MT="Z:/ai-models/mtp-heads/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
gen(){ name=$1; shift; echo "=== $name START $(date +%H:%M:%S)"; $B generate --gpu --backend cuda --model "$M" --kv q8 --chunk 4096 --ctx 131072 -n 128 --prompt - "$@" < deep_prompt.txt > "$name.txt" 2> "$name.log"; echo "=== $name EXIT=$? $(date +%H:%M:%S)"; }
gen deep_nodraft
gen deep_dflash --draft-model "$D"
gen deep_mtp --mtp 4 --mtp-model "$MT"
echo "=== DIFFS"
diff deep_nodraft.txt deep_dflash.txt >/dev/null && echo "ND vs DFLASH: IDENTICAL" || echo "ND vs DFLASH: diverge"
diff deep_nodraft.txt deep_mtp.txt >/dev/null && echo "ND vs MTP: IDENTICAL" || echo "ND vs MTP: diverge"
diff deep_dflash.txt deep_mtp.txt >/dev/null && echo "DFLASH vs MTP: IDENTICAL" || echo "DFLASH vs MTP: diverge"
grep -h "accepted" deep_dflash.log deep_mtp.log
echo ALL_DEEP_IDENTITY_DONE
