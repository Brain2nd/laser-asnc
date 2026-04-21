#!/bin/bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
cd /workspace/NeuronSpark-V1
MODEL=NousResearch/Llama-2-70b-hf

for i in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== attempt $i ==="
  if huggingface-cli download $MODEL --local-dir-use-symlinks False --resume 2>&1 | tail -5; then
    echo "=== download succeeded ==="
    break
  fi
  echo "=== retry in 10s ==="
  sleep 10
done

# final check
ls -la ~/.cache/huggingface/hub/models--NousResearch--Llama-2-70b-hf/snapshots/*/ 2>/dev/null | head
