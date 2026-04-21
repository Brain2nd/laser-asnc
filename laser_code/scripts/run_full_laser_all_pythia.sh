#!/bin/bash
# Full-LASER PPL on all 8 Pythia sizes (K_sm=64, per-channel codecs).
set -e
cd /home/dgxspark/Desktop/A2S
PY=/home/dgxspark/miniconda3/envs/AI001/bin/python

# Confirm flags are set for ALL (expected already set)
$PY - <<'EOF'
import re
f = "/home/dgxspark/Desktop/A2S/exp_full_laser_pythia.py"
s = open(f).read()
s = re.sub(r'USE_ACTIVATION_ASNC = \w+', 'USE_ACTIVATION_ASNC = True', s)
s = re.sub(r'USE_SOFTMAX_ASNC = \w+', 'USE_SOFTMAX_ASNC = True', s)
s = re.sub(r'USE_LN_ASNC = \w+', 'USE_LN_ASNC = True', s)
s = re.sub(r'USE_DCR = \w+', 'USE_DCR = True', s)
s = re.sub(r'K_SOFTMAX = \d+', 'K_SOFTMAX = 64', s)
open(f, "w").write(s)
EOF

for MODEL_ID in 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b; do
  RESULT=/home/dgxspark/Desktop/A2S/results_full_laser_pythia-${MODEL_ID}.json
  if [ -f "$RESULT" ] && grep -q full_laser_ppl "$RESULT" 2>/dev/null; then
    echo "=== [skip] pythia-${MODEL_ID} ==="
    continue
  fi
  echo "=== pythia-${MODEL_ID} ==="
  $PY -u /home/dgxspark/Desktop/A2S/exp_full_laser_pythia.py \
    --model EleutherAI/pythia-${MODEL_ID} \
    --result "$RESULT" 2>&1 | grep -vE '^ppl:' | tail -20
done

echo
echo "=== Summary ==="
for MODEL_ID in 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b; do
  RESULT=/home/dgxspark/Desktop/A2S/results_full_laser_pythia-${MODEL_ID}.json
  if [ -f "$RESULT" ]; then
    $PY -c "import json; r=json.load(open('$RESULT')); print(f'pythia-${MODEL_ID:>5}: FP16={r[\"fp16_ppl\"]:.4f} LASER={r[\"full_laser_ppl\"]:.4f} Δ={r[\"delta_ppl\"]:+.4f}')"
  fi
done
