#!/bin/bash
# Isolate each ASNC component's PPL contribution on Pythia-70m.
set -e
cd /home/dgxspark/Desktop/A2S
PY=/home/dgxspark/miniconda3/envs/AI001/bin/python

set_flags() {
  local act=$1 sm=$2 ln=$3 dcr=$4
  $PY - <<EOF
import re
f = "/home/dgxspark/Desktop/A2S/exp_full_laser_pythia.py"
s = open(f).read()
s = re.sub(r'USE_ACTIVATION_ASNC = \w+', f'USE_ACTIVATION_ASNC = {$act}', s)
s = re.sub(r'USE_SOFTMAX_ASNC = \w+', f'USE_SOFTMAX_ASNC = {$sm}', s)
s = re.sub(r'USE_LN_ASNC = \w+', f'USE_LN_ASNC = {$ln}', s)
s = re.sub(r'USE_DCR = \w+', f'USE_DCR = {$dcr}', s)
open(f, "w").write(s)
EOF
}

run_one() {
  local name=$1 act=$2 sm=$3 ln=$4 dcr=$5
  echo "=== $name (act=$act sm=$sm ln=$ln dcr=$dcr) ==="
  set_flags "$act" "$sm" "$ln" "$dcr"
  rm -f /home/dgxspark/Desktop/A2S/results_full_laser_pythia-70m.json
  $PY /home/dgxspark/Desktop/A2S/exp_full_laser_pythia.py \
    --model EleutherAI/pythia-70m \
    --result /home/dgxspark/Desktop/A2S/results_full_laser_pythia-70m.json 2>&1 | \
    grep -E "FP16 PPL|Full LASER PPL|ΔPPL"
  cp /home/dgxspark/Desktop/A2S/results_full_laser_pythia-70m.json \
     /home/dgxspark/Desktop/A2S/results_pythia70m_"$name".json
  echo
}

run_one "GeLU_only"    True  False False False
run_one "Softmax_only" False True  False False
run_one "LN_only"      False False True  False
run_one "DCR_only"     False False False True
run_one "ALL"          True  True  True  True
