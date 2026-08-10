#!/bin/zsh
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data
./.venv/bin/python gen_local.py >> gen.log 2>&1 || { echo "GEN FAILED" >> pipeline.log; exit 1; }
echo "GEN COMPLETE, starting training" >> pipeline.log
./.venv/bin/python train_local.py > train.log 2>&1 || { echo "TRAIN FAILED" >> pipeline.log; exit 1; }
echo "PIPELINE DONE" >> pipeline.log
