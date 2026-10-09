#!/bin/bash
# 128줄 입력 미세조정 → 평가. 작업 폴더에 data-v13/ 와 runs/unet/best.pt(v13 U-Net)가 있어야 한다.
set -e
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=${CONFIG:-$HERE/../../../configs/train_unet_v13_crop128.json}
$PY "$HERE/crop_train.py" --config "$CONFIG" --dataset data-v13 --warm-start runs/unet/best.pt --output runs/unet-crop128 --device cuda > unet-crop128-train.log 2>&1
$PY "$HERE/crop_eval.py" > unet-crop128-eval.log 2>&1
