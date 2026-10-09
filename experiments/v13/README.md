# v13 학습과 128줄 입력 · drivable 실험 (2026-10-08 ~ 10-10)

12&13팀 발표(2026-10-10)에 쓴 최종 모델을 만든 코드입니다. 데이터·가중치·결과는 저장소에 올리지 않습니다(`.gitignore`).

## 결과 요약

- **U-Net v13** (`LaneUNet` base 16, 1.94M 파라미터)
  - 고정 test 487장, 행 110 이상, 255 무시, 후처리 없음
  - 전경 mIoU 93.69 (좌 93.68, 우 94.24, 횡단보도 96.58, 방지턱 90.26), 좌우 혼동 1.05%
  - 방지턱은 방지턱 라벨이 있는 180장 기준
- **PIDNet-S v13** (비교, 코드는 별도 `pinky-pidnet-segmentation`)
  - 91.98 (94.46 / 94.64 / 93.04 / 85.79), 좌우 혼동 0.06%
  - 횡단보도·방지턱과 전체 평균이 낮아 U-Net을 최종으로 골랐다.
- **128줄 입력** (112–239행만 입력, v13에서 미세조정)
  - 조향에 쓰는 144–239행 성능은 원래 모델과 0.3%p 이내
  - 112–143행은 화면 가장자리라 떨어진다.
- **drivable 머리** (`slim16`, 18,914 파라미터, 몸통 고정)
  - 사람이 검수한 1,428장(train 1,075 / val 190 / test 163)으로 학습
  - test IoU 0.989, 선 바깥 오칠 0.8% (val 0.981 / 4.2%)
  - 차선 픽셀은 몸통 출력과 같다(변경 0).
- **속도** (노트북 CPU, PyTorch 4스레드): 240줄 U-Net 45.3 ms → 128줄 + 머리 23.4 ms (약 52%). Pinky(Pi 5)에서는 아직 재지 않았다.
- **전달 모델** `v13-drivable-20261010-ede0ae96`
  - ONNX opset 17, 입력 `[1,3,128,320]`, 출력 `[1,6,128,320]`
  - 출력 채널: 배경, 왼쪽·오른쪽 차선, 횡단보도, 방지턱, drivable
  - ONNX와 PyTorch 결과 차이는 최대 3.7e-5다.

## 파일

| 경로 | 내용 |
|---|---|
| `../../configs/train_unet_v13.json` | v13 U-Net 학습 설정(그대로) |
| `../../configs/train_unet_v13_crop128.json` | 128줄 미세조정 설정(lr 3e-5, `ignore_top` 0) |
| `evaluate_one.py` | 고정 test로 U-Net/PIDNet 체크포인트 평가(PIDNet은 `pinky_pidnet` 필요) |
| `crop128/crop_train.py` | 프레임을 112–239행으로 잘라 `pinky_lane.train`을 그대로 실행 |
| `crop128/crop_eval.py`, `crop128/crop_eval_bands.py` | 원래 모델과 128줄 모델을 같은 행 구간으로 평가 |
| `crop128/export_crop.py` | 128줄 모델 TorchScript와 sidecar |
| `crop128/run_crop.sh` | 위 순서 실행 예(`PYTHON`, `CONFIG` 환경 변수로 바꿈) |
| `drivable/train_context_crop.py` | 몸통 고정 + 넓은 맥락 drivable 머리 학습(`--head slim16 --crop 112`) |
| `drivable/drivable_head.py`, `drivable/rosy_lane_model.py` | 머리·데이터셋 정의. rosy-platform(Apache-2.0) `learning/training/perception/training/`에서 가져옴(`drivable_head.py`는 일부 수정) |
| `drivable/build_package.py` | ONNX/TorchScript 내보내기, ONNX–PyTorch 일치 검사, 결과 이미지, model manifest |
| `label_rule/derive_v5.py` (+ `derive_v3.py`, `regions2.py`, `assemble5.py`) | drivable 라벨 규칙: 로봇(화면 아래 가운데)에서 흰 선을 넘지 않고 닿는 바닥 |
| `review_tool/` | 1,428장 사람 검수 웹 도구(브러시·다각형·SAM2.1 점/박스) |

## 실행 순서

저장소 루트에서 `python -m pip install -e .` 한 뒤, 데이터가 있는 작업 폴더에서 실행합니다.

```bash
# 1. v13 U-Net (data-v13: 5클래스, 255 = 무시). v12 U-Net(무시 영역 유지, sha256 95e02a1b…)에서 이어 학습
python -m pinky_lane.train --config configs/train_unet_v13.json --dataset data-v13 \
    --warm-start warm-unet.pt --output runs/unet --device cuda
python experiments/v13/evaluate_one.py --architecture unet --checkpoint runs/unet/best.pt --dataset data-v13 --output results/unet-v13.json

# 2. 128줄 입력 미세조정 → 평가 → TorchScript
python experiments/v13/crop128/crop_train.py --config configs/train_unet_v13_crop128.json \
    --dataset data-v13 --warm-start runs/unet/best.pt --output runs/unet-crop128 --device cuda
python experiments/v13/crop128/crop_eval.py
python experiments/v13/crop128/export_crop.py

# 3. drivable 머리(몸통 고정)
python experiments/v13/drivable/train_context_crop.py --code experiments/v13/drivable \
    --data <사람 검수 drivable 데이터셋> --parent runs/unet-crop128/unet_v13_crop128.torchscript.pt \
    --out runs/drivable-slim16 --head slim16 --crop 112
```

`crop_eval*.py`, `export_crop.py`는 작업 폴더의 `data-v13/`, `runs/unet/`, `runs/unet-crop128/`를 상대 경로로 읽습니다. `build_package.py`는 전달 폴더 구조(`code/training`, `parent/`, `head/`)를 인자로 받습니다.

## 한계

- 1,428장은 두 차선이 20행 이상 보이는 프레임(D-554 선택)이라 교차로·회전교차로 장면이 적습니다.
- 카메라 출처가 섞여 있습니다(8kcn / 9dfk / rosy_26 / 이전 v11 영상).
- 플랫폼 Gazebo 폐루프에서는 240줄 U-Net v13이 회전교차로 앞 코너에서 멈췄습니다. 같은 영상에서는 규칙 방식도 경계를 찾지 못해 원인을 가리는 중입니다(rosy-platform `docs/validation/lane-loop-failure-audit-2026-10-09`).
