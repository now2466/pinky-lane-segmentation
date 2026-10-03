# 모델 및 학습

## LaneUNet 구조

4-class와 5-class 모델은 같은 LaneUNet 본체를 사용합니다. 인코더 채널은 `16 → 32 → 64 → 128`, 가운데 블록은 `256`입니다. 각 블록은 `3×3 Conv → BatchNorm → ReLU` 두 층으로 구성됩니다. 2배 최대 풀링으로 해상도를 줄이고, 디코더의 전치 합성곱과 skip connection으로 복원한 뒤 `1×1` head가 클래스별 로짓을 출력합니다. 공유하는 기존 체크포인트의 아키텍처는 4-class `LaneUNet-v1`이며, 5-class 모델은 `LaneUNet-v1-5class`로 구분합니다.

기존 체크포인트의 모듈 이름과 본체 파라미터 키를 유지합니다. 4-class 체크포인트는 4-class 모델에 그대로 불러올 수 있습니다. 4-class에서 5-class로 warm-start하면 본체와 기존 4개 출력 채널을 복사하고, 새 과속방지턱 출력 채널은 새로 초기화합니다.

입력 계약은 RGB `float32` `[N,3,240,320]`, `[0,1]` 범위입니다. 출력은 확률이 아닌 로짓 `[N,4|5,240,320]`; 추론 마스크는 클래스 축의 `argmax`를 취한 `uint8` 결과입니다. 전처리는 BGR로 읽은 이미지를 RGB로 바꾸고 255로 나눕니다.

## 손실과 라벨 정책

4-class 학습은 픽셀 빈도에 따른 가중치를 사용합니다.

```text
weight[c] = clip(sqrt(max_count / count[c]), 1, 6)
loss = weighted_cross_entropy(ignore_index=255) + 0.3 × foreground_Dice_loss
```

각 클래스 픽셀 수는 학습 split에서 계산합니다. 네 클래스 모두 학습 데이터에 나타나야 합니다.

5-class 학습은 `fully_labeled_5class`와 `legacy_partial_class4`를 구분합니다. 완전 라벨 항목은 일반적인 클래스별 cross-entropy를 사용합니다. 레거시 항목의 라벨 0은 배경과 과속방지턱의 합산 확률에 대한 marginal cross-entropy로 계산합니다. 레거시 항목에서는 과속방지턱 Dice를 계산하지 않고, 완전 라벨 항목에서만 클래스 4 Dice를 계산합니다. 전체 손실은 `partial_label_cross_entropy + 0.3 × masked_foreground_Dice_loss`입니다. 모든 정책에서 `255`는 손실과 지표 계산에서 제외됩니다.

## 데이터 증강과 평가

현재 설정은 밝기/대비 변화만 적용합니다. 좌우 반전과 회전은 비활성화되어 있습니다. 좌우 반전은 왼쪽/오른쪽 차선 역할을 바꾸며, 회전은 이 버전의 지원 범위가 아닙니다.

학습 중 체크포인트 선택은 validation split으로 이뤄집니다. 5-class 점수는 완전 라벨 validation의 원본 그룹별 foreground mIoU를 먼저 계산한 뒤 그룹 평균을 냅니다. 레거시 partial-label validation은 과속방지턱 성능 점수에 섞지 않습니다. `pinky-evaluate`는 별도로 `val` 또는 `test` split 지표를 JSON으로 기록합니다.

현재 지표 계산은 confusion-matrix의 class union이 0이면 IoU를 `null`로 기록하고 `mean_iou`, `foreground_mean_iou` 계산에서 제외합니다. 과거 점수는 absent class를 0으로 포함해 집계했습니다. 과거 결과와 비교할 때는 absent class 처리와 평균 방식을 동일하게 맞춰야 합니다.

기본 학습 설정은 `inference.ignore_top=110`을 둡니다. 이는 해당 설정이 사용하는 추론 ROI 기준이며, 전체 이미지의 모든 행에서 성능을 보장한다는 뜻은 아닙니다. 데이터의 `255` 마스크와 평가 지표가 유효 픽셀을 결정합니다.

## 실행

Dry-run은 데이터 사전 검사, 한 샘플 순전파, 유한한 loss인지 확인하고 학습을 시작하지 않습니다.

```bash
pinky-train --config configs/train_4class.json --dataset data/track4 --output outputs/check-4class --device cpu --dry-run
pinky-train --config configs/train_5class.json --dataset data/track --output outputs/check-5class --device cpu --dry-run
```

실제 학습은 새 출력 경로를 사용합니다.

```bash
pinky-train --config configs/train_4class.json --dataset data/track4 --output outputs/run-4class
pinky-train --config configs/train_5class.json --dataset data/track --output outputs/run-5class --warm-start weights/best_checkpoint.pt --device auto --epochs 20
pinky-train --config configs/train_5class_cpu.json --dataset data/track --output outputs/run-5class-cpu
```

각 명령은 설정에 맞는 데이터셋 manifest에 사용합니다. CPU 설정은 batch size 2, worker 0, thread 4로 CPU 사용량과 병렬 작업을 제한합니다. `--device`는 `cpu`, `cuda`, `auto`를 받습니다. `--epochs N`은 설정의 epoch 수를 덮어씁니다. 실행 폴더에는 사용 설정, 데이터/체크포인트 provenance, epoch 지표와 `best.pt`, `last.pt`가 저장됩니다. 이미 존재하는 출력 경로는 덮어쓰지 않습니다.

```bash
pinky-evaluate --model outputs/run-5class/best.pt --dataset data/track --split test --device cpu --output outputs/test.json
pinky-evaluate --model outputs/run-5class/best.pt --dataset data/track --split val --device cpu --output outputs/val.json
```

## 추론, ROI, 체크포인트

학습 체크포인트는 클래스 순서와 아키텍처 메타데이터를 확인합니다. TorchScript 모델은 `--format torchscript`로 불러옵니다. TorchScript 옆에 JSON 메타데이터 파일이 있으면 클래스 순서와 추론 설정을 읽습니다.

4-class 레거시 체크포인트는 `ignore_top` 기본값이 `110`입니다. 5-class 모델은 체크포인트 메타데이터에 `inference.ignore_top`이 있어야 하며, 없으면 추론 명령에 `--ignore-top N`을 지정해야 합니다. 내보내기에도 같은 옵션을 줄 수 있습니다.

```bash
pinky-infer-image --model weights/best_checkpoint.pt --image frame.jpg --output outputs/image --format checkpoint --ignore-top 110 --device cpu
pinky-infer-video --model weights/best_checkpoint.pt --video clip.mp4 --output outputs/video --format checkpoint --ignore-top 110 --device cpu --batch-size 8 --max-frames 100
pinky-export --model weights/best_checkpoint.pt --output weights/model.torchscript.pt --ignore-top 110
```

이미지 CLI는 `mask.png`, `overlay.png`, `prediction.json`을 출력합니다. 영상 CLI는 오버레이 영상과 프레임별 JSONL, 요약 JSON을 출력하며 오디오는 포함하지 않습니다. `ffmpeg`가 있으면 H.264 MP4를 만들고, 없으면 OpenCV MJPG AVI를 사용합니다. 추론은 입력 영상을 320×240으로 바꾸고 `argmax` 후 맨 위 `ignore_top`개 행을 255로 표시합니다. 모든 출력 명령은 이미 존재하는 파일이나 디렉터리를 덮어쓰지 않습니다.

저장소에는 학습 가중치가 기본 포함되지 않습니다. 기존 로컬 TorchScript 파일은 이름과 무관하게 `--format torchscript`로 불러올 수 있습니다. 팀에서 별도로 공유하는 `best_checkpoint.pt`도 별도 경로에서 지정합니다. 추론에는 SAM 의존성이 없습니다.

## Python API

체크포인트 API는 모델, 실행 장치, 클래스 수, ROI 상단 제외 행 수를 반환합니다. `image`는 이미 RGB `float32` `[1,3,240,320]` 및 `[0,1]` 범위로 전처리된 텐서입니다.

```python
import torch
from pinky_lane.checkpoints import load_model

model, device, num_classes, ignore_top = load_model(
    "weights/best_checkpoint.pt",
    device="cpu",
    model_format="checkpoint",
)
with torch.inference_mode():
    logits = model(image.to(device))
mask = logits.argmax(dim=1).to(torch.uint8)
mask[:, :ignore_top, :] = 255
```

기존 로컬 4-class TorchScript 파일은 패키지 helper와 별개로 직접 읽을 수 있습니다. `torch.jit.load`는 JSON sidecar의 ROI 메타데이터를 적용하지 않으므로, argmax 다음 상단 110행을 명시적으로 제외합니다. 아래 파일명은 예시이며 가중치 파일은 저장소에 포함하지 않습니다.

```python
import torch

model = torch.jit.load("0930_best_model.torchscript.pt", map_location="cpu").eval()
with torch.inference_mode():
    logits = model(image)  # image: RGB float32 [1,3,240,320], values in [0,1]
mask = logits.argmax(dim=1).to(torch.uint8)
mask[:, :110, :] = 255
```

기존 모델 manifest에는 `epoch=7`의 `val_foreground_miou=0.83385`가 기록되어 있습니다. 해당 validation 프레임은 train과 같은 녹화에서 시간상 나뉜 샘플이므로 독립 녹화 holdout 성능이나 새로운 학습 결과로 해석하지 마세요.

별도 호환성 확인에서는 기존 checkpoint/state/TorchScript 경로의 forward 최대 절대 오차가 0이었고, 원본 5-class CE/Dice 손실과도 정확히 일치했습니다. 이는 구현 호환성 확인이며 새 학습이나 일반화 성능을 뜻하지 않습니다.
