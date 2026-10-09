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

학습 중 체크포인트 선택은 validation split으로 이뤄집니다. 과거 설정의 기본 점수(`legacy_source_macro`)는 완전 라벨 validation의 원본 그룹별 foreground mIoU를 평균합니다. 과거 실행 재현을 위해 이 기본값은 유지합니다. 개선 실험은 `configs/train_5class_stable.json`의 `fixed_foreground`를 명시적으로 사용합니다. `pinky-evaluate`는 별도로 `val` 또는 `test` split 지표를 JSON으로 기록합니다.

현재 지표 계산은 confusion-matrix의 class union이 0이면 IoU를 `null`로 기록하고 `mean_iou`, `foreground_mean_iou` 계산에서 제외합니다. 과거 점수는 absent class를 0으로 포함해 집계했습니다. 과거 결과와 비교할 때는 absent class 처리와 평균 방식을 동일하게 맞춰야 합니다.

새 `fixed_foreground` 보고서는 좌·우 차선과 횡단보도에 모든 해당 라벨 정책의
confusion을 합산하고, 방지턱에는 완전 5클래스 정책만 사용합니다. 평균에 포함할
클래스는 예측 유무가 아닌 정답 픽셀 유무로 고정합니다. 클래스별 precision,
recall, FP/FN 픽셀, 배경 대비 FP rate, 차선 좌우 혼동률을 별도로 기록합니다.
정답이 없는 클래스의 FP는 숨기지 않고 별도로 보고합니다. 이 점수는 pixel 기반
class-macro이며 과거 source-macro 점수와 직접 비교하지 않습니다. 레거시 배경은
방지턱 여부가 미확정이므로 방지턱 평가에 포함하지 않습니다.

안정화 설정은 기존 v11 warm-start, 학습률 0.00003, 최대 30에폭, plateau 3회 후
학습률 절반 감소, 채택 후보 정체 8회 후 조기 종료를 사용합니다. validation의
각 정답 클래스 IoU가 초기 warm-start보다 0.01 넘게 하락하면 채택하지 않습니다.
`best.pt`에 초기 모델(epoch 0)을 먼저 보존하므로 학습 후보가 전부 실패하면 그대로
남습니다. 이 보호 기준은 validation 기준이며 test/현장 성능을 보장하지 않습니다.
`last.pt`는 보호 기준을 통과하지 않은 후보일 수 있습니다. `training-result.json`
및 epoch 기록의 `checkpoint_accepted`를 확인하고 실제 개선 여부를 판단하세요.

확인된 배경의 CE 보정은 유지하되 `confirmed_background_dice=false`이면 방지턱
Dice 음성 픽셀에 그 배경을 추가하지 않습니다. 이는 CE-only 비교 조건이며,
기존 완전 라벨의 CE/Dice와 나머지 미확정 레거시 배경 정책은 그대로 유지합니다.

추가 후보의 `trainable_modules=["head"]`는 기존 특징 추출/복원 본체의 가중치와
BatchNorm 통계까지 동결해 출력 분류층만 조정합니다. `lane_distillation_weight`
옵션은 초기 warm-start 모델을 동결한 교사로 사용합니다. 교사가 확신도 0.75
이상으로 정답 좌/우 차선·횡단보도 라벨과 일치한 픽셀에만 temperature 2의 KL
손실을 더합니다. 교사의 오답·배경·방지턱·ignore 픽셀은 모방하지 않습니다.
이 옵션들은 실험 후보이며 개선이 입증된 설정이나 기본 배포 옵션이 아닙니다.

`small_bump_repeat_factor`는 0보다 크고 `small_bump_max_pixels`(기본512) 이하인
정답 방지턱 이미지를 추가 반복합니다. `confirmed_background_repeat_factor`는
확인된 배경 음성 사례를 반복합니다. 모든 원본 학습 이미지를 에폭마다 최소1회
포함하고 추가 반복 항목을 섞으며, validation/test 분할은 바꾸지 않습니다.
기존 weighted replacement sampling과 동시에 사용할 수 없습니다. 평가의
`by_bump_gt_size`는 완전 라벨 사례를 absent/small/medium/large로 나눠 기록하며,
전체 mIoU와 함께 작은 방지턱 누락 여부를 확인하기 위한 진단 지표입니다.

```bash
pinky-train --config configs/train_5class_stable.json --dataset data/track --output outputs/stable --warm-start weights/v11_best.pt --device cuda
```

오류 검토는 `pinky-hard-cases` 또는 `python -m pinky_lane.hard_cases`로 실행합니다.
전체 전경 오류·좌우 역할 혼동·방지턱 FP/FN·작은 FP 연결 영역마다 상위 프레임을
따로 선정하고 원본/라벨/예측 비교 이미지를 저장합니다. 라벨/분할은 바꾸지 않습니다.
미확정 레거시 배경에서 나온 방지턱 예측은 FP로 단정하지 않습니다. 검증/테스트
오류 이미지를 그대로 학습에 넣지 마세요. 이 명령은 모델 진단이며 주행 안전 검증이 아닙니다.

```bash
pinky-hard-cases --model outputs/stable/best.pt --dataset data/track --split val --device cuda --output outputs/stable-val-cases
```

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

## 선택적 공간 후처리

`pinky-infer-video --postprocess-config configs/postprocess_conservative.json`은
기본 argmax를 바꾸지 않고 선택적으로 후처리를 적용합니다. `--compare-raw`를 추가하면
왼쪽 RAW / 오른쪽 CORRECTED 비교 영상을 저장합니다. 학습·평가 지표와 TorchScript
가중치에는 이 후처리가 자동 적용되지 않습니다.

320×240 마스크 기준 과속방지턱 softmax 점수 0.75 이상, 다른 클래스와의 점수차
0.25 이상, 연결 영역 48픽셀 이상만 유지합니다. 제거 픽셀은 최선의 비방지턱 클래스로
돌립니다. softmax는 보정된 신뢰 확률이 아니며, 이 초기값은 정답 기반 최적값이 아닙니다.

차선은 연결된 좌우 통합 차선 내 32픽셀 이하 조각이 전체의 8% 이하이고, 반경 3픽셀
주변에서 반대 클래스 지지가 12픽셀 이상·90% 이상인 경우만 후보입니다. 조각의 모든
픽셀에서 좌우 점수차가 0.30 이하이고 반대 클래스 점수가 0.15 이상일 때 반대로 바꿉니다.
큰 영역·분리된 조각·강한 점수는 유지하며 화면 좌우 위치로 클래스를 정하지 않습니다.
교차로에서 잘못 합쳐진 경계는 여전히 오보정 가능성이 있어 검수가 필요합니다.
침식·팽창으로 마스크를 넓히지 않으며 시간축 평활화도 적용하지 않습니다.

프레임 JSONL에 원본/보정 클래스 픽셀 수와 보정량, 요약 JSON에 조건과 입력 해시를
기록합니다. 제거량은 정확도 개선 증거가 아닙니다. 정상 차선·실제 방지턱 누락 여부를
함께 확인한 후 배포하세요.

## Python API (기본 추론)

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

## 방지턱 회귀의 짝 비교

`python -m pinky_lane.compare_models`는 같은 split과 정답을 유지한 채 원본과 후보를
비교합니다. 방지턱 양성 프레임의 정답 총 면적을 1–48, 49–512, 513 이상 픽셀로
나누고 각 구간 TP/FP/FN과 recall을 기록합니다. 정답 면적은 객체 하나의 면적과
다를 수 있습니다. 실제 거리 대신 정답의 세로 중심 위치를 기록하며, BGR 평균은
촬영 밝기의 단순 진단값입니다. 이 값에서 인과 관계를 단정하지 않습니다.

비교는 ROI 제외 없이 모델 해상도에서 이루어집니다. negative 프레임은 제외하므로
면적별 FP를 전체 오탐률로 보고하지 않습니다. 전체 클래스 성능은 `pinky-evaluate`
결과로 별도 확인합니다. 사례를 보고 training split으로 옮기거나 정답을 임의로
수정하지 않습니다. 반복 검토한 test는 개발 벤치마크이며 새 녹화 검증을 대신하지 않습니다.

## 실제 방지턱 양성 보존과 crop 실험

추가 `bump_positive_loss`는 fully-labeled 정답 class4 픽셀만 사용합니다. 양성 픽셀의
CE를 프레임마다 평균한 후 양성 프레임 사이에서 평균하므로 큰 방지턱이 작은 방지턱을
픽셀 개수로 압도하지 않습니다. optional 작은 면적 가중치를 적용할 수 있습니다.
legacy 미라벨 background, ignore255, 확정 negative254를 양성으로 변경하지 않습니다.
기존 batch CE/Dice에 추가하는 가중치 기본값은 0입니다.

양성 crop은 모든 정답 방지턱 픽셀을 포함할 수 있는 영역에서 원점을 무작위로 고릅니다.
RGB는 linear, 정답 ID는 nearest로 원래 크기까지 복원합니다. 좌우 class ID를 바꾸거나
flip/rotation을 적용하지 않습니다. 확정 negative/ignore sentinel도 같은 기하 변환을
따릅니다. crop 비활성화 시 추가 난수를 소비하지 않으므로 기존 photometric augmentation
순서를 유지합니다. 원본 640 영상의 정보 복원과 기존 320 crop augmentation은 다릅니다.

`save_best_trained`는 class IoU guard를 통과한 실제 학습 epoch 중 validation 점수로
선정한 별도 checkpoint를 보존합니다. 기존 best가 초기 epoch0에 남더라도 학습 후보의
오류를 검토할 수 있습니다. 최종 모델 개선·오탐·작은 방지턱 보존 게이트는 이후 별도로
검증하며 test 점수로 checkpoint를 고르지 않습니다.
같은 옵션의 `best-experimental.pt`는 guard를 실패한 epoch까지 포함하여 validation 최고
점수의 실제 학습 결과를 보존합니다. checkpoint에 실패한 guard 목록을 기록하며 기존
`best.pt`나 안전 후보를 대체하지 않습니다. 모든 학습 epoch가 guard를 실패해도 초기
모델만 보고 학습 결과를 놓치지 않기 위한 진단 자료입니다.

`bump_shrink_probability`는 작은 방지턱 데이터 부족을 검증하기 위한 별도 train-only
augmentation입니다. fully-labeled 양성 프레임의 이미지와 정답을 같은 비율로 줄이고
캔버스 중앙에 배치합니다. RGB는 AREA로 줄인 뒤 reflect padding하며 padding의 정답은
255입니다. padding을 확정 배경으로 가르치지 않습니다. 정답은 nearest만 적용하며 모든
방지턱 정답이 사라지면 원본을 유지합니다. 원본 양성 면적 상한으로 과도하게 큰 객체를
제외할 수 있습니다. 이는 검수한 training 정답의 기하 변환이며 validation/test에서
새 양성 표본을 가져오거나 원본 native 640 GT를 제조하는 기능이 아닙니다.
