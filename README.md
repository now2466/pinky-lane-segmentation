# Pinky Lane Segmentation

`now2466/pinky-lane-segmentation`은 Pinky 주행 영상의 차선과 노면 표식을 분할하는 PyTorch 프로젝트입니다. 기존 4-class LaneUNet 가중치와 호환되는 4-class 모델, 과속방지턱을 추가한 5-class 모델, 데이터 검사·학습·평가·이미지/영상 추론·TorchScript 내보내기 명령을 제공합니다.

이 저장소에는 학습 데이터와 모델 가중치를 넣지 않습니다. 데이터와 가중치는 별도로 준비해 로컬 경로에 둡니다. 모델은 픽셀별 역할을 분할하며, 교차로에서 어떤 경로를 선택할지 단독으로 결정하지 않습니다.

## 가져오기

```bash
git clone https://github.com/now2466/pinky-lane-segmentation.git
cd pinky-lane-segmentation
```

## 저장소 구조

```text
pinky-lane-segmentation/
├── .github/workflows/ci.yml
├── configs/
│   ├── train_4class.json
│   ├── train_5class.json
│   └── train_5class_cpu.json
├── docs/
├── examples/dataset_manifest.json
├── scripts/check_public_source.py
├── tests/
├── src/pinky_lane/
│   ├── augmentations.py  checkpoints.py  dataset.py
│   ├── evaluate.py  export.py  inference.py
│   ├── infer_image.py  infer_video.py  losses.py
│   └── metrics.py  models.py  train.py
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

`data/`, `weights/`, `outputs/`는 사용자가 로컬에서 준비하는 경로이며 저장소에 업로드하지 않습니다.

## 클래스와 입출력

| ID | 클래스 |
| --- | --- |
| 0 | 배경 |
| 1 | 왼쪽 차선 |
| 2 | 오른쪽 차선 |
| 3 | 횡단보도 |
| 4 | 과속방지턱 (5-class 모델만 해당) |
| 255 | 무시/미라벨 영역 (모델 클래스가 아님) |

모델 입력은 RGB `float32` 텐서 `[N, 3, 240, 320]`, 값 범위 `[0, 1]`입니다. 출력은 클래스별 로짓 `[N, 4|5, 240, 320]`이며, `argmax` 결과는 `uint8` 클래스 마스크입니다. `255`는 학습 가능한 클래스가 아니라 정답에서 제외할 픽셀을 뜻하며, 추론에서는 `ignore-top`으로 제외한 행에도 사용합니다.

## 설치

Python `>=3.10,<3.14`가 필요합니다. ROS는 필요하지 않습니다. CPU 또는 CUDA가 포함된 PyTorch 2.9.1 휠 중 하나를 먼저 선택합니다. PyTorch가 안내하는 CPU 및 CUDA 12.8 인덱스는 [공식 설치 명령](https://pytorch.org/get-started/previous-versions/)에서 확인할 수 있습니다.

가상환경과 pip를 사용하는 경우:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU 전용 휠을 선택하는 경우
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 또는 CUDA 12.8 휠을 선택하는 경우 위 CPU 명령 대신 실행
# python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

저장소의 `uv.lock`을 사용하는 환경에서는 프로젝트 루트에서 잠금 파일에 고정된 구성을 설치합니다. `uv sync`의 잠금 동작은 [uv 공식 문서](https://docs.astral.sh/uv/concepts/projects/sync/)를 참고하세요.

```bash
uv sync --locked --extra dev
```

`uv.lock`은 프로젝트의 PyTorch 배포본을 고정합니다. Linux x86_64 잠금 항목은 CUDA 런타임 의존성을 포함한 PyPI 배포본입니다. CPU 전용 휠 또는 공식 CUDA 12.8 휠을 명시적으로 선택하려면 위 pip 설치에서 해당 인덱스를 고르세요. `--device auto`는 설치된 휠을 고르는 옵션이 아니라 실행 시 CUDA 사용 가능 여부를 선택합니다. 영상 H.264 디코딩/인코딩에는 시스템 `ffmpeg`가 선택적으로 필요합니다.

## 데이터 준비

데이터셋은 `manifest.json`과 `train`, `val`, `test` 분할로 구성합니다.

```text
data/track/
├── manifest.json
├── train/{images/*.jpg,masks/*.png}
├── val/{images/*.jpg,masks/*.png}
└── test/{images/*.jpg,masks/*.png}
```

이미지와 마스크는 각각 `320×240`이어야 하며, 파일명 stem이 서로 같아야 합니다. 5-class 학습에서는 manifest 각 항목에 명시적인 `supervision_policy`가 필요합니다. v11 비례 분할 manifest는 완전 5-class 라벨과 기존 부분 라벨을 함께 포함하므로 5-class 설정으로 사용합니다. 4-class 설정에는 별도의 4-class 데이터셋을 지정하세요. 데이터 스키마와 마스크 작성 규칙은 [데이터 형식 문서](docs/dataset_format.md)를 참고하세요.

```bash
pinky-check-dataset --dataset data/track --num-classes 5
```

검사는 세 split의 파일·마스크·manifest 일치 여부, 선택적 SHA-256, split 사이의 완전 동일 이미지 중복을 확인하고 개수를 보고합니다. 원본 동영상이 여러 split에 걸쳐 있는 경우도 별도 개수로 보고하므로, 프레임 단위 split의 결과를 독립 녹화 검증으로 해석하지 마세요.

## 학습 및 dry-run

먼저 dry-run으로 데이터 검사와 모델 순전파를 확인할 수 있습니다. 이 명령은 optimizer 학습을 시작하지 않습니다.

```bash
pinky-train --config configs/train_5class.json --dataset data/track --output outputs/check-5class --device cpu --dry-run
```

일반 학습은 설정 파일을 선택합니다.

```bash
pinky-train --config configs/train_4class.json --dataset data/track4 --output outputs/run-4class
pinky-train --config configs/train_5class.json --dataset data/track --output outputs/run-5class
pinky-train --config configs/train_5class_cpu.json --dataset data/track --output outputs/run-5class-cpu
```

첫 번째 명령은 4-class manifest에만 사용하고, 다음 두 명령은 5-class manifest에 사용합니다. CPU 설정은 batch size 2, `workers=0`, `cpu_threads=4`, CPU device를 사용하고 AMP를 끕니다. `--warm-start weights/best_checkpoint.pt`, `--device cpu|cuda|auto`, `--epochs N`을 추가로 지정할 수 있습니다. 학습 결과는 새 출력 폴더에 기록되며, 이미 존재하는 출력 경로는 덮어쓰지 않습니다. 학습 절차와 partial-label 정책은 [모델 및 학습 문서](docs/model_and_training.md)에 정리되어 있습니다.

`--warm-start`는 모델 가중치를 초기값으로 읽으며 optimizer와 epoch는 새로 시작합니다. 중단된 optimizer 상태를 복구하는 resume 기능은 제공하지 않습니다.

## 평가, 추론, 내보내기

학습 폴더의 `best.pt` 또는 별도로 준비한 가중치를 지정합니다. 아래 출력 경로는 실행 전 존재하지 않아야 합니다.

```bash
pinky-evaluate --model outputs/run-5class/best.pt --dataset data/track --split test --device cpu --output outputs/test.json

pinky-infer-image --model weights/best_checkpoint.pt --image frame.jpg --output outputs/image --format checkpoint --ignore-top 110 --device cpu

pinky-infer-video --model weights/best_checkpoint.pt --video clip.mp4 --output outputs/video --format checkpoint --ignore-top 110 --device cpu --batch-size 8 --max-frames 100

pinky-export --model weights/best_checkpoint.pt --output weights/model.torchscript.pt --ignore-top 110
```

4-class 레거시 가중치는 `ignore-top` 기본값이 110입니다. 5-class 가중치는 체크포인트 메타데이터에 값이 있거나 `--ignore-top N`을 직접 지정해야 합니다. 저장소는 가중치를 기본 제공하지 않습니다. 기존 로컬 `0930_best_model.torchscript.pt` 파일을 보유한 경우 추론에 `--format torchscript`를 지정할 수 있습니다. 추론 명령의 대상 경로도 기존 파일/폴더를 덮어쓰지 않습니다.

## 개발 및 공개 전 검사

```bash
python -m pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
python -m pytest -q
python -m ruff check src tests scripts
python scripts/check_public_source.py
```

pre-commit은 Gitleaks `v8.30.1`과 `scripts/check_public_source.py`를 실행합니다. GitHub Actions에는 `tests`와 `secrets` 두 job이 있습니다. `tests`는 pytest/ruff와 공개 소스 검사를 실행하고, `secrets`는 checksum을 확인한 Gitleaks `v8.30.1`로 전체 Git 이력을 검사합니다. 공개 소스 검사는 Git index 내용을 대상으로 개인 홈 절대 경로, 사설 IPv4 주소, 비공개 산출물과 자격 증명 파일을 차단합니다.

## 문서

- [데이터셋 형식](docs/dataset_format.md)
- [모델 및 학습](docs/model_and_training.md)
- [아키텍처 요약](docs/architecture-brief.md)
