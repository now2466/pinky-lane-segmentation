# 아키텍처 요약

## 목적과 범위

Pinky 주행 영상의 픽셀을 배경, 좌/우 차선, 횡단보도, 과속방지턱으로 분할합니다. 4-class 체크포인트와 본체 키 호환을 유지하면서 5-class 학습을 지원합니다. 이 모델은 segmentation 구성요소이며, 교차로에서 진행 경로를 단독으로 결정하는 경로 계획기가 아닙니다. ROS, SAM, 외부 추론 서비스는 필요하지 않습니다.

## 데이터 흐름

1. `pinky-check-dataset`이 manifest, 파일 쌍, 크기, 마스크, 해시와 split 통계를 확인합니다.
2. 데이터 로더가 320×240 이미지를 RGB `float32` `[0,1]` 텐서로 바꾸고 마스크를 `uint8` 라벨로 읽습니다.
3. LaneUNet이 4 또는 5 채널 로짓을 냅니다. 학습은 설정된 partial-label 정책을 반영한 loss를 사용합니다.
4. 평가와 추론은 클래스 마스크/지표를 만들고, 필요하면 TorchScript 형식으로 내보냅니다.

## 모듈 경계

| 모듈 | 책임 |
| --- | --- |
| `dataset.py` | manifest 기반 split 로딩, 데이터 검사, 라벨 정책 |
| `augmentations.py` | 밝기/대비 증강, 기하 변환 설정 검증 |
| `models.py` | 체크포인트 호환 LaneUNet |
| `losses.py`, `metrics.py` | partial-label 손실과 split/source별 지표 |
| `checkpoints.py` | 체크포인트 메타데이터, device, warm-start, TorchScript 로딩 |
| `train.py`, `evaluate.py` | 학습과 독립 평가 CLI |
| `infer_image.py`, `infer_video.py`, `export.py` | 이미지/영상 추론과 TorchScript 내보내기 CLI |

CLI 출력 경로는 덮어쓰지 않습니다. 데이터, 체크포인트, 실행 산출물은 공개 소스와 분리해 로컬 경로에서 관리합니다.

## 실행 환경

이 프로젝트는 Python 중심의 단일 ML 패키지이므로 Docker나 ROS 런타임 없이 프로젝트 가상환경으로 실행합니다. Python 지원 범위는 `>=3.10,<3.14`입니다. `uv.lock` 기반 설치와 프로젝트별 `venv`/pip 설치를 모두 사용할 수 있습니다. PyTorch 2.9.1, NumPy 2.2.6, `opencv-python-headless` 4.12.0.88을 사용합니다. PyTorch CPU 휠 또는 CUDA 12.8 휠을 명시적으로 선택합니다. 영상 H.264 처리에 필요한 경우에만 시스템 `ffmpeg`를 추가합니다.

## 데이터·모델·검증 경계

공개 저장소에는 사용자 주행 영상/마스크와 모델 가중치를 포함하지 않습니다. 학습된 `best.pt`와 TorchScript 파일은 사용자가 별도로 제공합니다. 역사적 validation 점수는 [모델 및 학습 문서](model_and_training.md)의 split 한계와 함께 읽어야 합니다.

pre-commit은 Gitleaks와 공개 소스 검사를 실행합니다. GitHub Actions는 CPU 테스트·정적 검사와 전체 Git 이력의 Gitleaks 검사를 실행합니다. 로컬 검증에서는 17개 테스트, 정적 검사, Gitleaks 검사를 통과했고, 기존 체크포인트 출력 호환성과 이미지·영상 CLI를 확인했습니다. 실제 주행 데이터의 새 모델 학습은 실행하지 않았습니다.
