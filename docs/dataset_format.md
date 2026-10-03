# 데이터셋 형식

## 디렉터리와 파일

`--dataset`은 아래 구조의 데이터셋 루트를 가리킵니다. 데이터 파일은 저장소에 포함하지 않습니다.

```text
data/track/
├── manifest.json
├── train/
│   ├── images/<stem>.jpg
│   └── masks/<stem>.png
├── val/
│   ├── images/<stem>.jpg
│   └── masks/<stem>.png
└── test/
    ├── images/<stem>.jpg
    └── masks/<stem>.png
```

각 split의 이미지는 `320×240` 컬러 JPEG, 마스크는 `320×240` 단일 채널 `uint8` PNG입니다. 같은 항목의 이미지와 마스크는 파일명 stem이 같아야 합니다. manifest의 `sample_id`에는 디렉터리가 아닌 stem만 기록합니다. `sample_id` 대신 정수 `frame`을 쓰면 여섯 자리 stem으로 변환합니다.

## 클래스 ID

| 값 | 의미 |
| --- | --- |
| 0 | 배경 |
| 1 | 왼쪽 차선 |
| 2 | 오른쪽 차선 |
| 3 | 횡단보도 |
| 4 | 과속방지턱 (5-class 완전 라벨에서만) |
| 255 | 무시/미라벨 픽셀 |

4-class 마스크는 `0–3`과 `255`만 사용합니다. 5-class 완전 라벨 마스크는 `0–4`와 `255`를 사용할 수 있습니다. `legacy_partial_class4` 마스크에는 클래스 `4`를 쓰지 않습니다.

## Manifest

루트 `manifest.json`에는 `schema_version`과 항목 배열 `items`가 필요합니다. `fiveclass-user-reviewed-export-v11-proportional-speedbump-split-v1` 스키마를 지원합니다. 현재 로더는 기존 Pinky/LR export 스키마도 호환 목록으로 받아들입니다. 선택 항목인 `counts`에는 split별 manifest 항목 수를 둘 수 있으며, 제공하면 실제 개수와 일치해야 합니다.

각 `items` 항목에는 다음 필드를 사용합니다.

| 필드 | 규칙 |
| --- | --- |
| `split` | `train`, `val`, `test` 중 하나 |
| `sample_id` 또는 `frame` | 파일 stem을 식별 |
| `supervision_policy` | 5-class 학습에서는 필수. 아래 정책 참조 |
| `source_video_sha256` 또는 `source_group` | 선택. 원본 녹화 그룹을 기록 |
| `image_sha256`, `mask_sha256` | 선택. 제공하면 파일 해시를 검사 |

`sample_id`는 비어 있지 않은 파일명 stem이어야 하며 경로 구분자를 포함할 수 없습니다. split마다 manifest 항목과 실제 이미지/마스크 파일이 정확히 일치해야 합니다.

## 부분 라벨 정책

5-class 모델은 항목마다 라벨 정책을 명시해야 합니다.

| 정책 | 용도 | 유효 라벨 |
| --- | --- | --- |
| `fully_labeled_5class` | 5개 클래스를 모두 구분해 라벨링 | `0–4`, `255` |
| `legacy_partial_class4` | 과속방지턱 클래스가 따로 없는 레거시 라벨 | `0–3`, `255` |

레거시 마스크의 `0`은 실제 배경일 수도, 라벨링되지 않은 과속방지턱일 수도 있습니다. 학습기는 해당 픽셀에서 배경과 과속방지턱의 합산 확률을 사용하고, 레거시 항목으로 클래스 4 Dice를 계산하지 않습니다. 이 정책은 빠진 과속방지턱 라벨을 배경으로 단정하지 않도록 합니다. `255` 픽셀은 모든 loss와 지표에서 제외합니다.

4-class 학습은 완전 라벨 4-class 데이터를 사용합니다. manifest의 정책이 없으면 `fully_labeled_4class`로 취급하며, `legacy_partial_class4`도 클래스 4가 없는 4-class 타깃으로 읽을 수 있습니다. 5-class 완전 라벨을 4-class 타깃으로 자동 축소하지는 않습니다.

## 데이터 사전 검사

```bash
pinky-check-dataset --dataset data/track --num-classes 5
```

검사는 세 split을 읽어 이미지 크기, 마스크 크기/자료형/라벨, 파일 목록과 manifest의 일치, 선택적 해시, manifest의 선택적 split 개수를 검사합니다. split을 가로지르는 완전히 동일한 이미지 해시는 오류입니다. 검사 결과에는 split별 샘플 수, 정책별 개수, 클래스가 포함된 프레임 수, manifest SHA-256, 그리고 여러 split에 걸친 원본 그룹 개수가 포함됩니다.

같은 원본 동영상에서 뽑은 프레임이 train/val/test에 나뉘어 있을 수 있습니다. 사전 검사는 이를 `source_groups_spanning_splits`로 보고하지만 자동으로 합치거나 오류 처리하지 않습니다. 따라서 프레임 단위 검증 점수는 녹화 환경이 분리된 독립 평가를 뜻하지 않습니다.
