# CycleVAD Controlled Ablation

**E8B · E12 · E11B의 원본 실험 방법을 재실행하고, 적응형 임계값·정지 경보·정지 점수 결합을 6개 구성으로 비교한 저장소입니다.** 2026-10-16 실제 학습과 평가를 완료했습니다.

이전 실험은 전체 16개 장면을 대상으로 한 탐색이었지만, 이번 실험은 원본 CycleVAD와 비교할 수 있도록 **IPAD R01–R04의 데이터 분할, 특징 추출 설정, 모델 학습, 보정, 평가 규칙을 맞추는 것**을 목적으로 했습니다. Python·GPU·라이브러리 버전의 완전한 일치는 요구하지 않았으며, 같은 로컬 실행 안에서 Baseline과 추가 구성을 비교했습니다.

- 원본 구현: [PigeonLabs/KNU_Capstone1_CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD), 고정 commit `7aa58ce893181599c4b5bd6f779fa2461ce3f103`.
- 선행 확장: [b-jongwon/cyclevad-adaptive-stall-ad](https://github.com/b-jongwon/cyclevad-adaptive-stall-ad). 이 저장소의 과거 16장면 결과를 이번 4장면 결과와 혼합하지 않습니다.
- 완료 범위: **3 seeds × 5 folds × 4 scenes × 3 experiments = 180개 실험 단위**. 각 원본 내부 비교군과 readout에 6개 추가 구성을 적용했습니다.
- 검증: **Baseline 지표 비교 15,808건**, 점수·경보 보존 및 인과적 prefix 검사, 코드·산출물 해시 확인을 통과했습니다.

[주 결과표](reports/TABLES.md) · [3개 seed 전체 해석](reports/FINAL_RESULTS.md) · [주 설정 CSV](reports/primary_all_seeds.csv) · [전체 지표 CSV](reports/all_metrics.csv) · [최종 검증](provenance/final_validation.json) · [출처](docs/ATTRIBUTION.md)

## 1. 실험에서 확인하려는 것

원본 CycleVAD의 이상 점수 위에 다음 요소를 추가했을 때 무엇이 달라지는지 확인했습니다.

1. **적응형 임계값:** 영상 시작부의 점수 수준이 평소보다 높을 때 경보 기준을 일부 올리면 오경보와 탐지율이 어떻게 달라지는가?
2. **정지 경보:** 평소 그 공정 위치에서 보이던 움직임보다 움직임이 부족한 상태가 이어질 때 별도 경보를 추가하면 놓치던 이상을 찾는가?
3. **정지 점수 결합:** 정지 증거를 연속 이상 점수에 결합하면 AUROC/AP의 순위 구분 능력도 좋아지는가?

이 세 가지 효과를 분리하기 위해 6개 구성을 사용했습니다. 임계값이나 경보만 바꾼 구성은 연속 점수가 같으므로 AUROC/AP도 같아야 합니다. 점수 결합 구성에서는 AUROC/AP와 경보 성능이 모두 달라질 수 있습니다.

## 2. 실제 실행 환경과 설정 방법

### 2.1 실제 사용한 환경

이번 실험은 **기존 Windows 프로젝트의 `.venv`를 재사용**했습니다. 팀원과 같은 Linux/Python 환경을 새로 구축해서 실행한 결과가 아닙니다. 실행 당시 확인한 값은 다음과 같습니다.

| 항목 | 실제 실행값 |
|---|---|
| 운영체제 | `Windows-11-10.0.26300-SP0` |
| Python | `3.12.10` |
| GPU | `NVIDIA GeForce RTX 4060` |
| GPU 메모리 | `약 8 GB` |
| PyTorch | `2.7.1+cu128` |
| torchvision | `0.22.1+cu128` |
| PyTorch CUDA runtime | `12.8` |
| transformers | `4.57.6` |
| NumPy | `2.2.6` |
| SciPy | `1.15.3` |
| scikit-learn | `1.6.1` |
| joblib / threadpoolctl | `1.6.0 / 3.7.0` |

정확한 기록은 [environment.json](provenance/environment.json), 당시 전체 설치 목록은 [requirements-actual.txt](provenance/requirements-actual.txt)에 있습니다. 전체 설치 목록에는 프로젝트의 다른 작업용 패키지도 포함됩니다. 이번 CycleVAD 실험의 주요 의존성은 루트의 [requirements.txt](requirements.txt)에 정리했습니다. OpenAI/Gemini API나 유료 API 호출은 이 파이프라인에 필요하지 않습니다.

### 2.2 이번 실행을 준비한 과정

1. 원본 저장소를 고정 commit으로 확보하고, 소스와 설정 파일의 SHA256을 기록했습니다.
2. 기존 CUDA 환경과 고정 DINOv2 가중치를 사용했습니다. DINOv2는 학습 중 갱신하지 않는 동결 encoder입니다.
3. 로컬 IPAD 영상 목록·프레임 수·라벨 해시를 원본 E0 기록과 비교했습니다.
4. 원본의 5분할 파일을 그대로 사용하고 분할 사이에 같은 영상이 겹치지 않는지 검사했습니다.
5. 기존 16장면 실험의 특징·학습 모델·결과를 새 실험의 Baseline으로 재사용하지 않고, 이번 조건으로 특징을 추출하고 모델을 다시 학습했습니다.
6. 정상 자료로 점수 보정과 임계값을 확정한 후 6개 구성의 결과를 계산했습니다. 테스트 라벨은 지표 계산에만 사용했습니다.

실제 실행 원본은 [provenance/original_execution](provenance/original_execution)에 보존했습니다. 공개 저장소의 `experiment/`는 다른 PC에서도 실행할 수 있도록 경로와 출력 위치를 정리한 어댑터입니다. 두 코드의 차이는 [PORTABILITY.md](docs/PORTABILITY.md)에 설명했습니다.

### 2.3 새 PC에서 환경을 준비하는 방법

다음은 이번에 사용한 주요 버전으로 새 가상환경을 만드는 **재실행 예시**입니다. 이미 준비된 환경이 있다면 그대로 사용할 수 있으며, 실제 버전은 실행 시 기록됩니다.

Windows PowerShell:

```powershell
git clone https://github.com/b-jongwon/cyclevad-controlled-ablation.git
cd cyclevad-controlled-ablation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

가상환경 활성화 명령 없이 해당 환경의 Python을 직접 호출하는 방식입니다. 위 확인에서 CUDA가 사용 가능해야 현재 `device=cuda` 설정으로 실행할 수 있습니다. CUDA 12.8 wheel 설치 명령과 실제 PC의 NVIDIA 드라이버 설치는 별개입니다.

Linux에서는 같은 저장소에서 `python -m venv .venv`로 환경을 만들고 `source .venv/bin/activate` 후 같은 `python -m pip ...` 명령을 사용할 수 있습니다. **이번 완료 결과 자체는 위 표의 Windows 환경에서 생성했습니다.**

## 3. 데이터 구성과 분할

IPAD의 R01–R04만 사용했습니다. 학습 영상은 정상 영상이며 테스트에는 정상·이상 프레임이 포함됩니다.

| 장면 | 정상 학습 영상 | 학습 프레임 | 테스트 영상 | 테스트 프레임 | 유효 평가 프레임 | 제외 프레임 |
|---|---:|---:|---:|---:|---:|---:|
| R01 | 34 | 7,808 | 15 | 3,685 | 3,685 | 0 |
| R02 | 30 | 17,756 | 15 | 9,618 | 7,706 | 1,912 |
| R03 | 22 | 15,346 | 17 | 12,005 | 12,005 | 0 |
| R04 | 25 | 9,732 | 19 | 8,154 | 8,154 | 0 |
| 합계 | 111 | 50,642 | 66 | 33,462 | 31,550 | 1,912 |

R02의 `testing/12`, `testing/13`, `testing/14`는 프레임 수와 라벨 길이가 맞지 않습니다. 원본 `strict-v1` 규칙대로 이 세 영상 전체 **1,912프레임을 모든 비교 구성에서 동일하게 평가 제외**했습니다. 라벨을 임의로 자르거나 보간하지 않았습니다. 따라서 이 결과를 전체 원본 프레임을 사용한 공식 benchmark와 동일하다고 주장하지 않습니다.

### 3.1 정상 영상 5분할의 역할

| 분할 | 역할 | 사용하지 않는 정보 |
|---|---|---|
| `fit` | tracker·정상 외형 모델·평균·subspace 학습 | 실제 test 라벨 |
| `validation` | 원본 규칙의 Fourier 차수·ridge 선택 | 실제 test 성능 |
| `reference` | 원본 성분의 정상 점수 보정, 추가 모듈의 motion 분포·초기 기준 추정 | 실제 test 라벨 |
| `threshold` | 원본 점수·정지 점수·결합 점수의 경보 임계값 산정 | 실제 test 라벨 |
| `normal_evaluation` | 위 과정에서 제외된 정상 영상의 오경보 평가 | 해당 영상으로의 재학습·임계값 재선택 |

분할은 [normal_folds.json](configs/experiments/followup/normal_folds.json)의 **영상 ID와 순서 그대로** 사용했습니다. 5개 fold를 돌면 각 정상 영상이 `normal_evaluation`에 한 번씩 포함됩니다. 파일 기준 분할의 겹침은 검사했지만, 서로 다른 파일이 같은 원본 recording에서 나왔는지는 별도 정보가 없어 입증하지 못했습니다.

아래는 fold0의 영상 수입니다. 나머지 fold의 정확한 할당은 분할 파일에 있습니다.

| 장면 | FIT | Validation | Reference | Threshold | Normal evaluation |
|---|---:|---:|---:|---:|---:|
| R01 | 15 | 4 | 4 | 4 | 7 |
| R02 | 12 | 4 | 4 | 4 | 6 |
| R03 | 8 | 3 | 3 | 3 | 5 |
| R04 | 11 | 3 | 3 | 3 | 5 |

실제 테스트 영상은 **fold0에서만** 평가했습니다. 5개 fold의 실제 test 성능을 평균 낸 결과가 아닙니다. 5개 fold는 정상 OOF 평가를 위한 구성이고, seeds 42/43/44는 모델 학습의 반복 변동을 확인하기 위한 구성입니다.

## 4. 특징 추출과 원본 모델 설정

| 항목 | 고정 설정 |
|---|---|
| Encoder | 동결 `facebook/dinov2-base` |
| 가중치 revision | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| 입력 영상 처리 | 전체 프레임을 보존하는 336×336 letterbox, 중심 crop 없음 |
| 특징 layer | −1, −3 |
| Patch 표현 | 6×6 grid, 고정 seed의 64차원 projection |
| 특징 추출 연산/저장 | CUDA FP16 autocast, 특징 캐시 FP16 |
| Batch / workers | 8 / 4 |
| 특징 캐시 | stride1과 stride2를 원본처럼 각각 추출 |
| 주 판정 cadence | stride2; 중간 원본 프레임에는 직전 점수·경보 유지 |
| 공정 위치 grid / latent | 128 / 24 |
| Fourier / ridge 후보 | {2,4,8} / {0.01,0.1,1.0} |
| 정상 FIT sample cap | 4,096, 원본의 영상 균형 sampling |
| Subspace rank 상한 / local memory | 64 / 96 |
| 통계 모델 CPU thread 제한 | 4 |

정확한 기본 설정은 [replay_pinned.json](configs/experiments/replay_pinned.json)에 있습니다. 실제 retained rank는 원본의 분산·표본 수 규칙에 따라 달라질 수 있습니다. E12는 별도의 공통 총 rank 규칙으로 shared/phase-specific PCA의 용량을 통제합니다.

DINOv2 전체를 fine-tuning하거나 AE/GRU를 추가로 학습한 실험이 아닙니다. 여기서 학습하는 부분은 원본 tracker, Fourier 평균, PCA/subspace, local memory 및 정상 자료의 보정 통계입니다. Encoder 가중치와 특징 projection은 seed42 설정으로 고정하며 모델 seed를 바꿔도 특징 추출 조건은 바꾸지 않습니다.

## 5. 실제 실험 진행 순서

```text
IPAD 영상 목록·라벨 해시·분할 검사
  → 고정 DINOv2로 stride2 / stride1 특징 추출
  ├ E8 정상 모델 학습 → E8B 위치 조건 대조 실험
  └ E9S 시간 진행 모델 학습 → E11 통합 모델
                              ├ E12 평균·subspace 비교
                              └ E11B A/C/P/CP 성분 비교
  → 각 원본 비교군의 정상 reference/threshold로 추가 모듈 보정
  → 동일 Baseline 위에 고정된 6개 구성 적용
  → 실제 test 및 정상 holdout 지표 계산
  → 영상별·장면별 수치, seed별 Macro, 일치·인과성 검사 저장
```

seed42를 먼저 실행하고, 그 정상 validation으로 고른 Fourier/ridge 선택을 원본 규칙에 따라 seed43/44에 고정했습니다. 각 seed 안에서는 fold0부터 fold4까지, 각 fold 안에서는 R01–R04 순서로 진행했습니다. 설정 선택이나 추가 모듈의 파라미터를 이번 test 결과를 보고 바꾸지 않았습니다.

특징 추출은 **177개 영상 × 2개 stride = 354개 캐시**를 생성했습니다. 이어서 원본 의존 단계와 6개 구성을 실행했습니다. 완료 기록의 전체 실행 시간은 **79.26분**입니다. 이 값은 해당 PC에서의 특징 추출·학습·평가를 포함한 한 번의 실행 시간이며, 온라인 추론 FPS나 다른 PC의 예상 소요 시간을 뜻하지 않습니다.

### 5.1 E8B·E12·E11B를 각각 어떤 기준으로 사용했는가?

| 실험 | 원본 비교 목적 | 이번 주 표의 사전 고정 Baseline | 보존한 원본 비교 범위 |
|---|---|---|---|
| E8B | 공정 위치 조건부 평균이 도움이 되는지, 시간/관측/필터 위치와 위치 교란 대조 비교 | `M3__conditional_only` | M0/M1/M2/M3, 5개 M4 offset, seed42의 fixed-rank·retuned 대조 및 두 readout |
| E12 | 평균 표현과 shared/phase-specific subspace, 정상 FIT 자료량의 영향 비교 | `S1_f100__head_only` | S0–S5 계열, 25/50/100% nested FIT subset, head 단독 및 A와의 max 결합 |
| E11B | 일반 외형 A, 조건부 외형 C, 진행 이상 P 및 결합의 성분 비교 | `CP=max(C,P)` | B0/B1/B2/B3/C/P/CP의 7개 readout |

E8B의 M3는 인과적으로 필터링한 위치를 사용하는 조건부 외형입니다. E12의 S1은 연속 평균과 공유 잔차 PCA이며, 여기서 `f100`은 해당 head에 전체 FIT subset을 사용했다는 뜻입니다. E12의 tracker와 일반 외형 A는 subset 비율과 관계없이 원본처럼 전체 FIT로 학습한 상태를 공유하므로 **전체 시스템의 few-shot 실험이라고 해석하지 않습니다.**

E12에서 특정 subset의 rank나 phase-bin 영상 coverage가 부족하면 원본 규칙대로 unavailable로 남겼습니다. 다른 모델을 대신 넣어 비교 단위 수를 맞추지 않았습니다. 추가 모듈은 각 실험의 원본 내부 비교군/readout 모두에 붙였으며, README의 세 주 표만 실행한 것은 아닙니다.

원본 E9S/E11/E11B가 생성한 **합성 시간 교란 결과도 원본 재실행 산출물로 보존**했습니다. 다만 이번 추가 모듈 6개 구성의 주 비교표는 실제 test와 정상 영상 평가에서 만든 결과입니다. 추가 모듈을 붙인 합성 교란 전체를 새로 평가했다고 주장하지 않습니다.

## 6. 추가한 6개 구성과 계산 방식

### 6.1 구성별로 실제 바뀌는 부분

| 구성 | 연속 이상 점수 | 경보 기준 | 별도 정지 경보 |
|---|---|---|---|
| Baseline | 원본 `s(t)` | 원본 고정 T | 없음 |
| Adaptive threshold only | 원본 `s(t)` | 시작부 기준으로 일부 상승한 Tdynamic | 없음 |
| Stall alarm only | 원본 `s(t)` | 원본 고정 T | OR 결합 |
| Stall score fusion only | 결합 `f(t)` | 정상 자료로 다시 구한 Tf | 없음 |
| Adaptive threshold + Fusion | 결합 `f(t)` | 동적 Tf | 없음 |
| Adaptive threshold + Fusion + Stall alarm | 결합 `f(t)` | 동적 Tf | OR 결합 |

같은 실험·seed·fold·원본 readout의 6개 행은 **같은 특징 캐시, 학습 모델, 데이터 분할, 정지 보정 모델**을 공유합니다. 달라지는 것은 위 표의 점수 결합·임계값 적응·정지 경보 적용 여부입니다. 결합 점수에 맞춰 임계값을 다시 구하는 것은 의도된 처리의 일부이며, 테스트 라벨을 이용한 재최적화가 아닙니다.

### 6.2 정지 점수 u(t)

현재 global 특징 `g(t)`와 과거 특징 사이의 cosine 움직임을 사용합니다.

```text
motion_lag(t) = 1 − dot(g(t), g(t−lag))
lag = 1, 4 fresh observations
```

주 cadence가 stride2이므로 lag1/4는 원본 프레임 간격 2/8에 해당합니다. 이 값은 optical flow나 객체 추적 결과가 아니라 동결 DINOv2 특징의 변화량입니다.

정상 `reference`에서 공정 위치 θ를 16개 bin으로 나누어 각 lag의 lower-tail 분포를 추정합니다. 해당 bin에 유효 표본이 64개 미만이면 전체 정상 reference 분포로 fallback합니다. 낮은 motion이 해당 위치의 정상 분포에서 얼마나 드문지 lower-tail surprisal로 바꾼 뒤 두 lag의 **min**을 취하고 EMA α=0.5를 적용합니다. 두 시간 간격에서 모두 움직임 부족 증거가 있어야 높은 점수가 나오는 구성입니다.

정지 임계값 `Tstall`은 정상 `threshold` 영상의 전체 프레임 기준 **q99.5**입니다. `u(t)>Tstall`이 새 관측 3개에서 연속으로 성립하면 정지 경보를 켭니다. 영상이 바뀌면 연속 횟수와 시간 상태를 초기화합니다.

E8B의 위치 교란 대조군에서도 stall 보정에는 공통의 원래 tracker θ를 사용했습니다. 대조군의 좌표 교란 때문에 추가 모듈까지 동시에 바뀌는 혼선을 피하기 위한 통제입니다.

### 6.3 기존 정지 코드의 시작부 인과성 수정

선행 확장의 motion 초기화에는 lag가 없는 시작 관측을 뒤의 첫 유효 motion 값으로 채우는 부분이 있었습니다. 이번 실험에서는 이를 제거했습니다.

- lag가 없는 표본은 정지 분포 fitting에서 제외합니다.
- 최초 4개 관측의 정지 점수는 0입니다.
- 이후에도 현재와 과거 특징만 사용합니다.
- 짧은 prefix만 넣었을 때의 점수·경보·임계값이 전체 영상을 넣었을 때의 같은 prefix와 일치하는지 검사합니다.

이는 추가 모듈의 초기화 수정이며 **원본 Baseline 점수·경보를 변경하지 않습니다.**

### 6.4 정지 점수의 연속 점수 결합

```text
b(t) = s(t) / T
v(t) = u(t) / Tstall
f(t) = max(b(t), 1.1 × v(t))
Tf   = 정상 threshold 자료에서 f의 해당 실험 주 규칙으로 재보정
```

구현에서는 분모를 최소 `1e−6`으로 둡니다. 선행 확장의 이름 `max_mix_0.1`은 **정지 점수를 10% 가중 평균한다는 뜻이 아닙니다.** 정지 분기를 1.1배한 뒤 원본 정규화 점수와 max를 취합니다. 정규화 값은 정상 경보 기준 대비 강도이며 불량 확률이 아닙니다.

### 6.5 적응형 임계값

첫 24개 새 관측 중 낮은 20%의 중앙값을 시작부 수준으로 사용합니다. 정상 reference 영상에서 같은 방식으로 구한 값들의 중앙값을 기준 수준으로 둡니다.

```text
delta = max(0, 현재 영상 시작부 수준 − 정상 reference 기준 수준)
Tdynamic(t) = T                     # 첫 23개 관측
              T + 0.3 × delta       # 24번째 관측부터
```

결합 구성은 T 대신 Tf와 결합 점수의 reference 기준을 사용합니다. 24개 관측이 되기 전에는 미래 시작부를 미리 읽지 않습니다. 24개보다 짧은 영상은 고정 임계값을 유지합니다. 이 적응은 임계값을 올리는 방향이므로 오경보와 함께 이상 탐지율도 낮아질 수 있습니다.

### 6.6 실험별로 유지한 임계값 산정 규칙

| 항목 | 정상 threshold 자료에서의 산정 |
|---|---|
| E8B Baseline / Fusion | 전체 프레임 q99 |
| E12 Baseline / Fusion | 전체 프레임 q99 |
| E11B Baseline / Fusion | 정상 영상별 동일 가중 D-window 최대값 q99 |
| 세 실험의 정지 분기 Tstall | 전체 프레임 q99.5 |

E11B의 D는 원본 FIT 공정 주기 Tperiod의 약 20%입니다. 정상 영상의 사전 지정 위치에서 D-window 최대값을 구하고, 영상마다 총 가중치가 같게 분위수를 계산합니다. **창 최대값 q99와 프레임 q99는 서로 다른 운영점**입니다. 이를 하나의 규칙으로 통일하면 원본 실험과 달라지므로 그대로 유지했습니다. 명목 q99가 실제 test의 frame FPR=1%를 보장하지도 않습니다.

## 7. 평가 지표와 평균 방식

| 지표 | 이 실험에서의 의미 |
|---|---|
| AUROC | 연속 점수로 정상·이상 프레임의 순서를 구분하는 능력 |
| AP | 연속 점수 기반 precision–recall 요약 지표 |
| FPR | 정상 프레임 중 실제 경보가 켜진 비율, `FP/(FP+TN)` |
| Recall | 이상 프레임 중 실제 경보가 켜진 비율, `TP/(TP+FN)` |
| F1 | 실제 경보에서 `2TP/(2TP+FP+FN)` |

유효 프레임만 사용해 **장면별로 먼저 지표를 계산한 뒤 R01–R04를 동일 가중 평균**합니다. 영상 전체를 합친 global micro 평균과 다릅니다. Macro F1은 장면별 F1의 평균이며 Macro precision과 Recall에서 다시 계산하지 않습니다. 임계값 비교는 strict `score > threshold`입니다.

stride2에서 계산한 점수와 경보는 중간 원본 프레임에 직전 값을 유지한 뒤 전체 유효 프레임에서 평가합니다. 따라서 짝수 프레임만 평가해 분모가 줄어드는 방식이 아닙니다. 정지 연속 횟수는 held frame이 아닌 새 관측에서만 증가합니다.

영상별 event coverage와 탐지 지연도 상세 수치에 보존합니다. 이벤트는 영상별 연속 이상 라벨 구간으로 계산하므로 서로 다른 영상의 경계를 이어 붙여 하나의 이벤트로 세지 않습니다. README 주 표는 요청된 AUROC/AP/FPR/Recall/F1을 표시합니다.

## 8. 주 결과 — seed42

실제 test의 fold0 결과이며 4장면 Macro, 단위는 %입니다. 각 표의 Baseline은 이번 환경에서 다시 학습한 모델입니다. 팀원 공개 수치를 그대로 옮긴 값이 아닙니다.

### E8B — `M3__conditional_only`

| Configuration | AUROC | AP | FPR ↓ | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.40 | 72.46 | 16.76 | 44.82 | 44.66 |
| + Adaptive threshold only | 79.40 | 72.46 | 15.66 | 44.07 | 44.21 |
| + Stall alarm only | 79.40 | 72.46 | 17.01 | 49.66 | 49.22 |
| + Stall score fusion only | 79.62 | 74.66 | 16.85 | 48.54 | 47.64 |
| + Adaptive threshold + Fusion | 79.62 | 74.66 | 11.39 | 41.04 | 44.58 |
| + Adaptive threshold + Fusion + Stall alarm | 79.62 | 74.66 | 11.49 | 41.27 | 44.70 |

### E12 — `S1_f100__head_only`

| Configuration | AUROC | AP | FPR ↓ | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.39 | 72.50 | 16.79 | 44.97 | 44.78 |
| + Adaptive threshold only | 79.39 | 72.50 | 15.66 | 44.16 | 44.29 |
| + Stall alarm only | 79.39 | 72.50 | 17.04 | 49.81 | 49.34 |
| + Stall score fusion only | 79.66 | 74.75 | 16.77 | 48.69 | 47.84 |
| + Adaptive threshold + Fusion | 79.66 | 74.75 | 10.77 | 40.43 | 44.69 |
| + Adaptive threshold + Fusion + Stall alarm | 79.66 | 74.75 | 10.90 | 40.72 | 44.85 |

### E11B — `CP`

| Configuration | AUROC | AP | FPR ↓ | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.79 | 73.26 | 12.34 | 36.57 | 36.67 |
| + Adaptive threshold only | 79.79 | 73.26 | 4.54 | 17.10 | 26.59 |
| + Stall alarm only | 79.79 | 73.26 | 12.59 | 41.41 | 42.63 |
| + Stall score fusion only | 79.05 | 74.06 | 0.75 | 11.53 | 19.41 |
| + Adaptive threshold + Fusion | 79.05 | 74.06 | 0.73 | 11.04 | 18.56 |
| + Adaptive threshold + Fusion + Stall alarm | 79.05 | 74.06 | 0.99 | 12.90 | 21.51 |

### 8.1 주 결과의 해석

**정지 경보만 추가한 구성**은 seed42의 세 주 실험에서 가장 높은 F1을 보였습니다. E8B는 44.66→49.22, E12는 44.78→49.34, E11B는 36.67→42.63입니다. 대신 FPR도 각각 약 0.25%p 늘었습니다. 이는 같은 오경보 수준에서의 개선을 입증한 결과가 아니라 탐지–오경보 교환 관계를 포함한 결과입니다.

**E8B/E12의 정지 점수 결합**은 AP를 높였습니다. 적응형 임계값까지 붙이면 FPR이 더 낮아지지만 Recall도 감소해, 전체 결합의 F1 이득은 매우 작아집니다. 이 두 실험에서 AP 개선과 경보 F1 개선을 같은 의미로 해석하면 안 됩니다.

**E11B 전체 결합**은 AP가 73.26→74.06으로 올랐지만 AUROC는 79.79→79.05로 떨어졌고, Recall은 36.57→12.90, F1은 36.67→21.51로 낮아졌습니다. FPR 12.34→0.99만으로 좋은 설정이라고 결론 내릴 수 없습니다. E11B는 창 최대값 기반으로 결합 임계값을 재보정하므로 프레임 q99를 쓰는 다른 표와 운영점이 다릅니다. 관측된 탐지 손실은 결과에 그대로 남겼으며 이를 숨기기 위해 임계값을 다시 고르지 않았습니다.

### 8.2 seed43·44에서도 같은 방향인가?

아래는 각 seed에서 **전체 결합 − Baseline**의 차이입니다. 단위는 percentage point(%p)이며, FPR은 음수가 감소를 뜻합니다.

| 실험 | Seed | ΔAUROC | ΔAP | ΔFPR | ΔRecall | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| E8B | 42 | +0.22 | +2.20 | -5.27 | -3.56 | +0.04 |
| E8B | 43 | +0.31 | +2.27 | -4.80 | -3.25 | -0.20 |
| E8B | 44 | +0.19 | +2.02 | -6.27 | -3.18 | +0.73 |
| E12 | 42 | +0.27 | +2.25 | -5.89 | -4.25 | +0.07 |
| E12 | 43 | +0.29 | +2.27 | -5.64 | -4.23 | -0.17 |
| E12 | 44 | +0.22 | +2.06 | -5.01 | -2.18 | +0.54 |
| E11B | 42 | -0.73 | +0.79 | -11.34 | -23.68 | -15.17 |
| E11B | 43 | -0.71 | +0.83 | -11.58 | -23.67 | -15.07 |
| E11B | 44 | -0.85 | +0.38 | -11.43 | -23.97 | -15.72 |

전체 결합의 AP 상승은 세 실험의 세 seed 모두에서 유지됐습니다. E8B/E12의 전체 결합 F1은 seed43에서 Baseline보다 낮았고, E11B의 AUROC·Recall·F1 하락은 세 seed에서 공통으로 나타났습니다. 따라서 모든 seed·장면·지표에서 개선됐다는 주장은 하지 않습니다. seed별 점 추정의 일관성은 확인했지만 이번 추가 모듈 차이에 대한 별도 신뢰구간이나 통계적 유의성을 제시한 것은 아닙니다.

모든 원본 내부 대조군과 장면별 결과는 [전체 CSV](reports/all_metrics.csv)에, 세 주 설정의 seed별 수치는 [primary_all_seeds.csv](reports/primary_all_seeds.csv)에 있습니다. 특정 설정을 이번 test 결과로 다시 선택해 최종 방법으로 확정하지 않았습니다.

## 9. 재실행·재개·단계별 실행

### 9.1 데이터 배치와 사전 점검

원본 IPAD를 별도로 준비합니다. 데이터 폴더는 다음 구조여야 합니다.

```text
IPAD_dataset/
  R01/
    training/frames/01/0.jpg ...
    testing/frames/01/0.jpg ...
    test_label/001.npy ...
  R02/ ...
  R03/ ...
  R04/ ...
```

프레임 이름은 원본처럼 0부터 연속된 숫자여야 하며 영상 ID와 프레임 수·라벨 SHA256이 기록과 맞는지 검사합니다. 아래 PowerShell의 데이터 경로는 자신의 실제 폴더로 바꿉니다.

```powershell
.\.venv\Scripts\python.exe experiment/run.py --stage prepare --data-root "D:\data\IPAD_dataset" --output-root work
```

### 9.2 전체 실행

```powershell
.\.venv\Scripts\python.exe experiment/run.py --data-root "D:\data\IPAD_dataset" --output-root work
```

기본값은 seeds 42/43/44, folds 0–4, scenes R01–R04입니다. **새 특징·모델·평가 결과는 `work/`에 저장**되며 공개된 `results/`의 완료 수치를 덮어쓰지 않습니다. 고정 DINOv2 revision은 로컬 캐시에 없으면 처음 실행할 때 다운로드합니다. 오프라인으로 실행하려면 해당 revision을 미리 캐시에 준비하고 관련 Hugging Face offline 설정을 사용할 수 있습니다.

같은 명령으로 재개하면 완료된 특징과 실험 단위를 건너뜁니다. 실행 코드·설정·protocol이 달라진 상태에서 같은 출력 폴더에 이어 쓰려 하면 중단합니다. 방법을 바꾸어 새 실험을 할 때는 별도 `--output-root`를 사용해야 합니다.

### 9.3 부분 실행과 단계 구분

```powershell
# 원본과 같은 데이터 사전 점검 후, seed42/fold0/R01의 모델·평가만 확인
.\.venv\Scripts\python.exe experiment/run.py --data-root "D:\data\IPAD_dataset" --output-root work --seeds 42 --folds 0 --scenes R01

# 특징 추출만 실행
.\.venv\Scripts\python.exe experiment/run.py --stage extract --data-root "D:\data\IPAD_dataset" --output-root work

# 특징 추출 완료 후 원본 학습·평가 단계만 실행
.\.venv\Scripts\python.exe experiment/run.py --stage stats --data-root "D:\data\IPAD_dataset" --output-root work

# 필요한 원본 단계까지 완료한 후 6개 추가 구성 계산
.\.venv\Scripts\python.exe experiment/run.py --stage addons --data-root "D:\data\IPAD_dataset" --output-root work

# 계산된 결과로 표만 다시 생성
.\.venv\Scripts\python.exe experiment/run.py --stage report --output-root work
```

현재 구현에서 `--scenes`, `--seeds`, `--folds`는 모델·추가 평가 단위를 제한합니다. 데이터 점검과 특징 추출은 전체 R01–R04에 대해 수행하므로 R01 부분 실행을 해도 특징 캐시는 네 장면 모두 준비합니다. seed43/44만 실행하려면 해당 fold의 seed42 원본 학습 결과가 먼저 필요합니다. 보통 기본 전체 명령을 사용하는 것이 가장 간단합니다.

### 9.4 진행 및 완료 확인

| 파일 | 의미 |
|---|---|
| `work/feature_progress.json` | 354개 특징 캐시 중 진행 위치 |
| `work/feature_complete.json` | 전체 특징 추출 완료 |
| `work/progress.json` | 현재 학습 단계·seed·fold·장면 |
| `work/ablation/.../complete.json` | 해당 추가 실험 단위 완료·검사 상태 |
| `work/failure.json` | 예외 발생 시 오류와 traceback |
| `work/all_completion.json` | 이번 실행에서 요청한 seed/fold/scene 범위 완료 |

부분 실행도 `all_completion.json`을 만들 수 있으므로 **파일 존재만으로 전체 완료라고 판단하지 않고 그 안의 seeds/folds/scenes를 확인**해야 합니다. 현재 공개한 완료 기록은 전체 3×5×4 범위입니다. 콘솔 로그는 직접 리다이렉트할 수 있지만 공개 실행 명령이 `execution.log`라는 파일을 자동으로 만드는 것은 아닙니다.

## 10. 검증은 무엇을 확인했는가?

| 검증 | 확인한 내용 |
|---|---|
| 원본 소스·설정 manifest | 원본 commit에서 가져온 파일의 byte 해시 보존 |
| 데이터·분할 검사 | 177개 영상의 목록·프레임 수·라벨 해시, 분할 겹침과 정상 OOF coverage |
| Baseline 점수·경보 검사 | 추가 모듈을 끈 행이 이번 원본 재실행의 점수·경보와 동일 |
| Baseline 지표 15,808건 비교 | 구성·분할별 AUROC/AP/FPR/Recall을 이번 원본 재실행과 대조; 수치 허용오차 1e−12 |
| Prefix 검사 | 시작부·lag·적응 경계에서 미래 관측을 추가해도 과거 출력이 바뀌지 않는지 확인 |
| 테스트 77개 | 원본 unit/smoke tests와 추가 구성의 인과성·지표 테스트 통과 |
| 공개 산출물 검사 | 2,644개 수치 파일의 저장·압축 전후 해시, 180개 단위와 54개 주 Macro 행 확인 |
| 공개용 prepare | 공개 경로 어댑터로 실제 데이터 177개 사전 점검 통과 |

**15,808건은 독립적인 실험 반복 횟수가 아니라 지표 비교 레코드 수**입니다. 같은 로컬 학습 결과에서 Baseline이 보존됐다는 검사이며, 팀원의 공개 AUROC/AP를 비트 단위로 재현했다는 뜻은 아닙니다. 원본 공개 수치와의 차이는 각 단위의 `upstream_baseline_deltas.json`에 별도로 저장했습니다.

새 PC에서 공개 산출물과 코드 해시를 확인하려면:

```powershell
.\.venv\Scripts\python.exe scripts/verify_results.py
.\.venv\Scripts\python.exe -m pytest tests experiment/test_addons.py -q
```

`verify_results.py`는 공개된 완료 수치를 검사하며 모델을 새로 학습하지 않습니다. 실제 완료 실험 이후 공개용 어댑터로 전체 79분짜리 실험을 다시 돌린 것은 아닙니다. 원본 실행 코드와 완료 수치를 보존하고 어댑터의 단위 테스트·실데이터 prepare·해시 검증을 수행했습니다.

## 11. 저장소 구조와 상세 결과 읽기

```text
src/cycle_vad/             원본 tracker·모델·각 실험 함수
configs/                  고정 encoder/model 설정, 영상 분할, 원본 실험 계획
experiment/               공개용 실행 어댑터와 6개 구성, 추가 테스트
tests/                    원본 테스트
scripts/verify_results.py 공개 수치·원본 코드 해시 검증
results/replay/            이번 원본 방법 재학습·평가 수치
results/ablation/          6개 구성의 scene/영상별 지표 및 검증 기록
reports/                  주 표, 전체 CSV, 3-seed 보고서와 Macro
provenance/               실제 환경, 원본 실행 코드, 완료·검증·압축 manifest
docs/                     방법, 출처, 공개 경로 변경 설명
work/                     사용자가 재실행할 때 만드는 로컬 출력(추적 제외)
```

큰 상세 JSON은 `.json.gz`로 무손실 압축해 보관합니다. 예를 들어 전체 Macro 파일의 실제 경로는 `reports/macro.json.gz`입니다. `.json` 파일이 없을 때 `.json.gz`를 확인해야 합니다. 저장 경로와 압축 전후 크기·SHA256은 [artifact_manifest.json](provenance/artifact_manifest.json)에 있습니다.

```python
import gzip
import json
from pathlib import Path

path = Path("reports/macro.json.gz")
with gzip.open(path, "rt", encoding="utf-8") as f:
    macro = json.load(f)

rows = [r for r in macro
        if r["experiment"] == "E11B"
        and r["base_arm"] == "CP"
        and r["seed"] == 42]
for row in rows:
    print(row["configuration"], row["auroc"], row["ap"], row["fpr"], row["recall"], row["f1"])
```

JSON/CSV의 지표 값은 0–1 비율이고 README 표는 이를 100배한 %입니다. 영상별 결과는 `results/ablation/<실험>/seed<번호>/<장면>/fold<번호>/per_video.json[.gz]`, scene별 지표는 같은 위치의 `metrics.json[.gz]`에 있습니다. 원본 내부 대조군의 unavailable 여부는 `results/replay/`의 해당 `run.json`에서 확인할 수 있습니다.

## 12. 공개 범위와 결과 해석의 한계

- IPAD 이미지·영상·원본 ZIP, 특징 캐시, 프레임별 NPZ 예측, 학습 checkpoint와 인증 정보는 포함하지 않습니다. 이 파일들은 원래 로컬 실행 폴더에 보존했습니다. 이 저장소는 코드·설정·전체 수치 JSON/CSV·검증 기록을 공개합니다.
- 원본 공개 저장소에 팀원의 특징 캐시·학습 checkpoint·영상 content hash가 없어 영상 픽셀이나 학습 결과의 byte 동일성은 증명하지 못합니다. 실제 사용한 소프트웨어와 하드웨어 차이는 기록했습니다.
- Baseline과 추가 모듈은 같은 로컬 실행 안에서 비교했으므로 이번 추가 기능의 효과를 볼 수 있습니다. 팀원 공개 수치와 이번 수치 사이의 차이를 전부 추가 모듈의 효과라고 해석하면 안 됩니다.
- 기존 영상에 대한 탐색이며 선행 테스트 결과가 모듈 설계에 영향을 주었습니다. 새 데이터의 독립 확인 실험이나 모든 지표의 일관된 우월성을 입증한 결과가 아닙니다.
- 실제 cycle/단계 ground truth와 recording group 정보가 없어 공정 위치 정확도와 원본 촬영 단위 독립성을 주장하지 않습니다.
- FPR 감소, 순위 지표 개선, 실제 이상 탐지율 개선은 서로 다른 관찰입니다. 특히 E11B에서 나타난 탐지 손실을 포함해 해석해야 합니다.

기본 CycleVAD 구현의 출처와 이번 변경 범위는 [ATTRIBUTION.md](docs/ATTRIBUTION.md)에 정리했습니다. 이번 저장소의 추가 비교를 원본 CycleVAD 전체의 새로운 발명으로 표현하지 않습니다.
