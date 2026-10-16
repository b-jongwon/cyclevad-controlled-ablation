# CycleVAD Controlled Ablation

**E8B · E12 · E11B의 실험 방법을 맞춘 적응형 임계값·정지 경보·정지 점수 결합 비교.** 2026-10-16 실제 재학습 및 평가 완료.

- 원본: [PigeonLabs/KNU_Capstone1_CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD), commit `7aa58ce893181599c4b5bd6f779fa2461ce3f103`.
- 이전 확장: [b-jongwon/cyclevad-adaptive-stall-ad](https://github.com/b-jongwon/cyclevad-adaptive-stall-ad). 이전 전체16장면 탐색과 이번 R01–R04 통제 실험은 구분한다.
- 3 seeds(42/43/44) × 5 folds × 4 scenes × 3 experiments = **180개 실험 단위** 완료. 각 원본 내부 비교군/readout에 6개 구성을 적용했다.
- **Baseline 지표 일치 검사 15,808건**, 원본/실행 코드 해시 및 인과적 prefix 검증 통과. 이 검사는 이번 로컬 Baseline 보존 검사이며 팀원 공개 수치와의 완전 동일성 증명은 아니다.

[전체 3-seed 결과](reports/FINAL_RESULTS.md) · [전체 지표 CSV](reports/all_metrics.csv) · [주 설정 3-seed CSV](reports/primary_all_seeds.csv) · [검증 기록](provenance/final_validation.json) · [방법·변인통제](docs/METHODS.md) · [출처](docs/ATTRIBUTION.md)

## 핵심 관찰

- seed42에서 **정지 경보만 추가하면 세 주 실험 모두 F1 상승**, FPR도 소폭 상승했다.
- 전체 결합의 AP 상승은 3개 seed 모두 유지됐다. E8B/E12의 전체 결합 F1 개선은 seed43에서 유지되지 않았다.
- **E11B 전체 결합은 3개 seed 모두 AUROC/Recall/F1 하락.** 낮은 FPR과 탐지 손실을 함께 보고해야 한다.
- 이번 실행에서 테스트 결과로 파라미터를 다시 고르지 않았다. 추가 모듈의 설계는 과거 탐색에 영향을 받았으므로 독립 확인 실험이라고 주장하지 않는다.

## 주 결과 — seed42

R01–R04 실제 test의 scene-equal Macro, %. 실제 test는 fold0만 사용한다. AUROC/AP는 연속 점수, FPR/Recall/F1은 각 구성의 실제 경보로 계산한다.

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

## 실행

CUDA 사용 가능한 Python 환경에서 저장소 루트 기준:

```bash
python -m venv .venv
# 가상환경 활성화 후: 이번 실행에서 사용한 CUDA 패키지 예시
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
python experiment/run.py --data-root /path/to/IPAD_dataset --output-root work
```

위 명령은 기본 3 seeds × 5 folds × 4 scenes를 실행한다. 원본 내부 대조군과 의존 단계(E8/E9S/E11)도 실행한다. 동일 명령으로 완료 단위를 건너뛰며 재개한다. 새 결과·캐시·모델은 `work/`에 저장되며 공개된 `results/`를 덮어쓰지 않는다. 가중치는 고정 revision으로 처음 실행 시 다운로드한다. 버전/OS/GPU 일치는 실험 요건이 아니며 실제 사용 환경을 기록한다.

짧은 실행 점검:

```bash
python experiment/run.py --stage prepare --data-root /path/to/IPAD_dataset --output-root work
python experiment/run.py --data-root /path/to/IPAD_dataset --output-root work --seeds 42 --folds 0 --scenes R01
```

이 부분 실행의 결과를 전체 4장면 Macro로 표시하지 않는다.

검증:

```bash
python -m pytest tests experiment/test_addons.py -q
python scripts/verify_results.py
```

## 결과와 재현 기록

- `results/replay/`: 이번 원본 방법 재실행 수치. 팀원 기존 수치를 복사해 새 실험으로 표시하지 않았다.
- `results/ablation/`: 6개 구성의 scene/영상별 지표, 완료 기록, Baseline 일치 검사 및 팀원 공개 수치와의 차이.
- `reports/`: 주 비교표, 3-seed 해석, 전체 CSV/JSON.
- `provenance/`: 원본 commit/source manifest, 당시 실행 코드, 실험 계획, 실제 환경 및 최종 검증.
- 큰 상세 JSON은 `.json.gz`로 무손실 압축했다. Python `gzip.open(path,"rt",encoding="utf-8")`으로 읽을 수 있다. 압축 전후 SHA256과 크기는 `provenance/artifact_manifest.json`에 있다.
- IPAD 원본 이미지/영상/ZIP, 특징 캐시, 프레임별 NPZ, 학습 checkpoint 및 인증 정보는 포함하지 않는다. 해당 산출물은 로컬 실행 폴더에 보존했다.

## 비교의 범위

원본과 영상 목록/프레임 수/라벨 SHA256/분할/설정/학습·평가 방법을 맞췄다. 비교군들은 같은 실행의 특징과 학습 모델을 공유한다. 공개 원본에 원본 특징 캐시/checkpoint/영상 content hash가 없어 픽셀·모델·수치의 비트 단위 동일성은 증명하지 않는다. 기존 데이터에 대한 탐색이며 recording group 독립성도 확인되지 않았다.
