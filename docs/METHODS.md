# CycleVAD 동일 실험 방법 재실행 — 2026-10-16

사용자 최종 요청에 따라 **버전·OS·GPU를 바꾸지 않고 기존 실행 환경을 사용**한다. 맞추는 대상은 데이터, 분할, 특징 추출 설정, 모델, 학습, 보정, 평가 방법이다. 원본 저장소는 수정하지 않는다.

기준: PigeonLabs/KNU_Capstone1_CycleVAD 커밋 `7aa58ce893181599c4b5bd6f779fa2461ce3f103`. 이전 로컬 커밋 `075e21d`와 비교했을 때 E8B/E12/E11B 및 그 의존 학습 모듈은 동일하다. 기존 b-jongwon 확장의 `max_mix_0.1`을 사전 고정해 적용한다. 과거 테스트를 보고 정했던 방식이므로 독립 확인 실험이라고 쓰지 않는다.

## 원본과 일치시키는 항목

- IPAD **R01–R04**. 이전 전체 16장면 평균을 팀원 4장면 평균과 섞지 않는다.
- `normal_folds.json`의 FIT/validation/reference/threshold/normal_evaluation 파일, 순서 및 5개 fold 그대로.
- seeds **42/43/44**. seed42 정상 validation으로 고른 Fourier/ridge를 다른 seed에서 고정한다.
- DINOv2-base revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, 336 letterbox, layers −1/−3, 6×6 patch grid, patch projection dim64, FP16, batch8/workers4. stride1/stride2 특징은 원본처럼 각각 추출한다.
- E8→E8B 및 E9S→E11→E12/E11B 의존 학습을 원본 함수로 실행한다. **원본 내부 비교군, E12의 25/50/100% FIT subset·공통 rank·두 readout을 전부 보존**한다. E12가 coverage 부족으로 unavailable인 경우 다른 모델로 대체하지 않는다.
- E8B/E12의 주 임계값은 정상 threshold **frame q99**. E11B는 정상 영상별 동일 가중 **D-window maximum q99**. Fusion도 각 실험의 같은 규칙으로 재보정한다.
- 실제 test는 fold0만 사용. R02 testing12/13/14의 불일치 라벨은 원본 strict-v1대로 영상 전체 제외(1,912프레임). 원본 프레임으로 이전 점수를 유지해 평가한다.
- scene별 모든 유효 프레임으로 AUROC/AP/FPR/Recall/F1을 계산한 뒤 R01–R04 동일 가중 Macro. F1은 실제 alarm의 TP/FP/FN으로 계산하며 Macro precision/recall에서 역산하지 않는다.

## 사진의 6개 행

| 구성 | 연속 점수 | 경보 |
|---|---|---|
| Baseline | 원본 s | s > 원본 T |
| + Adaptive threshold only | 원본 s | s > 동적 T |
| + Stall alarm only | 원본 s | 원본 경보 OR 정지 경보 |
| + Stall score fusion only | f | f > 정상 재보정 Tf |
| + Adaptive threshold + Fusion | f | f > 동적 Tf |
| + Adaptive threshold + Fusion + Stall alarm | f | 동적 fusion 경보 OR 정지 경보 |

`f=max(s/T, 1.1×u/Tstall)`. 정지 u는 정상 reference의 phase16-bin lower-tail lag1/4 motion을 min 결합하고 EMA0.5 적용. Tstall은 정상 threshold의 frame q99.5. 정지 경보는 초과가 **3개 새 관측 연속**일 때 발생한다. 관측 stride2이므로 lag1/4는 원본 프레임 간격 2/8이다. phase bin이 64표본 미만이면 전체 정상 reference로 fallback한다.

적응형 임계값은 처음 24관측 중 낮은20%의 중앙값과 정상 reference 초기 기준의 양의 차이에 gain0.3을 곱해 올린다. 24번째 관측부터 적용하고 그 이전에는 기본 임계값을 사용한다. 짧은 영상의 미래/전체 길이를 읽어 조정하지 않는다.

기존 확장의 미래 참조 motion 시작부 초기화는 수정한다. lag가 없는 관측을 reference fitting에서 제외하고 최초4관측의 정지 점수를 0으로 둔다. 모델에 정답, 합성 onset, 미래 프레임을 입력하지 않는다. **원본 baseline은 이 수정의 영향을 받지 않는다.** 점수·경보·임계값의 prefix 일치와 원본 로컬 baseline 점수·지표 일치를 자동 검사한다.

## 주 표와 전체 대조군

주 표는 실행 전에 **E8B M3 conditional_only / E12 S1 100% head_only / E11B CP**로 고정했다. 3개 표×6개 행이며, 결과를 보고 baseline을 고르지 않는다. 사진의 6개 구성은 주 표뿐 아니라 **각 원본 내부 비교군과 readout 모두**에 적용한다. E8B의 위치 대조군에서도 stall은 공통 원래 tracker phase를 사용해 stall까지 바뀌는 혼선을 피한다.

`reports/TABLES.md`가 사진 형식의 표, `reports/all_metrics.csv`가 전체 결과, `reports/macro.json`이 seed별 Macro다. 완료되지 않은 scene 집합은 `1/4 scenes`처럼 표시하므로 전체 Macro로 오해하지 않는다. `results/ablation/.../upstream_baseline_deltas.json`은 공개 팀원 수치와 이번 재학습 baseline의 차이다. 차이가 0이라고 가정하지 않는다.

## 실행 및 재개

프로젝트 루트 PowerShell에서:

```powershell
python experiment/run.py --data-root /path/to/IPAD_dataset --output-root work
```

같은 명령으로 재개한다. 완료된 특징·학습·추가 구성을 건너뛴다. 코드·설정·protocol이 바뀌면 기존 결과에 섞지 않고 중단한다. `--stage prepare|extract|stats|addons|report`로 단계만 실행할 수 있다. 기본값이 3 seeds×5 folds×4 scenes다. 먼저 seed42/fold0의 4개 장면 주 결과가 나오고 이후 나머지를 진행한다.

진행은 `feature_progress.json`, `progress.json`, `execution.log`, 오류는 `execution.err.log`와 `failure.json`에서 확인한다. 전체 완료 증거는 `all_completion.json`이다. 이를 만들기 전에는 전체 완료라고 보고하지 않는다.

## 재현의 범위

공개 원본에는 팀원 특징 캐시와 학습 checkpoint가 없다. 따라서 동일 방법으로 **재학습한 baseline과 같은 실행에서 비교**한다. 로컬 영상 수·프레임 수·라벨 SHA256을 팀원 E0 기록과 비교하고 분할 겹침을 검사한다. 원본 이미지의 content hash는 공개되지 않아 픽셀 단위 동일성을 증명할 수 없다. GPU나 버전의 동일성은 사용자의 명시적인 범위 수정에 따라 요구하지 않는다.


## 공개본 경로 변경

`experiment/run.py`는 공개 저장소용 경로 어댑터다. 원본 학습 함수와 실험 수식을 바꾸지 않고 `--output-root`, 고정 source manifest 검사 및 최초 가중치 다운로드를 지원한다. 실제 실행했던 수정 전 파일은 `provenance/original_execution/`에 byte 그대로 남겼다. 완료 실험의 계획/코드 해시는 그 원본 실행 파일을 가리킨다. 공개용 어댑터를 사용한 전체 재학습은 다시 수행하지 않았으며 prepare/단위 테스트/산출물 검증으로 경로와 의미 보존을 확인한다.
