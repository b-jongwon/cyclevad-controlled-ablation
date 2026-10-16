# 출처와 변경 범위

기본 특징 추출, CycleTracker/PhysicalTracker, CycleVAD, E8/E8B/E9S/E11/E12/E11B 및 기존 tests는 [PigeonLabs/KNU_Capstone1_CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD/tree/7aa58ce893181599c4b5bd6f779fa2461ce3f103)의 해당 commit에서 가져왔다. `src/cycle_vad/`와 `configs/`의 원본 파일 해시는 `provenance/upstream_source_manifest.json`에 보존했다.

적응형 임계값, phase 조건부 motion lower-tail, 정지 연속 경보와 `max_mix_0.1`의 선행 확장은 [b-jongwon/cyclevad-adaptive-stall-ad](https://github.com/b-jongwon/cyclevad-adaptive-stall-ad)에서 수행했다. 이번 `experiment/addons.py`는 그 규칙을 고정하고 기존 motion 시작부의 미래 관측 복사를 제거해 원본 방법과 비교한다. lag가 없는 표본을 calibration에서 제외하고 첫4관측의 stall score를 0으로 둔다.

이번 저장소의 새 산출물은 2026-10-16 실제 재학습/평가한 수치, 6개 구성 비교, Baseline/prefix 검증, 이식 가능한 실행 어댑터 및 설명 문서다. 과거 16장면 수치를 이번 4장면 실험 결과로 사용하지 않았다. 기본 CycleVAD 전체를 새 알고리즘으로 주장하지 않는다.

원본 코드의 출처와 권리 표시는 유지하며 전체 코드를 새로운 MIT 등의 라이선스로 재라이선스하지 않는다. 원본 IPAD 데이터 및 DINOv2 가중치는 저장소에 포함하지 않는다.
