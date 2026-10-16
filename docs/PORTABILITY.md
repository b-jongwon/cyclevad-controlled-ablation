# 공개 실행 어댑터

- 실제 실행 코드 원본은 `provenance/original_execution/`에 보존한다.
- `experiment/run.py`: 특정 PC 경로 대신 저장소 root와 `--output-root` 사용, vendored source manifest 검증, 최초 모델 다운로드 허용. CUDA minor version을 같게 요구하지 않는다.
- `experiment/addons.py`: 팀원 비교 수치 경로를 `provenance/upstream_metrics`로 변경한다. 점수·학습·보정·경보·평가 수식은 원본 실행 코드와 동일하다.
- 원본 `src/cycle_vad/`는 byte 그대로다. 새 저장소 commit을 팀원 commit으로 오인하지 않도록 source hash를 별도로 검증한다.
- 공개용 어댑터에서 79분짜리 전체 실험을 다시 돌리지는 않았다. 원본 완료 결과는 전체 검증했고 공개 어댑터는 테스트 및 실제 데이터 prepare를 검증했다.
