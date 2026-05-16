# GitHub 업로드 가이드

## 업로드 대상

GitHub에는 이 `06_submission` 폴더만 올리는 것을 권장한다.

전체 프로젝트 루트에는 원천 API 데이터와 중간 산출물이 포함되어 있어 용량이 크고, 제출 심사자가 확인해야 하는 핵심 산출물도 `06_submission`에 정리되어 있다.

## 포함된 핵심 산출물

- `README.md`: 제출본 구성 요약
- `01_report/분석_결과보고서.md`: 분석 결과보고서
- `01_report/요구사항_충족_점검표.md`: 지정과제 요구사항 대응표
- `02_dashboard/procurement_decision_dashboard.html`: 실행 없이 열 수 있는 대시보드
- `03_code/`: 후보 생성, feature 생성, AI 예측, 대시보드 생성 코드
- `04_outputs/`: 후보 목록, AI 예측 결과, 프리셋 점수, 성능 지표, 저장 모델
- `05_data_summary/`: 데이터 출처와 처리 요약

## 권장 Git 명령

```bash
cd /data4tb/kec/kec/06_submission
git init
git add .
git commit -m "Prepare procurement supplier dashboard submission"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## 주의사항

- 공공데이터 API 키는 GitHub에 올리지 않는다.
- `01_raw_data/`, `02_processed/` 같은 대용량 원천/실행 산출물은 제출 repo에 포함하지 않는다.
- 제출본 대시보드는 정적 HTML이므로 `02_dashboard/procurement_decision_dashboard.html`을 브라우저에서 바로 열어 확인할 수 있다.
- 신규 데이터가 들어오면 `run_procurement_dashboard_pipeline.py`가 자동 재학습 또는 저장 모델 예측을 선택한다.
