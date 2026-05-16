# 공공기관 사회적 책임 구매 목표 달성 공급망 분석 모델 제출본

## 구성

- `00_assignment/`: 지정과제 요구사항 요약
- `01_report/`: 분석 결과보고서, 요구사항 충족 점검표, 실행 방법
- `02_dashboard/`: 발주 실무 의사결정 지원 대시보드 HTML
- `03_code/`: Python 분석 코드
- `04_outputs/`: 후보 목록, AI 예측 결과, 프리셋 점수, 성능 지표
- `05_data_summary/`: 활용 데이터 출처 및 처리 요약

## 핵심 산출물

- 분석 결과보고서: `01_report/분석_결과보고서.md`
- 요구사항 점검표: `01_report/요구사항_충족_점검표.md`
- 대시보드: `02_dashboard/procurement_decision_dashboard.html`
- 실행 파이프라인: `03_code/run_procurement_dashboard_pipeline.py`
- 저장 모델 예측: `03_code/score_hgb_supplier_candidates.py`

## 현재 제출본 기준

- 적용 품목: 유기응집제, 기타수질분석기, 여과장치, 무기응집제, 수처리용여과재, 정량펌프, 송풍기, 유량계, 수중펌프, 제수밸브
- 후보 업체: 품목별 50개, 총 500개
- AI 엔진: HistGradientBoosting 기반 후보업체 랭킹 모델
- AI 실행 방식: 학습 데이터가 갱신되면 HGB 재학습, 단순 후보 갱신은 저장 모델로 예측
- 프리셋 해석: 균형형, 안정성 중심, 사회적 가치 중심, 위험 회피 중심
- 갱신 방식: 신규 품목 누적 목록 기반 파이프라인 재생성, 예약 갱신 스크립트 제공
