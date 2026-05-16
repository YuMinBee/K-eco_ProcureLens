#!/usr/bin/env python
"""Score the current supplier candidates with a previously trained HGB model.

Training is intentionally separated from inference. Use
`train_supplier_rank_model_histgradientboostingclassifier_train20to24.py` only
when the historical training window or model settings change.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
HGB_SCRIPT = SCRIPTS / ".pipeline_hgb" / "train_supplier_rank_model_histgradientboostingclassifier_train20to24.py"
MODEL_OUTPUTS = ROOT / "02_processed" / "model_outputs"
DEFAULT_JOBLIB = MODEL_OUTPUTS / "supplier_rank_histgradientboostingclassifier.joblib"
DEFAULT_PKL = MODEL_OUTPUTS / "supplier_rank_histgradientboostingclassifier.pkl"
SUBMISSION_MODEL_OUTPUTS = ROOT / "04_outputs" / "model_outputs"
SUBMISSION_JOBLIB = SUBMISSION_MODEL_OUTPUTS / "supplier_rank_histgradientboostingclassifier.joblib"
SUBMISSION_PKL = SUBMISSION_MODEL_OUTPUTS / "supplier_rank_histgradientboostingclassifier.pkl"


def load_hgb_module(script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"HGB helper script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("hgb_supplier_rank_helper", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import HGB helper script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(module: Any, model_path: Path | None) -> Any:
    path = model_path
    if path is None:
        for candidate in (DEFAULT_JOBLIB, SUBMISSION_JOBLIB, DEFAULT_PKL, SUBMISSION_PKL):
            if candidate.exists():
                path = candidate
                break
        else:
            path = DEFAULT_JOBLIB
    if not path.exists():
        raise FileNotFoundError(f"Trained HGB model not found: {path}")

    if path.suffix == ".joblib":
        if getattr(module, "joblib", None) is None:
            raise RuntimeError("joblib is required to load the HGB model.")
        artifact = module.joblib.load(path)
    else:
        with path.open("rb") as f:
            artifact = pickle.load(f)

    if isinstance(artifact, dict):
        features = artifact.get("features")
        if features and list(features) != list(module.FEATURE_COLUMNS):
            print("warning: saved model feature list differs from current helper script", file=sys.stderr)
        return artifact["model"]
    return artifact


def build_prior_award_stats(module: Any, index: Any, matcher: Any) -> tuple[Any, int]:
    prior = module.PriorAwardStats.create()
    events = module.load_award_events(index, matcher)
    for event in events:
        amount = module.parse_float(event.row.get("낙찰금액"))
        participant_count = module.parse_float(event.row.get("참가업체수"))
        prior.update(event.winner_bizno, event.item_code, event.date, amount, participant_count)
    return prior, len(events)


def main() -> int:
    parser = argparse.ArgumentParser(description="저장된 HGB 모델로 현재 후보업체만 예측합니다.")
    parser.add_argument("--model-path", type=Path, help="저장된 .joblib 또는 .pkl 모델 경로")
    parser.add_argument("--hgb-script", type=Path, default=HGB_SCRIPT, help="HGB 학습 스크립트/helper 경로")
    parser.add_argument("--skip-rule-comparison", action="store_true")
    args = parser.parse_args()

    module = load_hgb_module(args.hgb_script)
    model = load_model(module, args.model_path)

    print("공급물품 인덱스 생성 중...")
    index = module.build_supply_index()
    print(f"품목명 패턴 {len(index.name_to_code):,}개, 공급업체-품목 조합 {len(index.first_reg):,}개")
    matcher = module.AhoCorasickMatcher(list(index.name_to_code))
    binary_sets = module.load_binary_sets()

    print("과거 낙찰 통계 적재 중...")
    prior, event_count = build_prior_award_stats(module, index, matcher)
    print(f"낙찰 이벤트 {event_count:,}건 반영")

    print("현재 후보업체 HGB 예측 중...")
    official_rows = module.score_official_candidates(index, binary_sets, prior, model)
    module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    module.write_csv(
        module.OUTPUT_DIR / "HGB_후보업체_AI예측점수.csv",
        official_rows,
        ["물품별_AI순위", "전체_AI순위", "candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명", "AI예측점수"] + module.FEATURE_COLUMNS,
    )
    if not args.skip_rule_comparison:
        module.write_ai_rule_comparison(official_rows)

    print(f"완료: {len(official_rows):,}개 후보 예측")
    print(f"결과 폴더: {module.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
