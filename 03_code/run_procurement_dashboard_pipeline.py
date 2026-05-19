#!/usr/bin/env python
"""Run the procurement dashboard data pipeline end to end.

Examples:
  python 03_code/run_procurement_dashboard_pipeline.py
  python 03_code/run_procurement_dashboard_pipeline.py --item 4710160801:유기응집제
  python 03_code/run_procurement_dashboard_pipeline.py --items-csv input_items.csv --top-per-item 50
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "03_code"
PROCESSED = ROOT / "02_processed"
STATUS_PATH = PROCESSED / "pipeline_status.json"
# New item codes are accumulated here so later runs keep previously added dashboard items.
ITEM_REGISTRY = PROCESSED / "대시보드_물품목록.csv"
SUBMISSION_ITEM_REGISTRY = ROOT / "04_outputs" / "대시보드_물품목록.csv"
HGB_ZIP = ROOT / "05_15_추가" / "0513 adjusted.zip"
HGB_SCRIPT_MEMBER = "scripts/histgradientboosting/train_supplier_rank_model_histgradientboostingclassifier_train20to24.py"
GENERATED_HGB_SCRIPT = SCRIPTS / ".pipeline_hgb" / "train_supplier_rank_model_histgradientboostingclassifier_train20to24.py"
HGB_MODEL_PATH = PROCESSED / "model_outputs" / "supplier_rank_histgradientboostingclassifier.joblib"
SUBMISSION_HGB_MODEL_PATH = ROOT / "04_outputs" / "model_outputs" / "supplier_rank_histgradientboostingclassifier.joblib"
STANDARD = PROCESSED / "한국어_표준데이터셋"
HGB_TRAINING_SOURCE_PATHS = [
    STANDARD / "조달청_나라장터_사용자정보서비스_조달업체공급물품정보조회.csv",
    STANDARD / "조달청_나라장터_낙찰정보서비스_물품.csv",
    STANDARD / "조달청_나라장터_사용자정보서비스_부정당제재업체정보조회.csv",
    STANDARD / "창업진흥원_창업기업확인서발급기업정보_조회서비스.csv",
    STANDARD / "한국장애인고용공단_장애인_표준사업장_실시간_조회.csv",
    STANDARD / "사회적기업_조회.csv",
]
DEFAULT_DASHBOARD_ITEMS = [
    ("4710160801", "유기응집제"),
    ("4111331901", "기타수질분석기"),
    ("4016150601", "여과장치"),
    ("4710160802", "무기응집제"),
    ("4710169801", "수처리용여과재"),
    ("4015150501", "정량펌프"),
    ("4010160101", "송풍기"),
    ("4111250101", "유량계"),
    ("4015151301", "수중펌프"),
    ("4014169401", "제수밸브"),
]


def norm_digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def parse_item_spec(value: str) -> tuple[str, str]:
    text = (value or "").strip()
    if not text:
        return "", ""
    for separator in (":", ",", "|"):
        if separator in text:
            code, name = text.split(separator, 1)
            break
    else:
        code, name = text, ""
    return norm_digits(code), name.strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    return []


def load_items_csv(path: Path | None) -> list[tuple[str, str]]:
    if not path:
        return []
    items: list[tuple[str, str]] = []
    for row in read_csv_rows(path):
        code = row.get("물품번호") or row.get("세부품명번호") or row.get("item_code") or row.get("code") or ""
        name = row.get("물품명") or row.get("세부품명") or row.get("item_name") or row.get("name") or ""
        item_code = norm_digits(code)
        if item_code:
            items.append((item_code, str(name or "").strip()))
    return items


def load_items_from_candidates(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for row in read_csv_rows(path):
        code = norm_digits(row.get("물품번호"))
        name = str(row.get("물품명") or "").strip()
        if code:
            items.append((code, name))
    return items


def merge_items(*groups: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    index_by_code: dict[str, int] = {}
    for group in groups:
        for code, name in group:
            if not code:
                continue
            if code not in index_by_code:
                index_by_code[code] = len(merged)
                merged.append((code, name))
            elif name:
                merged[index_by_code[code]] = (code, name)
    return merged


def write_items_csv(path: Path, items: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["물품번호", "물품명"])
        writer.writeheader()
        for code, name in items:
            writer.writerow({"물품번호": code, "물품명": name})


def configure_item_registry(args: argparse.Namespace) -> None:
    args.effective_items = list(args.item or [])
    args.effective_items_csv = args.items_csv
    args.dashboard_items = []

    if not args.remember_items:
        return

    # Keep the dashboard item list persistent: a new --item is appended, not used as a one-off replacement.
    input_items = merge_items(
        [parse_item_spec(item) for item in args.item or []],
        load_items_csv(args.items_csv),
    )
    if args.replace_items:
        base_items: list[tuple[str, str]] = []
    elif ITEM_REGISTRY.exists():
        base_items = load_items_csv(ITEM_REGISTRY)
    elif SUBMISSION_ITEM_REGISTRY.exists():
        base_items = load_items_csv(SUBMISSION_ITEM_REGISTRY)
    else:
        base_items = merge_items(DEFAULT_DASHBOARD_ITEMS, load_items_from_candidates(PROCESSED / "후보업체_목록.csv"))

    items = merge_items(base_items, input_items)
    if not items:
        items = list(DEFAULT_DASHBOARD_ITEMS)

    if not args.dry_run:
        write_items_csv(ITEM_REGISTRY, items)
        print(f"대시보드 물품 목록: {ITEM_REGISTRY} ({len(items)}개)")

    args.effective_items = []
    args.effective_items_csv = ITEM_REGISTRY
    args.dashboard_items = [f"{code}:{name}" if name else code for code, name in items]


def patch_hgb_script(content: bytes) -> bytes:
    """Make the archived HGB script tolerant of optional legacy CSV files."""
    text = content.decode("utf-8")
    old = '''def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")
'''
    new = '''def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")
'''
    if old not in text:
        raise RuntimeError("HGB 스크립트의 read_csv_rows 패치 위치를 찾지 못했습니다.")
    return text.replace(old, new, 1).encode("utf-8")


def run_step(name: str, command: list[str], dry_run: bool = False) -> None:
    label = " ".join(command)
    print(f"\n[{name}]\n{label}")
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_hgb_script(zip_path: Path, member: str = HGB_SCRIPT_MEMBER) -> Path:
    if not zip_path.exists():
        if GENERATED_HGB_SCRIPT.exists():
            # Submission packages can include the already patched HGB script
            # without the original archive.
            return GENERATED_HGB_SCRIPT
        raise FileNotFoundError(f"HGB zip 파일을 찾지 못했습니다: {zip_path}")
    GENERATED_HGB_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise FileNotFoundError(f"HGB 학습 스크립트를 zip에서 찾지 못했습니다: {member}")
        content = archive.read(member)
    # The archived HGB script references a few legacy CSV names; patch it at extraction time.
    GENERATED_HGB_SCRIPT.write_bytes(patch_hgb_script(content))
    return GENERATED_HGB_SCRIPT


def hgb_training_sources_newer_than_model(model_path: Path) -> bool:
    if not model_path.exists():
        return True
    model_mtime = model_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > model_mtime for path in HGB_TRAINING_SOURCE_PATHS)


def candidate_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPTS / "select_candidate_suppliers.py"),
        "--top-per-item",
        str(args.top_per_item),
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(args.end_year),
        "--output",
        str(PROCESSED / "후보업체_목록.csv"),
        "--summary",
        str(ROOT / "05_data_summary" / "품목_후보업체_선정요약.md"),
    ]
    for item in getattr(args, "effective_items", args.item or []):
        command.extend(["--item", item])
    items_csv = getattr(args, "effective_items_csv", args.items_csv)
    if items_csv:
        command.extend(["--items-csv", str(items_csv)])
    return command


def feature_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(SCRIPTS / "build_official_feature_inputs.py")]
    if args.use_api:
        command.append("--use-api")
    if args.use_smpp:
        command.append("--use-smpp")
    if args.today:
        command.extend(["--today", args.today])
    return command


def preset_command() -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "score_supplier_features.py"),
        "--candidates",
        str(PROCESSED / "후보업체_목록.csv"),
        "--all-presets",
        "--output-dir",
        str(PROCESSED / "scored_outputs"),
    ]


def ai_command(args: argparse.Namespace, dry_run: bool = False) -> list[str] | None:
    if args.ai_engine == "none":
        return None
    if args.ai_engine == "rf":
        return [
            sys.executable,
            str(SCRIPTS / "train_supplier_rank_model.py"),
            "--engine",
            args.rf_engine,
            "--train-negatives",
            str(args.train_negatives),
            "--test-negatives",
            str(args.test_negatives),
            "--trees",
            str(args.rf_trees),
            "--max-depth",
            str(args.max_depth),
            "--min-samples-leaf",
            str(args.min_samples_leaf),
            "--seed",
            str(args.seed),
        ]

    hgb_model_path = HGB_MODEL_PATH if HGB_MODEL_PATH.exists() else SUBMISSION_HGB_MODEL_PATH
    should_score_only = dry_run or (
        args.ai_mode in {"auto", "score"}
        and hgb_model_path.exists()
        and (args.ai_mode == "score" or not hgb_training_sources_newer_than_model(hgb_model_path))
    )
    if should_score_only:
        if not dry_run and not GENERATED_HGB_SCRIPT.exists():
            ensure_hgb_script(args.hgb_zip)
        return [
            sys.executable,
            str(SCRIPTS / "score_hgb_supplier_candidates.py"),
            "--model-path",
            str(hgb_model_path),
        ]
    if args.ai_mode == "score":
        raise FileNotFoundError(f"저장된 HGB 모델을 찾지 못했습니다: {HGB_MODEL_PATH} 또는 {SUBMISSION_HGB_MODEL_PATH}")

    hgb_script = GENERATED_HGB_SCRIPT if dry_run else ensure_hgb_script(args.hgb_zip)
    return [
        sys.executable,
        str(hgb_script),
        "--train-negatives",
        str(args.train_negatives),
        "--test-negatives",
        str(args.test_negatives),
        "--max-iter",
        str(args.hgb_max_iter),
        "--learning-rate",
        str(args.hgb_learning_rate),
        "--max-depth",
        str(args.max_depth),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--seed",
        str(args.seed),
    ]


def dashboard_command() -> list[str]:
    return [sys.executable, str(SCRIPTS / "build_procurement_decision_dashboard.py")]


def now_text() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_status(payload: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def status_payload(args: argparse.Namespace, status: str, started_at: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "started_at": started_at,
        "finished_at": extra.pop("finished_at", ""),
        "trigger": args.trigger,
        "schedule_label": args.schedule_label,
        "items": args.item or [],
        "items_csv": str(args.items_csv) if args.items_csv else "",
        "dashboard_items": getattr(args, "dashboard_items", []),
        "item_registry": rel_path(ITEM_REGISTRY) if args.remember_items else "",
        "top_per_item": args.top_per_item,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "use_api": args.use_api,
        "use_smpp": args.use_smpp,
        "ai_engine": args.ai_engine,
        "ai_mode": args.ai_mode,
        "dry_run": args.dry_run,
        "dashboard_html": "02_dashboard/procurement_decision_dashboard.html",
    }
    payload.update(extra)
    return payload


def run_pipeline(args: argparse.Namespace) -> None:
    started_at = now_text()
    configure_item_registry(args)
    write_status(status_payload(args, "running", started_at))
    try:
        # Each step writes a stable CSV/HTML artifact consumed by the next step.
        if not args.skip_candidates:
            run_step("1. 후보업체 목록 생성", candidate_command(args), args.dry_run)
        if not args.skip_features:
            run_step("2. feature 생성", feature_command(args), args.dry_run)
        if not args.skip_presets:
            run_step("3. 프리셋 점수 생성", preset_command(), args.dry_run)
        if not args.skip_ai:
            command = ai_command(args, args.dry_run)
            if command:
                run_step(f"4. AI 예측 생성 ({args.ai_engine})", command, args.dry_run)
        if not args.skip_dashboard:
            dashboard_started_at = now_text()
            started_dt = dt.datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
            dashboard_started_dt = dt.datetime.strptime(dashboard_started_at, "%Y-%m-%d %H:%M:%S")
            write_status(
                status_payload(
                    args,
                    "dry_run" if args.dry_run else "success",
                    started_at,
                    finished_at=dashboard_started_at,
                    duration_seconds=round((dashboard_started_dt - started_dt).total_seconds(), 3),
                    dashboard_rebuild="running",
                )
            )
            run_step("5. 대시보드 HTML 재생성", dashboard_command(), args.dry_run)
    except Exception as exc:
        finished_at = now_text()
        write_status(
            status_payload(
                args,
                "failed",
                started_at,
                finished_at=finished_at,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        raise

    finished_at = now_text()
    started_dt = dt.datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
    finished_dt = dt.datetime.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    write_status(
        status_payload(
            args,
            "dry_run" if args.dry_run else "success",
            started_at,
            finished_at=finished_at,
            duration_seconds=round((finished_dt - started_dt).total_seconds(), 3),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="후보 생성부터 대시보드 재생성까지 실행합니다.")
    parser.add_argument("--item", action="append", help="세부품명번호 또는 '세부품명번호:물품명'. 여러 번 입력 가능")
    parser.add_argument("--items-csv", type=Path, help="물품번호/물품명 컬럼이 있는 CSV")
    parser.add_argument("--no-remember-items", dest="remember_items", action="store_false", help="대시보드 물품 목록에 누적하지 않고 이번 입력만 사용합니다.")
    parser.add_argument("--replace-items", action="store_true", help="대시보드 물품 목록을 이번 입력으로 교체합니다.")
    parser.add_argument("--top-per-item", type=int, default=50)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--use-api", action="store_true", help="정책/인증 공공 API를 조회합니다.")
    parser.add_argument("--use-smpp", action="store_true", help="SMPP 생산품 API도 조회합니다.")
    parser.add_argument("--today", help="정책 API 기준일, YYYYMMDD")
    parser.add_argument("--ai-engine", choices=["hgb", "rf", "none"], default="hgb")
    parser.add_argument("--ai-mode", choices=["auto", "score", "train"], default="auto", help="hgb일 때 auto는 저장 모델이 있으면 예측만, 없으면 학습을 실행합니다.")
    parser.add_argument("--hgb-zip", type=Path, default=HGB_ZIP)
    parser.add_argument("--rf-engine", choices=["auto", "sklearn", "simple"], default="auto")
    parser.add_argument("--train-negatives", type=int, default=10)
    parser.add_argument("--test-negatives", type=int, default=30)
    parser.add_argument("--rf-trees", type=int, default=40)
    parser.add_argument("--hgb-max-iter", type=int, default=200)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-presets", action="store_true")
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--trigger", choices=["manual", "scheduled"], default="manual")
    parser.add_argument("--schedule-label", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(remember_items=True)
    args = parser.parse_args()

    run_pipeline(args)
    print("\n파이프라인 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
