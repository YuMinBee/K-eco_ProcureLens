#!/usr/bin/env python
"""Merge supplier feature files and score candidates with preset weights.

Typical flow:
  python 03_code/score_supplier_features.py --init
  python 03_code/score_supplier_features.py --candidates 04_outputs/후보업체_목록.csv --all-presets

Business-level features are merged by 사업자번호_정규화.
Item-supplier features are merged by candidate_id or 물품번호 + 사업자번호_정규화.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "02_processed"
DEFAULT_CANDIDATES = PROCESSED / "후보업체_목록.csv"
DEFAULT_FEATURE_DIR = PROCESSED / "feature_inputs"
DEFAULT_PRESETS = PROCESSED / "점수_프리셋.csv"
DEFAULT_OUTPUT_DIR = PROCESSED / "scored_outputs"
DEFAULT_TEMPLATE = DEFAULT_FEATURE_DIR / "팀원_feature_입력양식.csv"

KEY_ALIASES = [
    "사업자번호_정규화",
    "사업자번호",
    "bizno",
    "brno",
    "business_no",
    "업체사업자번호",
    "낙찰업체사업자번호",
]
CANDIDATE_ID_ALIASES = ["candidate_id", "후보ID", "후보아이디"]
ITEM_CODE_ALIASES = ["물품번호", "세부품명번호", "dtilPrdctClsfcNo"]
BASE_COLUMNS = {"candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명", "대표물품", "제조업체"}

FEATURE_ALIASES = {
    "공급물품_일치": ["공급물품_일치", "공급물품 매칭", "매칭물품명", "세부품명번호", "물품번호", "물품명"],
    "대표물품": ["대표물품", "대표 물품", "대표물품여부", "대표물품 여부", "rprsntPrdctClsfcNoNmYn"],
    "제조업체": ["제조업체", "제조 여부", "제조여부", "제조업체여부", "mnfctYn"],
    "낙찰건수": ["낙찰건수", "낙찰횟수", "과거 낙찰 횟수", "과거낙찰건수"],
    "최근낙찰건수": ["최근낙찰건수", "최근 낙찰 건수", "최근기간낙찰건수"],
    "낙찰금액합계": ["낙찰금액합계", "과거 낙찰 금액", "과거낙찰금액", "낙찰금액"],
    "창업기업": ["창업기업", "창업기업여부"],
    "사회적기업": ["사회적기업", "사회적기업여부"],
    "사회적협동조합": ["사회적협동조합", "사회적협동조합여부"],
    "장애인표준사업장": ["장애인표준사업장", "장애인 표준사업장", "장애인표준사업장여부"],
    "공공구매인증서": ["공공구매인증서", "공공구매 인증서", "공공구매인증서여부"],
    "자활용사촌복지공장": ["자활용사촌복지공장", "자활용사촌 복지공장", "자활용사촌복지공장여부"],
    "생산품목매칭": ["생산품목매칭", "생산품목 매칭", "품목매칭", "물품생산품목매칭"],
    "부정당제재": ["부정당제재", "부정당제재여부", "현재부정당제재"],
    "현재제재여부": ["현재제재여부", "현재 제재 여부", "현재부정당제재", "부정당제재중"],
}

PRESET_ROWS = [
    ("균형형", "대표물품", 5, "가점", "binary", "Y", "선택 물품이 대표물품인지"),
    ("균형형", "제조업체", 5, "가점", "binary", "Y", "제조 가능 등록 여부"),
    ("균형형", "낙찰건수", 20, "가점", "log_minmax", "Y", "과거 낙찰 건수"),
    ("균형형", "최근낙찰건수", 10, "가점", "log_minmax", "Y", "최근 기간 내 낙찰 건수"),
    ("균형형", "낙찰금액합계", 10, "가점", "log_minmax", "Y", "과거 낙찰 금액 합계"),
    ("균형형", "창업기업", 8, "가점", "binary", "Y", "창업기업 확인 여부"),
    ("균형형", "사회적기업", 10, "가점", "binary", "Y", "사회적기업 여부"),
    ("균형형", "사회적협동조합", 7, "가점", "binary", "Y", "사회적협동조합 여부"),
    ("균형형", "장애인표준사업장", 10, "가점", "binary", "Y", "장애인 표준사업장 여부"),
    ("균형형", "공공구매인증서", 5, "가점", "binary", "Y", "공공구매 인증서 보유 여부"),
    ("균형형", "자활용사촌복지공장", 5, "가점", "binary", "Y", "자활용사촌 복지공장 여부"),
    ("균형형", "생산품목매칭", 5, "가점", "binary", "Y", "candidate_id 기준 생산품목 매칭 여부"),
    ("균형형", "부정당제재", 100, "감점", "binary", "Y", "현재 부정당제재 여부"),
    ("균형형", "현재제재여부", 60, "감점", "binary", "Y", "현재 제재 중인 업체 여부"),
    ("안정성 중심", "대표물품", 5, "가점", "binary", "Y", "선택 물품이 대표물품인지"),
    ("안정성 중심", "제조업체", 10, "가점", "binary", "Y", "제조 가능 등록 여부"),
    ("안정성 중심", "낙찰건수", 30, "가점", "log_minmax", "Y", "과거 낙찰 건수"),
    ("안정성 중심", "최근낙찰건수", 20, "가점", "log_minmax", "Y", "최근 기간 내 낙찰 건수"),
    ("안정성 중심", "낙찰금액합계", 20, "가점", "log_minmax", "Y", "과거 낙찰 금액 합계"),
    ("안정성 중심", "창업기업", 3, "가점", "binary", "Y", "창업기업 확인 여부"),
    ("안정성 중심", "사회적기업", 5, "가점", "binary", "Y", "사회적기업 여부"),
    ("안정성 중심", "사회적협동조합", 3, "가점", "binary", "Y", "사회적협동조합 여부"),
    ("안정성 중심", "장애인표준사업장", 5, "가점", "binary", "Y", "장애인 표준사업장 여부"),
    ("안정성 중심", "공공구매인증서", 2, "가점", "binary", "Y", "공공구매 인증서 보유 여부"),
    ("안정성 중심", "자활용사촌복지공장", 2, "가점", "binary", "Y", "자활용사촌 복지공장 여부"),
    ("안정성 중심", "생산품목매칭", 2, "가점", "binary", "Y", "candidate_id 기준 생산품목 매칭 여부"),
    ("안정성 중심", "부정당제재", 120, "감점", "binary", "Y", "현재 부정당제재 여부"),
    ("안정성 중심", "현재제재여부", 80, "감점", "binary", "Y", "현재 제재 중인 업체 여부"),
    ("사회적 가치 중심", "대표물품", 2, "가점", "binary", "Y", "선택 물품이 대표물품인지"),
    ("사회적 가치 중심", "제조업체", 3, "가점", "binary", "Y", "제조 가능 등록 여부"),
    ("사회적 가치 중심", "낙찰건수", 5, "가점", "log_minmax", "Y", "과거 낙찰 건수"),
    ("사회적 가치 중심", "최근낙찰건수", 5, "가점", "log_minmax", "Y", "최근 기간 내 낙찰 건수"),
    ("사회적 가치 중심", "낙찰금액합계", 5, "가점", "log_minmax", "Y", "과거 낙찰 금액 합계"),
    ("사회적 가치 중심", "창업기업", 15, "가점", "binary", "Y", "창업기업 확인 여부"),
    ("사회적 가치 중심", "사회적기업", 20, "가점", "binary", "Y", "사회적기업 여부"),
    ("사회적 가치 중심", "사회적협동조합", 15, "가점", "binary", "Y", "사회적협동조합 여부"),
    ("사회적 가치 중심", "장애인표준사업장", 20, "가점", "binary", "Y", "장애인 표준사업장 여부"),
    ("사회적 가치 중심", "공공구매인증서", 5, "가점", "binary", "Y", "공공구매 인증서 보유 여부"),
    ("사회적 가치 중심", "자활용사촌복지공장", 5, "가점", "binary", "Y", "자활용사촌 복지공장 여부"),
    ("사회적 가치 중심", "생산품목매칭", 10, "가점", "binary", "Y", "candidate_id 기준 생산품목 매칭 여부"),
    ("사회적 가치 중심", "부정당제재", 100, "감점", "binary", "Y", "현재 부정당제재 여부"),
    ("사회적 가치 중심", "현재제재여부", 50, "감점", "binary", "Y", "현재 제재 중인 업체 여부"),
    ("위험 회피 중심", "대표물품", 3, "가점", "binary", "Y", "선택 물품이 대표물품인지"),
    ("위험 회피 중심", "제조업체", 7, "가점", "binary", "Y", "제조 가능 등록 여부"),
    ("위험 회피 중심", "낙찰건수", 25, "가점", "log_minmax", "Y", "과거 낙찰 건수"),
    ("위험 회피 중심", "최근낙찰건수", 15, "가점", "log_minmax", "Y", "최근 기간 내 낙찰 건수"),
    ("위험 회피 중심", "낙찰금액합계", 15, "가점", "log_minmax", "Y", "과거 낙찰 금액 합계"),
    ("위험 회피 중심", "창업기업", 5, "가점", "binary", "Y", "창업기업 확인 여부"),
    ("위험 회피 중심", "사회적기업", 8, "가점", "binary", "Y", "사회적기업 여부"),
    ("위험 회피 중심", "사회적협동조합", 5, "가점", "binary", "Y", "사회적협동조합 여부"),
    ("위험 회피 중심", "장애인표준사업장", 8, "가점", "binary", "Y", "장애인 표준사업장 여부"),
    ("위험 회피 중심", "공공구매인증서", 4, "가점", "binary", "Y", "공공구매 인증서 보유 여부"),
    ("위험 회피 중심", "자활용사촌복지공장", 5, "가점", "binary", "Y", "자활용사촌 복지공장 여부"),
    ("위험 회피 중심", "생산품목매칭", 5, "가점", "binary", "Y", "candidate_id 기준 생산품목 매칭 여부"),
    ("위험 회피 중심", "부정당제재", 200, "감점", "binary", "Y", "현재 부정당제재 여부"),
    ("위험 회피 중심", "현재제재여부", 150, "감점", "binary", "Y", "현재 제재 중인 업체 여부"),
]

TEMPLATE_ROWS = [
    {
        "candidate_id": "1",
        "물품번호": "12345678",
        "물품명": "예시품목",
        "사업자번호": "123-45-67890",
        "사업자번호_정규화": "1234567890",
        "업체명": "예시기업",
        "대표물품": "Y",
        "제조업체": "N",
        "낙찰건수": "",
        "최근낙찰건수": "",
        "최근낙찰일": "",
        "평균낙찰금액": "",
        "낙찰금액합계": "",
        "창업기업": "Y",
        "사회적기업": "N",
        "사회적협동조합": "N",
        "장애인표준사업장": "N",
        "공공구매인증서": "Y",
        "인증개수": "",
        "인증유효여부": "",
        "자활용사촌복지공장": "N",
        "생산품목매칭": "",
        "부정당제재": "N",
        "제재횟수": "",
        "현재제재여부": "",
        "제재종료일": "",
        "매칭방식": "사업자번호매칭",
        "비고": "팀원별 담당 feature만 채우고 나머지는 비워도 됩니다.",
    }
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for column in row:
                if column not in columns:
                    columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def norm_bizno(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def first_existing(row: dict[str, Any], candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in row:
            return column
    return None


def find_key_column(row: dict[str, Any]) -> str | None:
    return first_existing(row, KEY_ALIASES)


def normalized_bizno_from_row(row: dict[str, Any]) -> str:
    key_col = find_key_column(row)
    return norm_bizno(row.get(key_col) if key_col else "")


def candidate_id_from_row(row: dict[str, Any]) -> str:
    column = first_existing(row, CANDIDATE_ID_ALIASES)
    return str(row.get(column) or "").strip() if column else ""


def item_code_from_row(row: dict[str, Any]) -> str:
    column = first_existing(row, ITEM_CODE_ALIASES)
    return str(row.get(column) or "").strip() if column else ""


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if parse_bool(text) is not None:
        return float(parse_bool(text) or 0)
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_bool(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    yes_values = {"y", "yes", "true", "1", "o", "예", "있음", "해당", "해당됨", "인증", "보유"}
    no_values = {"n", "no", "false", "0", "x", "아니오", "없음", "미해당", "미보유"}
    if text in yes_values:
        return 1
    if text in no_values:
        return 0
    return None


def coerce_feature_value(row: dict[str, Any], feature: str) -> float:
    aliases = FEATURE_ALIASES.get(feature, [feature])
    column = first_existing(row, aliases)
    if not column:
        return 0.0
    value = row.get(column)
    bool_value = parse_bool(value)
    if bool_value is not None:
        return float(bool_value)
    # Text-only evidence columns such as 매칭물품명 mean "matched" when non-empty.
    if feature in {"공급물품_일치"} and str(value or "").strip():
        return 1.0
    return parse_number(value)


def normalize(values: dict[str, float], method: str) -> dict[str, float]:
    if method == "binary":
        return {key: 1.0 if value > 0 else 0.0 for key, value in values.items()}
    if method == "none":
        return values
    working = values
    if method == "log_minmax":
        working = {key: math.log1p(max(0.0, value)) for key, value in values.items()}
    min_value = min(working.values(), default=0.0)
    max_value = max(working.values(), default=0.0)
    if max_value <= min_value:
        return {key: 1.0 if working.get(key, 0.0) > 0 else 0.0 for key in values}
    return {key: (working[key] - min_value) / (max_value - min_value) for key in values}


def load_presets(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_csv_rows(path)
    presets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (row.get("사용여부") or "Y").strip().upper() != "Y":
            continue
        preset = (row.get("프리셋") or "").strip()
        feature = (row.get("피처") or "").strip()
        if preset and feature:
            presets[preset].append(row)
    return dict(presets)


def load_candidates(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_csv_rows(path)
    merged: list[dict[str, Any]] = []
    columns: list[str] = []
    for index, row in enumerate(rows, 1):
        bizno = normalized_bizno_from_row(row)
        if not bizno:
            continue
        normalized = dict(row)
        if not normalized.get("candidate_id"):
            normalized["candidate_id"] = str(index)
        normalized["사업자번호_정규화"] = bizno
        if not normalized.get("사업자번호"):
            normalized["사업자번호"] = bizno
        merged.append(normalized)
        for column in normalized:
            if column not in columns:
                columns.append(column)
    return merged, columns


def merge_features(candidates: list[dict[str, Any]], feature_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    # Match feature rows from strongest to weakest key:
    # candidate_id -> item+business number -> business number.
    by_candidate_id = {candidate_id_from_row(row): row for row in candidates if candidate_id_from_row(row)}
    by_item_bizno: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_bizno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        bizno = normalized_bizno_from_row(row)
        item_code = item_code_from_row(row)
        if item_code and bizno:
            by_item_bizno[(item_code, bizno)].append(row)
        if bizno:
            by_bizno[bizno].append(row)

    columns: list[str] = []
    for row in candidates:
        for column in row:
            if column not in columns:
                columns.append(column)

    def apply_feature_row(target: dict[str, Any], feature_row: dict[str, Any]) -> None:
        for column, value in feature_row.items():
            if column in BASE_COLUMNS:
                continue
            if column not in columns:
                columns.append(column)
            if value in (None, ""):
                continue
            if not target.get(column):
                target[column] = value

    for path in feature_paths:
        for feature_row in read_csv_rows(path):
            candidate_id = candidate_id_from_row(feature_row)
            bizno = normalized_bizno_from_row(feature_row)
            item_code = item_code_from_row(feature_row)

            if candidate_id and candidate_id in by_candidate_id:
                targets = [by_candidate_id[candidate_id]]
            elif item_code and bizno and (item_code, bizno) in by_item_bizno:
                targets = by_item_bizno[(item_code, bizno)]
            elif bizno and bizno in by_bizno:
                targets = by_bizno[bizno]
            else:
                continue

            for target in targets:
                apply_feature_row(target, feature_row)
    return candidates, columns


def score_rows(rows: list[dict[str, Any]], preset_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    def score_key(index: int, row: dict[str, Any]) -> str:
        candidate_id = candidate_id_from_row(row)
        if candidate_id:
            return f"candidate:{candidate_id}"
        return f"row:{index}:{normalized_bizno_from_row(row)}"

    raw_by_feature: dict[str, dict[str, float]] = {}
    for preset_row in preset_rows:
        feature = preset_row["피처"]
        raw_by_feature[feature] = {
            score_key(index, row): coerce_feature_value(row, feature)
            for index, row in enumerate(rows)
        }

    normalized_by_feature: dict[str, dict[str, float]] = {}
    for preset_row in preset_rows:
        feature = preset_row["피처"]
        method = (preset_row.get("정규화") or "binary").strip()
        # Normalize per current candidate pool so each preset compares suppliers within the selected item set.
        normalized_by_feature[feature] = normalize(raw_by_feature[feature], method)

    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = score_key(index, row)
        total = 0.0
        detail_parts: list[str] = []
        feature_values: dict[str, Any] = {}
        for preset_row in preset_rows:
            feature = preset_row["피처"]
            weight = parse_number(preset_row.get("가중치"))
            direction = (preset_row.get("방향") or "가점").strip()
            raw = raw_by_feature[feature].get(key, 0.0)
            normalized = normalized_by_feature[feature].get(key, 0.0)
            contribution = normalized * weight
            if direction == "감점":
                contribution *= -1
            total += contribution
            feature_values[feature] = raw
            # Keep contribution details for dashboard tooltips and auditability.
            detail_parts.append(f"{feature}={contribution:.2f}")

        output = dict(row)
        output.update(feature_values)
        output["프리셋점수"] = round(total, 4)
        output["프리셋점수상세"] = "; ".join(detail_parts)
        scored.append(output)

    scored.sort(key=lambda row: (-float(row["프리셋점수"]), str(row.get("업체명", "")), str(row.get("사업자번호", ""))))
    for index, row in enumerate(scored, 1):
        row["순위"] = index
    return scored


def init_files() -> None:
    preset_columns = ["프리셋", "피처", "가중치", "방향", "정규화", "사용여부", "설명"]
    if not DEFAULT_PRESETS.exists():
        rows = [
            {
                "프리셋": preset,
                "피처": feature,
                "가중치": weight,
                "방향": direction,
                "정규화": normalization,
                "사용여부": enabled,
                "설명": description,
            }
            for preset, feature, weight, direction, normalization, enabled, description in PRESET_ROWS
        ]
        write_csv(DEFAULT_PRESETS, rows, preset_columns)
    if not DEFAULT_TEMPLATE.exists():
        write_csv(DEFAULT_TEMPLATE, TEMPLATE_ROWS)


def expand_feature_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return [
        path
        for path in paths
        if path.exists() and path.name != DEFAULT_TEMPLATE.name and "입력양식" not in path.name and "요약" not in path.name
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="후보 업체 feature 병합 및 프리셋 점수 계산")
    parser.add_argument("--init", action="store_true", help="점수 프리셋과 팀원 feature 입력양식 생성")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="후보 업체 CSV")
    parser.add_argument("--features", nargs="*", default=[str(DEFAULT_FEATURE_DIR / "*.csv")], help="팀원 feature CSV 경로 또는 glob")
    parser.add_argument("--presets-csv", type=Path, default=DEFAULT_PRESETS, help="프리셋 가중치 CSV")
    parser.add_argument("--preset", help="계산할 프리셋명")
    parser.add_argument("--all-presets", action="store_true", help="모든 프리셋을 각각 계산")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="결과 저장 폴더")
    parser.add_argument("--output", type=Path, help="단일 프리셋 결과 저장 경로")
    args = parser.parse_args()

    if args.init:
        init_files()
        print(f"프리셋 파일: {DEFAULT_PRESETS}")
        print(f"팀원 입력양식: {DEFAULT_TEMPLATE}")
        return 0

    init_files()
    if not args.candidates.exists():
        raise SystemExit(f"후보 업체 CSV가 없습니다: {args.candidates}")

    candidates, base_columns = load_candidates(args.candidates)
    feature_paths = expand_feature_paths(args.features)
    merged, merged_columns = merge_features(candidates, feature_paths)
    presets = load_presets(args.presets_csv)
    preset_names = list(presets) if args.all_presets else [args.preset or "균형형"]

    for preset_name in preset_names:
        if preset_name not in presets:
            raise SystemExit(f"프리셋을 찾지 못했습니다: {preset_name}")
        scored = score_rows(merged, presets[preset_name])
        output = (
            args.output
            if args.output and len(preset_names) == 1
            else args.output_dir / args.candidates.stem / f"업체추천_점수_{preset_name}.csv"
        )
        score_columns = ["순위", "프리셋점수"]
        feature_columns = [row["피처"] for row in presets[preset_name] if row["피처"] not in merged_columns]
        columns = score_columns + [column for column in merged_columns if column not in score_columns] + feature_columns + ["프리셋점수상세"]
        write_csv(output, scored, columns)
        print(f"{preset_name}: {output} ({len(scored)}개 업체)")

    print(f"후보 파일: {args.candidates}")
    print(f"feature 파일 수: {len(feature_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
