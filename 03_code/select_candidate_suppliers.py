#!/usr/bin/env python
"""Select item-level supplier candidates from G2B supply-product data.

Default output:
  04_outputs/후보업체_목록.csv

The file keeps stable columns shared with teammates:
candidate_id, 물품번호, 물품명, 사업자번호, 사업자번호_정규화, 업체명, 대표물품, 제조업체
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_raw_data"
PROCESSED = ROOT / "02_processed"

SUPPLY_DIR = RAW / "나라장터_사용자정보서비스" / "4. 조달업체공급물품정보조회" / "inqryDiv_2_등록일기준"
BASIC_DIR = RAW / "나라장터_사용자정보서비스" / "2. 조달업체기본정보조회"
AWARDS_DIR = RAW / "나라장터_낙찰정보서비스"

DEFAULT_OUTPUT = PROCESSED / "후보업체_목록.csv"
DEFAULT_SUMMARY = ROOT / "05_data_summary" / "품목_후보업체_선정요약.md"
DEFAULT_START_YEAR = 2022
DEFAULT_END_YEAR = 2025

DEFAULT_ITEMS = [
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
        raise argparse.ArgumentTypeError("품목 입력값이 비어 있습니다.")
    for sep in (":", ",", "|"):
        if sep in text:
            code, name = text.split(sep, 1)
            break
    else:
        code, name = text, ""
    item_code = norm_digits(code)
    item_name = name.strip()
    if not item_code:
        raise argparse.ArgumentTypeError(f"세부품명번호를 확인할 수 없습니다: {value}")
    return item_code, item_name


def is_yes(value: str | None) -> bool:
    return (value or "").strip().upper() in {"Y", "YES", "TRUE", "1", "O"}


def parse_amount(value: str | None) -> float:
    try:
        return float(str(value or "").replace(",", ""))
    except ValueError:
        return 0.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    return []


def load_items_csv(path: Path) -> list[tuple[str, str]]:
    rows = read_csv_rows(path)
    items: list[tuple[str, str]] = []
    for row in rows:
        code = row.get("물품번호") or row.get("세부품명번호") or row.get("item_code") or row.get("code") or ""
        name = row.get("물품명") or row.get("세부품명") or row.get("item_name") or row.get("name") or ""
        item_code = norm_digits(code)
        if item_code:
            items.append((item_code, str(name or "").strip()))
    return dedupe_items(items)


def dedupe_items(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen = set()
    for item_code, item_name in items:
        key = (item_code, item_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((item_code, item_name))
    return deduped


def path_year(path: Path) -> int | None:
    for part in path.parts:
        if re.fullmatch(r"20\d{2}", part):
            return int(part)
    return None


def in_year_range(path: Path, start_year: int, end_year: int) -> bool:
    year = path_year(path)
    return year is None or start_year <= year <= end_year


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명", "대표물품", "제조업체"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_supply_by_item(start_year: int, end_year: int) -> dict[tuple[str, str], dict[str, dict[str, bool]]]:
    # Supply-product registration is the mandatory eligibility filter for candidate suppliers.
    by_item: dict[tuple[str, str], dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(lambda: {"대표물품": False, "제조업체": False})
    )
    for path in sorted(SUPPLY_DIR.rglob("*.csv")):
        if not in_year_range(path, start_year, end_year):
            continue
        for row in read_csv_rows(path):
            bizno = norm_digits(row.get("bizno"))
            item_name = (row.get("dtilPrdctClsfcNoNm") or "").strip()
            item_code = norm_digits(row.get("dtilPrdctClsfcNo"))
            if bizno and item_code and item_name:
                flags = by_item[(item_code, item_name)][bizno]
                flags["대표물품"] = flags["대표물품"] or is_yes(row.get("rprsntPrdctClsfcNoNmYn"))
                flags["제조업체"] = flags["제조업체"] or is_yes(row.get("mnfctYn"))
    return by_item


def load_basic_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for path in sorted(BASIC_DIR.rglob("*.csv")):
        for row in read_csv_rows(path):
            bizno = norm_digits(row.get("bizno"))
            name = (row.get("corpNm") or "").strip()
            if bizno and name:
                names[bizno] = name
    return names


def load_award_stats(start_year: int, end_year: int) -> dict[str, dict[str, Any]]:
    # Award history is used only to prioritize eligible suppliers, not to decide eligibility itself.
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "amount": 0.0, "recent": "", "name": ""})
    for path in sorted(AWARDS_DIR.rglob("*.csv")):
        if "manifest" in path.name.lower():
            continue
        if not in_year_range(path, start_year, end_year):
            continue
        for row in read_csv_rows(path):
            bizno = norm_digits(row.get("bidwinnrBizno"))
            if not bizno:
                continue
            stat = stats[bizno]
            stat["count"] += 1
            stat["amount"] += parse_amount(row.get("sucsfbidAmt"))
            date = row.get("fnlSucsfDate") or row.get("rlOpengDt") or ""
            if date > stat["recent"]:
                stat["recent"] = date
            name = (row.get("bidwinnrNm") or "").strip()
            if name and not stat["name"]:
                stat["name"] = name
    return stats


def resolve_item_supply(
    supply_by_item: dict[tuple[str, str], dict[str, dict[str, bool]]],
    item_code: str,
    item_name: str,
) -> tuple[str, dict[str, dict[str, bool]]]:
    if item_name:
        return item_name, dict(supply_by_item.get((item_code, item_name), {}))

    matches = [(name, suppliers) for (code, name), suppliers in supply_by_item.items() if code == item_code]
    if not matches:
        return "", {}

    # A code can appear with slightly different names; merge suppliers and keep the most populated label.
    resolved_name = max(matches, key=lambda item: len(item[1]))[0]
    merged: dict[str, dict[str, bool]] = defaultdict(lambda: {"대표물품": False, "제조업체": False})
    for _, suppliers in matches:
        for bizno, flags in suppliers.items():
            merged[bizno]["대표물품"] = merged[bizno]["대표물품"] or flags.get("대표물품", False)
            merged[bizno]["제조업체"] = merged[bizno]["제조업체"] or flags.get("제조업체", False)
    return resolved_name, dict(merged)


def select_candidates(
    top_per_item: int,
    start_year: int,
    end_year: int,
    items: list[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    supply_by_item = load_supply_by_item(start_year, end_year)
    basic_names = load_basic_names()
    award_stats = load_award_stats(start_year, end_year)
    target_items = dedupe_items(items or DEFAULT_ITEMS)

    rows: list[dict[str, str]] = []
    summary: list[dict[str, Any]] = []

    candidate_id = 1
    for item_code, item_name in target_items:
        resolved_name, item_supply = resolve_item_supply(supply_by_item, item_code, item_name)
        output_name = item_name or resolved_name
        suppliers = sorted(item_supply)
        # Prefer suppliers with stronger public-procurement experience for the fixed-size dashboard pool.
        ranked = sorted(
            suppliers,
            key=lambda bizno: (
                -award_stats[bizno]["count"],
                -award_stats[bizno]["amount"],
                award_stats[bizno]["recent"] == "",
                basic_names.get(bizno) or award_stats[bizno]["name"] or "",
                bizno,
            ),
        )[:top_per_item]

        summary.append({
            "물품번호": item_code,
            "물품명": output_name,
            "공급등록업체수": len(suppliers),
            "낙찰이력보유업체수": sum(1 for bizno in suppliers if award_stats[bizno]["count"]),
            "선정업체수": len(ranked),
        })

        for bizno in ranked:
            flags = item_supply.get(bizno, {})
            rows.append({
                "candidate_id": candidate_id,
                "물품번호": item_code,
                "물품명": output_name,
                "사업자번호": bizno,
                "사업자번호_정규화": norm_digits(bizno),
                "업체명": basic_names.get(bizno) or award_stats[bizno]["name"] or "",
                "대표물품": "Y" if flags.get("대표물품") else "N",
                "제조업체": "Y" if flags.get("제조업체") else "N",
            })
            candidate_id += 1

    return rows, summary


def write_summary(
    path: Path,
    summary: list[dict[str, Any]],
    output: Path,
    start_year: int,
    end_year: int,
    top_per_item: int,
) -> None:
    try:
        output_label = output.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        output_label = output.as_posix()
    lines = [
        "# 품목 및 후보업체 선정 요약",
        "",
        f"현재 수집된 `조달업체공급물품정보조회` 데이터 중 {start_year}~{end_year}년 데이터를 기준으로 입력 품목 {len(summary)}개에 대한 후보 업체를 선정했다.",
        "",
        f"후보 업체는 각 품목의 공급물품 등록 업체 중 {start_year}~{end_year}년 나라장터 낙찰정보에서 낙찰건수와 낙찰금액이 높은 업체를 우선으로 최대 {top_per_item}개씩 선정했다.",
        "",
        "`공급물품_일치`는 점수 feature가 아니라 추천 후보에 들어오기 위한 필수 필터 조건이다.",
        "",
        "| 물품번호 | 물품명 | 공급 등록 업체 수 | 낙찰 이력 보유 업체 수 | 선정 업체 수 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['물품번호']} | {row['물품명']} | {row['공급등록업체수']:,} | "
            f"{row['낙찰이력보유업체수']:,} | {row['선정업체수']:,} |"
        )
    lines.extend([
        "",
        "생성 파일:",
        "",
        f"- `{output_label}`",
        "",
        "주의:",
        "",
        "- 이 파일은 실행 시점까지 수집된 조달업체공급물품정보 기준으로 생성된다.",
        f"- 현재 공유본은 {start_year}~{end_year}년 기준으로 재생성한 후보 업체 목록이다.",
        "- 2026년 데이터는 월별 수집 범위가 불완전하여 공유 기준 후보 선정과 낙찰 집계에서 제외한다.",
        "- 업체명은 조달업체 기본정보 또는 낙찰정보에서 확인된 이름을 우선 사용했다. 둘 다 없는 경우 빈칸으로 둔다.",
        "- `대표물품`, `제조업체`는 조달업체공급물품정보조회 원천 컬럼에서 후보 생성 시 자동 산출한다.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="품목별 후보 업체 목록을 생성합니다.")
    parser.add_argument(
        "--item",
        action="append",
        type=parse_item_spec,
        help="세부품명번호 또는 '세부품명번호:물품명'. 여러 번 입력 가능",
    )
    parser.add_argument("--items-csv", type=Path, help="물품번호/물품명 컬럼이 있는 입력 CSV")
    parser.add_argument("--top-per-item", type=int, default=50, help="품목별 후보 업체 수")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR, help="공급물품/낙찰정보 시작 연도")
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR, help="공급물품/낙찰정보 종료 연도")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="후보 업체 목록 CSV")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="선정 요약 Markdown")
    args = parser.parse_args()

    items = list(args.item or [])
    if args.items_csv:
        items.extend(load_items_csv(args.items_csv))
    if not items:
        items = DEFAULT_ITEMS

    rows, summary = select_candidates(args.top_per_item, args.start_year, args.end_year, items)
    write_csv(args.output, rows)
    write_summary(args.summary, summary, args.output, args.start_year, args.end_year, args.top_per_item)

    print(f"후보 업체 목록: {args.output} ({len(rows)}행, {args.start_year}~{args.end_year}년 기준)")
    for row in summary:
        print(
            f"{row['물품번호']} {row['물품명']}: "
            f"공급등록={row['공급등록업체수']:,}, "
            f"낙찰이력={row['낙찰이력보유업체수']:,}, "
            f"선정={row['선정업체수']:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
