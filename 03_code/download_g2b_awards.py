#!/usr/bin/env python
"""Download Nara Market goods award data from data.go.kr.

Set the API key only for the current process:
  $env:DATA_GO_KR_SERVICE_KEY = "..."

Single range:
  python 03_code/download_g2b_awards.py --start 202512010000 --end 202512312359

Monthly archive:
  python 03_code/download_g2b_awards.py --start-year 2022 --end-year 2025
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AWARDS_DIR = ROOT / "01_raw_data" / "나라장터_낙찰정보서비스"
DEFAULT_OUTPUT = AWARDS_DIR / "나라장터_낙찰정보서비스_물품_api_download.csv"
MANIFEST = AWARDS_DIR / "download_manifest.csv"
AWARDS_URL = "http://apis.data.go.kr/1230000/as/ScsbidInfoService/getScsbidListSttusThng"


def build_url(url: str, params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        # data.go.kr service keys are often already URL-encoded.
        if key == "serviceKey":
            encoded = str(value)
        else:
            encoded = urllib.parse.quote_plus(str(value))
        parts.append(f"{key}={encoded}")
    return f"{url}?{'&'.join(parts)}"


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request_url = build_url(url, params)
    request = urllib.request.Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    if not text.lstrip().startswith("{"):
        print(text[:1000], file=sys.stderr)
        raise RuntimeError("API did not return JSON.")
    return json.loads(text)


def coerce_items(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    body = data.get("response", {}).get("body", {})
    total_count = int(body.get("totalCount") or 0)
    items = body.get("items", {})
    if isinstance(items, list):
        return items, total_count
    item = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(item, dict):
        return [item], total_count
    if isinstance(item, list):
        return item, total_count
    return [], total_count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(row: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "year",
        "month",
        "start",
        "end",
        "path",
        "rows",
        "total_count",
        "status",
        "downloaded_at",
        "note",
    ]
    exists = MANIFEST.exists()
    with MANIFEST.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y%m%d").date()


def month_bounds(year: int, month: int, today: date | None = None) -> tuple[str, str, bool]:
    last_day = calendar.monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)
    if today and month_start > today:
        return f"{year}{month:02d}010000", f"{year}{month:02d}{last_day:02d}2359", True
    if today and month_start <= today <= month_end:
        month_end = today
    return f"{year}{month:02d}010000", f"{month_end:%Y%m%d}2359", False


def monthly_output_path(year: int, month: int) -> Path:
    last_day = calendar.monthrange(year, month)[1]
    return (
        AWARDS_DIR
        / f"{year}"
        / f"{month:02d}"
        / f"나라장터_낙찰정보서비스_물품_{year}{month:02d}01_{year}{month:02d}{last_day:02d}.csv"
    )


def download_range(
    service_key: str,
    start: str,
    end: str,
    inqry_div: str,
    rows_per_page: int,
    max_pages: int,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    all_rows: list[dict[str, Any]] = []
    total_count = 0
    # Page until the API totalCount is satisfied or the safety cap is reached.
    for page in range(1, max_pages + 1):
        data = request_json(
            AWARDS_URL,
            {
                "serviceKey": service_key,
                "pageNo": page,
                "numOfRows": rows_per_page,
                "inqryDiv": inqry_div,
                "inqryBgnDt": start,
                "inqryEndDt": end,
                "type": "json",
            },
        )
        rows, total_count = coerce_items(data)
        if not rows:
            break
        all_rows.extend(rows)
        if len(all_rows) >= total_count:
            break
        time.sleep(sleep_seconds)
    return all_rows, total_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Download G2B successful-bid goods data.")
    parser.add_argument("--start", help="조회 시작일시 YYYYMMDDHHMM")
    parser.add_argument("--end", help="조회 종료일시 YYYYMMDDHHMM")
    parser.add_argument("--start-year", type=int, help="월별 수집 시작 연도")
    parser.add_argument("--end-year", type=int, help="월별 수집 종료 연도")
    parser.add_argument("--today", help="현재 날짜 YYYYMMDD. 미래 월 스킵/현재 월 절단 기준")
    parser.add_argument("--skip-future", action="store_true", help="현재 날짜 이후 월은 폴더만 만들고 다운로드하지 않음")
    parser.add_argument("--inqry-div", default="1", help="1 등록일시, 2 공고일시, 3 개찰일시, 4 입찰공고번호")
    parser.add_argument("--rows", type=int, default=999, help="한 페이지 행 수")
    parser.add_argument("--max-pages", type=int, default=200, help="최대 조회 페이지")
    parser.add_argument("--sleep", type=float, default=0.15, help="페이지 요청 사이 대기 초")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY 환경변수가 필요합니다.")

    if args.start_year or args.end_year:
        if not args.start_year or not args.end_year:
            raise SystemExit("--start-year와 --end-year를 함께 지정해야 합니다.")
        if args.start_year > args.end_year:
            raise SystemExit("--start-year는 --end-year보다 클 수 없습니다.")

        today = parse_today(args.today)
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                output = monthly_output_path(year, month)
                output.parent.mkdir(parents=True, exist_ok=True)
                start, end, is_future = month_bounds(year, month, today)
                if args.skip_future and is_future:
                    # Keep a manifest row so skipped future periods are explicit.
                    append_manifest({
                        "year": year,
                        "month": f"{month:02d}",
                        "start": start,
                        "end": end,
                        "path": output,
                        "rows": 0,
                        "total_count": 0,
                        "status": "skipped_future",
                        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                        "note": "",
                    })
                    print(f"skip future: {year}-{month:02d}")
                    continue

                try:
                    rows, total_count = download_range(
                        service_key,
                        start,
                        end,
                        args.inqry_div,
                        args.rows,
                        args.max_pages,
                        args.sleep,
                    )
                    write_csv(output, rows)
                    status = "complete" if len(rows) >= total_count else "partial"
                    note = ""
                except Exception as exc:
                    rows = []
                    total_count = 0
                    status = "error"
                    note = str(exc)
                append_manifest({
                    "year": year,
                    "month": f"{month:02d}",
                    "start": start,
                    "end": end,
                    "path": output,
                    "rows": len(rows),
                    "total_count": total_count,
                    "status": status,
                    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                    "note": note,
                })
                print(f"{year}-{month:02d}: rows={len(rows)} total={total_count} status={status}")
        return 0

    if not args.start or not args.end:
        raise SystemExit("--start/--end 또는 --start-year/--end-year가 필요합니다.")

    all_rows, total_count = download_range(
        service_key,
        args.start,
        args.end,
        args.inqry_div,
        args.rows,
        args.max_pages,
        args.sleep,
    )

    write_csv(args.output, all_rows)
    print(f"saved: {args.output}")
    print(f"rows: {len(all_rows)}")
    print(f"totalCount: {total_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
