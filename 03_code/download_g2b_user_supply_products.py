#!/usr/bin/env python
"""Download G2B user-info supply product data by month.

Set the API key only for the current process:
  $env:DATA_GO_KR_SERVICE_KEY = "..."

Example:
  python 03_code/download_g2b_user_supply_products.py --start-year 2022 --end-year 2025
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPPLY_DIR = ROOT / "01_raw_data" / "나라장터_사용자정보서비스" / "4. 조달업체공급물품정보조회"
MANIFEST = SUPPLY_DIR / "download_manifest.csv"
URL = "https://apis.data.go.kr/1230000/ao/UsrInfoService02/getPrcrmntCorpSplyPrdctInfo02"


def build_url(url: str, params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        # Keep serviceKey as provided because many public-data keys are pre-encoded.
        encoded = str(value) if key == "serviceKey" else urllib.parse.quote_plus(str(value))
        parts.append(f"{key}={encoded}")
    return f"{url}?{'&'.join(parts)}"


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(build_url(url, params), headers={"User-Agent": "Mozilla/5.0"})
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


def output_path(inqry_div: str, year: int, month: int, day: int | None = None) -> Path:
    last_day = calendar.monthrange(year, month)[1]
    div_name = "등록일기준" if inqry_div == "2" else "변경일기준"
    if day is not None:
        return (
            SUPPLY_DIR
            / f"inqryDiv_{inqry_div}_{div_name}"
            / f"{year}"
            / f"{month:02d}"
            / f"{day:02d}"
            / f"사용자정보서비스_조달업체공급물품정보조회_inqryDiv_{inqry_div}_{year}{month:02d}{day:02d}_{year}{month:02d}{day:02d}.csv"
        )
    return (
        SUPPLY_DIR
        / f"inqryDiv_{inqry_div}_{div_name}"
        / f"{year}"
        / f"{month:02d}"
        / f"사용자정보서비스_조달업체공급물품정보조회_inqryDiv_{inqry_div}_{year}{month:02d}01_{year}{month:02d}{last_day:02d}.csv"
    )


def append_manifest(row: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "inqry_div",
        "year",
        "month",
        "day",
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


def load_complete_keys() -> set[tuple[str, str, str, str]]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        # Resume uses the manifest rather than file existence, so partial/error
        # downloads can be retried cleanly.
        return {
            (row.get("inqry_div", ""), row.get("year", ""), row.get("month", ""), row.get("day", ""))
            for row in rows
            if row.get("status") in {"complete", "skipped_future"}
        }


def download_range(
    service_key: str,
    inqry_div: str,
    start: str,
    end: str,
    rows_per_page: int,
    max_pages: int,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    all_rows: list[dict[str, Any]] = []
    total_count = 0
    for page in range(1, max_pages + 1):
        data = request_json(
            URL,
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
    parser = argparse.ArgumentParser(description="Download G2B supplier supply-product data.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--today", help="현재 날짜 YYYYMMDD. 미래 월 스킵/현재 월 절단 기준")
    parser.add_argument("--skip-future", action="store_true")
    parser.add_argument("--months", help="수집할 월. 예: 4,8,12")
    parser.add_argument("--inqry-divs", default="2,3", help="쉼표 구분. 2=등록일, 3=변경일")
    parser.add_argument("--granularity", choices=("month", "day"), default="day")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rows", type=int, default=999)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true", help="매니페스트상 완료/미래스킵 월은 건너뜀")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("--start-year는 --end-year보다 클 수 없습니다.")

    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY 환경변수가 필요합니다.")

    today = parse_today(args.today)
    complete_keys = load_complete_keys() if args.resume else set()
    inqry_divs = [part.strip() for part in args.inqry_divs.split(",") if part.strip()]
    months = [int(part.strip()) for part in args.months.split(",")] if args.months else list(range(1, 13))
    invalid_months = [month for month in months if month < 1 or month > 12]
    if invalid_months:
        raise SystemExit(f"잘못된 월입니다: {invalid_months}")

    for inqry_div in inqry_divs:
        if inqry_div not in {"2", "3"}:
            raise SystemExit("--inqry-divs는 2 또는 3만 지원합니다.")
        for year in range(args.start_year, args.end_year + 1):
            for month in months:
                key = (inqry_div, str(year), f"{month:02d}", "")
                output = output_path(inqry_div, year, month)
                output.parent.mkdir(parents=True, exist_ok=True)
                if key in complete_keys:
                    print(f"resume skip: div={inqry_div} {year}-{month:02d}")
                    continue
                start, end, is_future = month_bounds(year, month, today)
                if args.skip_future and is_future:
                    append_manifest({
                        "inqry_div": inqry_div,
                        "year": year,
                        "month": f"{month:02d}",
                        "day": "",
                        "start": start,
                        "end": end,
                        "path": output,
                        "rows": 0,
                        "total_count": 0,
                        "status": "skipped_future",
                        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                        "note": "",
                    })
                    print(f"skip future: div={inqry_div} {year}-{month:02d}")
                    continue

                if args.granularity == "month":
                    try:
                        rows, total_count = download_range(
                            service_key,
                            inqry_div,
                            start,
                            end,
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
                        "inqry_div": inqry_div,
                        "year": year,
                        "month": f"{month:02d}",
                        "day": "",
                        "start": start,
                        "end": end,
                        "path": output,
                        "rows": len(rows),
                        "total_count": total_count,
                        "status": status,
                        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                        "note": note,
                    })
                    print(f"div={inqry_div} {year}-{month:02d}: rows={len(rows)} total={total_count} status={status}")
                    continue

                # Daily jobs avoid very large monthly API responses and can run in parallel.
                last_day = calendar.monthrange(year, month)[1]
                day_jobs = []
                for day in range(1, last_day + 1):
                    day_date = date(year, month, day)
                    if args.skip_future and day_date > today:
                        continue
                    day_key = (inqry_div, str(year), f"{month:02d}", f"{day:02d}")
                    if day_key in complete_keys:
                        print(f"resume skip: div={inqry_div} {year}-{month:02d}-{day:02d}")
                        continue
                    day_start = f"{year}{month:02d}{day:02d}0000"
                    day_end = f"{year}{month:02d}{day:02d}2359"
                    day_output = output_path(inqry_div, year, month, day)
                    day_output.parent.mkdir(parents=True, exist_ok=True)
                    day_jobs.append((day, day_start, day_end, day_output))

                def run_day(job: tuple[int, str, str, Path]) -> dict[str, Any]:
                    day, day_start, day_end, day_output = job
                    try:
                        rows, total_count = download_range(
                            service_key,
                            inqry_div,
                            day_start,
                            day_end,
                            args.rows,
                            args.max_pages,
                            args.sleep,
                        )
                        write_csv(day_output, rows)
                        status = "complete" if len(rows) >= total_count else "partial"
                        note = ""
                    except Exception as exc:
                        rows = []
                        total_count = 0
                        status = "error"
                        note = str(exc)
                    return {
                        "inqry_div": inqry_div,
                        "year": year,
                        "month": f"{month:02d}",
                        "day": f"{day:02d}",
                        "start": day_start,
                        "end": day_end,
                        "path": day_output,
                        "rows": len(rows),
                        "total_count": total_count,
                        "status": status,
                        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                        "note": note,
                    }

                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                    future_map = {executor.submit(run_day, job): job for job in day_jobs}
                    for future in as_completed(future_map):
                        row = future.result()
                        append_manifest(row)
                        print(
                            f"div={row['inqry_div']} {row['year']}-{row['month']}-{row['day']}: "
                            f"rows={row['rows']} total={row['total_count']} status={row['status']}"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
