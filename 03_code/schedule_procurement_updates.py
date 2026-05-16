#!/usr/bin/env python
"""Schedule periodic procurement dashboard updates.

This is a lightweight scheduler for local/demo operation. For production, run
`run_procurement_dashboard_pipeline.py` from cron, systemd timer, Airflow, or
another scheduler using the same arguments.

Examples:
  python 03_code/schedule_procurement_updates.py --once -- --item 4710160801
  python 03_code/schedule_procurement_updates.py --run-now --interval-hours 24 -- --items-csv input_items.csv
  python 03_code/schedule_procurement_updates.py --daily-at 02:00 -- --item 4710160801 --use-api
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "03_code"
PROCESSED = ROOT / "02_processed"
PIPELINE_SCRIPT = SCRIPTS / "run_procurement_dashboard_pipeline.py"
SCHEDULE_STATUS = PROCESSED / "pipeline_schedule.json"
DEFAULT_LOG = PROCESSED / "pipeline_update.log"


def now() -> dt.datetime:
    return dt.datetime.now()


def format_time(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def parse_hhmm(value: str) -> tuple[int, int]:
    try:
        hour, minute = value.split(":", 1)
        hour_i = int(hour)
        minute_i = int(minute)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("시간은 HH:MM 형식이어야 합니다.") from exc
    if not 0 <= hour_i <= 23 or not 0 <= minute_i <= 59:
        raise argparse.ArgumentTypeError("시간은 00:00~23:59 범위여야 합니다.")
    return hour_i, minute_i


def schedule_label(args: argparse.Namespace) -> str:
    if args.once:
        return "one-shot"
    if args.daily_at:
        return f"daily {args.daily_at}"
    minutes = args.interval_minutes + args.interval_hours * 60
    return f"every {minutes} minutes"


def next_daily_run(daily_at: str, base: dt.datetime | None = None) -> dt.datetime:
    base = base or now()
    hour, minute = parse_hhmm(daily_at)
    target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= base:
        target += dt.timedelta(days=1)
    return target


def next_interval_run(args: argparse.Namespace, base: dt.datetime | None = None) -> dt.datetime:
    base = base or now()
    minutes = args.interval_minutes + args.interval_hours * 60
    if minutes <= 0:
        minutes = 24 * 60
    return base + dt.timedelta(minutes=minutes)


def write_schedule_status(payload: dict[str, object]) -> None:
    SCHEDULE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pipeline_command(args: argparse.Namespace, pipeline_args: list[str]) -> list[str]:
    # Pass through the same pipeline arguments used for manual dashboard updates.
    return [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--trigger",
        "scheduled",
        "--schedule-label",
        schedule_label(args),
        *pipeline_args,
    ]


def run_pipeline_once(args: argparse.Namespace, pipeline_args: list[str]) -> int:
    command = pipeline_command(args, pipeline_args)
    started = now()
    log_path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Status JSON lets the static dashboard/report show whether the last update succeeded.
    write_schedule_status(
        {
            "status": "running",
            "schedule": schedule_label(args),
            "last_run_started_at": format_time(started),
            "pipeline_args": pipeline_args,
            "log_file": str(log_path),
        }
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {format_time(started)} START {' '.join(command)} =====\n")
        log.flush()
        # Capture stdout/stderr in one log file for unattended scheduled runs.
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        finished = now()
        log.write(f"===== {format_time(finished)} END exit={result.returncode} =====\n")

    write_schedule_status(
        {
            "status": "waiting" if result.returncode == 0 else "failed",
            "schedule": schedule_label(args),
            "last_run_started_at": format_time(started),
            "last_run_finished_at": format_time(finished),
            "last_run_exit_code": result.returncode,
            "pipeline_args": pipeline_args,
            "log_file": str(log_path),
        }
    )
    return result.returncode


def sleep_until(target: dt.datetime) -> None:
    while True:
        seconds = (target - now()).total_seconds()
        if seconds <= 0:
            return
        time.sleep(min(seconds, 60))


def main() -> int:
    parser = argparse.ArgumentParser(description="대시보드 파이프라인을 주기적으로 실행합니다.")
    parser.add_argument("--once", action="store_true", help="한 번만 실행하고 종료합니다.")
    parser.add_argument("--run-now", action="store_true", help="시작 즉시 한 번 실행한 뒤 주기 실행합니다.")
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--interval-minutes", type=int, default=0)
    parser.add_argument("--daily-at", help="매일 실행할 시각, HH:MM")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    args, pipeline_args = parser.parse_known_args()

    if pipeline_args and pipeline_args[0] == "--":
        pipeline_args = pipeline_args[1:]

    if args.once:
        return run_pipeline_once(args, pipeline_args)

    if args.run_now:
        run_pipeline_once(args, pipeline_args)

    while True:
        target = next_daily_run(args.daily_at) if args.daily_at else next_interval_run(args)
        write_schedule_status(
            {
                "status": "waiting",
                "schedule": schedule_label(args),
                "next_run_at": format_time(target),
                "pipeline_args": pipeline_args,
                "log_file": str(args.log_file),
            }
        )
        print(f"next update: {format_time(target)}")
        sleep_until(target)
        run_pipeline_once(args, pipeline_args)


if __name__ == "__main__":
    raise SystemExit(main())
