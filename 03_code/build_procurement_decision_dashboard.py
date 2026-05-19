#!/usr/bin/env python
"""Build an input-driven procurement decision dashboard.

The dashboard is a self-contained HTML file. It embeds the current official
candidate set, teammate feature files, address fallbacks, and the 05/13 HGB
model output so users can enter an item code and budget and immediately review
ranked supplier groups.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "02_processed" if (ROOT / "02_processed" / "후보업체_목록.csv").exists() else ROOT / "04_outputs"
STANDARD = PROCESSED / "한국어_표준데이터셋"
FEATURE_DIR = PROCESSED / "feature_inputs"
MODEL_OUTPUTS = PROCESSED / "model_outputs"
DASHBOARD_DIR = ROOT / "02_dashboard"
DASHBOARD_HTML = DASHBOARD_DIR / "procurement_decision_dashboard.html"
PIPELINE_STATUS = PROCESSED / "pipeline_status.json"

CANDIDATES = PROCESSED / "후보업체_목록.csv"
YEONWOO = FEATURE_DIR / "연우_낙찰위험_features.csv"
YEONSU = FEATURE_DIR / "연수_정책인증_features.csv"
JUNWOO = FEATURE_DIR / "준우_사회적가치_features.csv"
HGB_ZIP = ROOT / "05_15_추가" / "0513 adjusted.zip"
HGB_PRED = "outputs/0513 histgradientboosting 20-24train/HGB_후보업체_AI예측점수.csv"
LOCAL_HGB_PRED = MODEL_OUTPUTS / "HGB_후보업체_AI예측점수.csv"
LOCAL_HGB_COMPARE = MODEL_OUTPUTS / "HGB_후보업체_AI_규칙점수_비교.csv"
CURRENT_AI = MODEL_OUTPUTS / "후보업체_AI_규칙점수_비교.csv"
PRESET_ORDER = ["균형형", "안정성 중심", "사회적 가치 중심", "위험 회피 중심"]

def norm_digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def read_zip_csv(zip_path: Path, member: str) -> list[dict[str, str]]:
    if not zip_path.exists():
        return []
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            return []
        with archive.open(member) as f:
            text = f.read().decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    return int(round(to_float(value, default)))


def yes(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"y", "yes", "true", "1", "o", "예", "있음", "해당", "인증", "보유"}


def newest(paths: Iterable[Path]) -> str:
    times = [p.stat().st_mtime for p in paths if p.exists()]
    if not times:
        return ""
    return dt.datetime.fromtimestamp(max(times)).strftime("%Y-%m-%d %H:%M")


def feature_index(path: Path) -> dict[str, dict[str, str]]:
    return {str(row.get("candidate_id") or "").strip(): row for row in read_csv_rows(path)}


def load_hgb_predictions() -> dict[str, dict[str, str]]:
    # Prefer the most recently generated HGB output; bundled/older outputs are fallback only.
    rows = read_csv_rows(LOCAL_HGB_PRED)
    if not rows:
        rows = read_zip_csv(HGB_ZIP, HGB_PRED)
    if not rows:
        rows = read_csv_rows(CURRENT_AI)
    return {str(row.get("candidate_id") or "").strip(): row for row in rows}


def load_rule_comparison() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(LOCAL_HGB_COMPARE)
    if not rows:
        rows = read_csv_rows(CURRENT_AI)
    return {str(row.get("candidate_id") or "").strip(): row for row in rows}


def preset_result_paths() -> list[Path]:
    nested = PROCESSED / "scored_outputs" / "후보업체_목록"
    flat = PROCESSED / "scored_outputs"
    paths = sorted(nested.glob("업체추천_점수_*.csv"))
    if not paths:
        paths = sorted(flat.glob("업체추천_점수_*.csv"))
    return paths


def load_preset_results() -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    found: list[str] = []
    for path in preset_result_paths():
        preset = path.stem.replace("업체추천_점수_", "")
        found.append(preset)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv_rows(path):
            grouped[row.get("물품번호") or row.get("물품명") or ""].append(row)
        # Preset rank is meaningful within each item, not across unrelated procurement items.
        for item_rows in grouped.values():
            item_rows.sort(key=lambda row: to_float(row.get("프리셋점수")), reverse=True)
            for rank, row in enumerate(item_rows, 1):
                cid = str(row.get("candidate_id") or "").strip()
                if not cid:
                    continue
                by_candidate[cid][preset] = {
                    "rank": rank,
                    "score": round(to_float(row.get("프리셋점수")), 4),
                    "detail": row.get("프리셋점수상세", ""),
                }
    presets = [p for p in PRESET_ORDER if p in found]
    presets.extend(p for p in sorted(set(found)) if p not in PRESET_ORDER)
    return dict(by_candidate), presets


def load_company_info(candidate_biz: set[str]) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_csv_rows(STANDARD / "조달청_나라장터_사용자정보서비스_조달업체기본정보조회.csv"):
        biz = norm_digits(row.get("사업자번호"))
        if biz not in candidate_biz:
            continue
        address = " ".join(part for part in [row.get("주소", ""), row.get("상세주소", "")] if part).strip()
        info[biz].update(
            {
                "address": address,
                "region": row.get("지역명", ""),
                "representative": row.get("대표자명", ""),
                "phone": row.get("전화번호", ""),
                "business_type": row.get("업체업무구분명", ""),
            }
        )

    latest_award_date: dict[str, str] = {}
    for row in read_csv_rows(STANDARD / "조달청_나라장터_낙찰정보서비스_물품.csv"):
        biz = norm_digits(row.get("사업자번호"))
        if biz not in candidate_biz:
            continue
        date = row.get("최종낙찰일자") or row.get("개찰일시") or row.get("등록일시") or ""
        if date < latest_award_date.get(biz, ""):
            continue
        latest_award_date[biz] = date
        current = info[biz]
        current.setdefault("address", row.get("업체주소", ""))
        current.setdefault("representative", row.get("대표자명", ""))
        current.setdefault("phone", row.get("업체전화번호", ""))
        current["latest_award_address"] = row.get("업체주소", "")
    return dict(info)


def load_supply_periods(candidate_pairs: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    periods: dict[tuple[str, str], dict[str, str]] = {}
    path = STANDARD / "조달청_나라장터_사용자정보서비스_조달업체공급물품정보조회.csv"
    for row in read_csv_rows(path):
        pair = ((row.get("세부품명번호") or "").strip(), norm_digits(row.get("사업자번호")))
        if pair not in candidate_pairs:
            continue
        entry = periods.setdefault(
            pair,
            {
                "registered_at": "",
                "changed_at": "",
                "representative_item": "N",
                "manufacturer": "N",
            },
        )
        reg = row.get("등록일시", "")
        chg = row.get("변경일시", "")
        if reg and (not entry["registered_at"] or reg < entry["registered_at"]):
            entry["registered_at"] = reg
        if chg and chg > entry["changed_at"]:
            entry["changed_at"] = chg
        if yes(row.get("대표물품여부")):
            entry["representative_item"] = "Y"
        if yes(row.get("제조여부")):
            entry["manufacturer"] = "Y"
    return periods


def indicator_names(row: dict[str, Any]) -> list[str]:
    names = []
    for key, label in [
        ("startup", "창업기업"),
        ("disabled_workplace", "장애인표준사업장"),
        ("public_purchase_cert", "공공구매인증"),
        ("social_enterprise", "사회적기업"),
        ("social_coop", "사회적협동조합"),
        ("veteran_factory", "자활용사촌복지공장"),
        ("production_item_match", "생산품목매칭"),
    ]:
        if row.get(key):
            names.append(label)
    return names


def build_records() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    # One dashboard record combines candidate identity, AI score, preset scores, and explainable indicators.
    candidates = read_csv_rows(CANDIDATES)
    yeonwoo = feature_index(YEONWOO)
    yeonsu = feature_index(YEONSU)
    junwoo = feature_index(JUNWOO)
    ai = load_hgb_predictions()
    rule = load_rule_comparison()
    preset_by_candidate, presets = load_preset_results()

    candidate_biz = {norm_digits(row.get("사업자번호_정규화") or row.get("사업자번호")) for row in candidates}
    candidate_pairs = {
        ((row.get("물품번호") or "").strip(), norm_digits(row.get("사업자번호_정규화") or row.get("사업자번호")))
        for row in candidates
    }
    company_info = load_company_info(candidate_biz)
    supply_periods = load_supply_periods(candidate_pairs)

    records: list[dict[str, Any]] = []
    items: list[dict[str, str]] = []
    seen_items = set()
    for base in candidates:
        cid = str(base.get("candidate_id") or "").strip()
        item_code = (base.get("물품번호") or "").strip()
        biz = norm_digits(base.get("사업자번호_정규화") or base.get("사업자번호"))
        item_name = base.get("물품명", "")
        if item_code not in seen_items:
            seen_items.add(item_code)
            items.append({"code": item_code, "name": item_name})

        yw = yeonwoo.get(cid, {})
        ys = yeonsu.get(cid, {})
        jw = junwoo.get(cid, {})
        pred = ai.get(cid, {})
        rule_row = rule.get(cid, {})
        balanced_preset = preset_by_candidate.get(cid, {}).get("균형형", {})
        info = company_info.get(biz, {})
        period = supply_periods.get((item_code, biz), {})
        rec = {
            "candidate_id": to_int(cid),
            "item_code": item_code,
            "item_name": item_name,
            "biz_no": biz,
            "company": base.get("업체명", ""),
            "representative_name": info.get("representative", ""),
            "address": info.get("address") or info.get("latest_award_address", ""),
            "region": info.get("region", ""),
            "business_type": info.get("business_type", ""),
            "representative_item": yes(period.get("representative_item") or base.get("대표물품")),
            "manufacturer": yes(period.get("manufacturer") or base.get("제조업체")),
            "supply_registered_at": period.get("registered_at", ""),
            "supply_changed_at": period.get("changed_at", ""),
            "ai_rank": to_int(pred.get("물품별_AI순위") or pred.get("물품별_AI순위")),
            "global_ai_rank": to_int(pred.get("전체_AI순위")),
            "ai_score": round(to_float(pred.get("AI예측점수")), 6),
            # The dashboard treats AI as the primary rank and presets as interpretation lenses.
            "rule_rank": to_int(
                balanced_preset.get("rank")
                or pred.get("규칙기반_균형형_물품별순위")
                or rule_row.get("규칙기반_균형형_물품별순위")
            ),
            "rule_score": round(
                to_float(
                    balanced_preset.get("score")
                    or pred.get("규칙기반_균형형점수")
                    or rule_row.get("규칙기반_균형형점수")
                ),
                4,
            ),
            "presets": preset_by_candidate.get(cid, {}),
            "award_count": to_float(yw.get("낙찰건수") or pred.get("이전전체낙찰건수")),
            "recent_award_count": to_float(yw.get("최근낙찰건수") or pred.get("최근365일낙찰건수")),
            "last_award_date": yw.get("최근낙찰일", ""),
            "avg_award_amount": to_float(yw.get("평균낙찰금액")),
            "award_amount_sum": to_float(yw.get("낙찰금액합계")),
            "sanction": yes(yw.get("부정당제재") or pred.get("부정당제재")),
            "sanction_count": to_float(yw.get("제재횟수")),
            "current_sanction": yes(yw.get("현재제재여부")),
            "sanction_end_date": yw.get("제재종료일", ""),
            "startup": yes(ys.get("창업기업") or pred.get("창업기업")),
            "disabled_workplace": yes(ys.get("장애인표준사업장") or pred.get("장애인표준사업장")),
            "public_purchase_cert": yes(ys.get("공공구매인증서")),
            "cert_count": to_float(ys.get("인증개수")),
            "cert_valid": yes(ys.get("인증유효여부")),
            "policy_match": ys.get("매칭방식", ""),
            "policy_note": ys.get("비고", ""),
            "social_enterprise": yes(jw.get("사회적기업") or pred.get("사회적기업")),
            "social_coop": yes(jw.get("사회적협동조합")),
            "veteran_factory": yes(jw.get("자활용사촌복지공장")),
            "production_item_match": yes(jw.get("생산품목매칭")),
        }
        rec["indicator_names"] = indicator_names(rec)
        rec["indicator_count"] = len(rec["indicator_names"])
        rec["risk_flags"] = [
            label
            for flag, label in [
                (rec["current_sanction"], "현재제재"),
                (rec["sanction"], "제재이력"),
            ]
            if flag
        ]
        records.append(rec)

    records.sort(key=lambda row: (row["item_code"], row.get("ai_rank") or 9999))
    return records, items, presets


def build_payload() -> dict[str, Any]:
    records, items, presets = build_records()
    source_paths = [
        CANDIDATES,
        YEONWOO,
        YEONSU,
        JUNWOO,
        LOCAL_HGB_PRED,
        LOCAL_HGB_COMPARE,
        PIPELINE_STATUS,
        HGB_ZIP,
        *preset_result_paths(),
        STANDARD / "조달청_나라장터_사용자정보서비스_조달업체기본정보조회.csv",
        STANDARD / "조달청_나라장터_낙찰정보서비스_물품.csv",
    ]
    return {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_updated_at": newest(source_paths),
        "items": items,
        "presets": presets,
        "pipeline_status": read_json(PIPELINE_STATUS),
        "records": records,
    }


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 기반 공급망 의사결정 대시보드</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #1d2733;
      --muted: #667085;
      --line: #d8dee8;
      --blue: #2866c7;
      --green: #177650;
      --amber: #a76513;
      --red: #c33d33;
      --ink: #102033;
      --chip: #edf2f7;
      --shadow: 0 10px 24px rgba(16, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .topbar {
      max-width: 1480px;
      margin: 0 auto;
      padding: 16px 24px;
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 23px;
      line-height: 1.2;
      font-weight: 800;
    }
    .subtitle {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    main {
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px 24px 44px;
    }
    .input-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.9fr 0.9fr 0.9fr;
      gap: 10px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    input, select {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
      font-size: 14px;
      outline: none;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      margin-bottom: 16px;
    }
    .section-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
    }
    h2 {
      margin: 0;
      font-size: 17px;
      line-height: 1.2;
    }
    .section-note {
      color: var(--muted);
      font-size: 12px;
    }
    .body-pad { padding: 14px 16px 16px; }
    .layout {
      display: block;
    }
    .visual-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      margin-bottom: 16px;
      align-items: stretch;
    }
    .viz-pad {
      padding: 14px 16px 16px;
      min-height: 0;
    }
    .score-list {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(5, minmax(170px, 1fr));
    }
    .score-row {
      display: grid;
      gap: 9px;
      min-width: 0;
      padding: 13px;
      border: 1px solid #e3e9f2;
      border-radius: 8px;
      background: #fbfcfe;
    }
    .score-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      font-size: 13px;
    }
    .score-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 800;
    }
    .score-value {
      font-weight: 850;
      color: var(--blue);
      font-variant-numeric: tabular-nums;
    }
    .total-bar,
    .stack-bar {
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: #e7edf4;
    }
    .total-fill {
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2866c7, #28a06b);
    }
    .stack-bar {
      display: flex;
      height: 8px;
    }
    .score-meta {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .score-metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .score-metric {
      min-width: 0;
      border-radius: 6px;
      background: #eef3f8;
      padding: 6px;
      text-align: center;
    }
    .score-metric b {
      display: block;
      color: var(--ink);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .rank-gap {
      font-weight: 800;
      color: var(--blue);
    }
    .seg { height: 100%; min-width: 1px; }
    .seg.ai { background: #2866c7; }
    .seg.preset { background: #7c3aed; }
    .seg.stability { background: #177650; }
    .seg.social { background: #8b5cf6; }
    .seg.budget { background: #e08a1e; }
    .seg.validity { background: #14a3a5; }
    .seg.risk { background: #c33d33; }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
      background: #64748b;
    }
    .scatter-wrap {
      min-height: 230px;
    }
    .scatter-wrap svg {
      width: 100%;
      height: 240px;
      display: block;
    }
    .axis {
      stroke: #cbd5e1;
      stroke-width: 1;
    }
    .grid-line {
      stroke: #eef2f7;
      stroke-width: 1;
    }
    .axis-label {
      fill: var(--muted);
      font-size: 11px;
    }
    .point {
      stroke: #fff;
      stroke-width: 1.5;
      opacity: 0.9;
    }
    .histogram {
      height: 210px;
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 6px;
      align-items: end;
      padding-top: 6px;
    }
    .hist-bin {
      display: grid;
      grid-template-rows: 1fr auto;
      gap: 7px;
      height: 100%;
      min-width: 0;
    }
    .hist-bar {
      align-self: end;
      border-radius: 5px 5px 0 0;
      background: #2866c7;
      min-height: 3px;
    }
    .hist-label {
      text-align: center;
      color: var(--muted);
      font-size: 10px;
      white-space: nowrap;
    }
    .table-wrap { overflow: auto; max-height: 590px; }
    .ranking-wrap { max-height: 820px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid #edf1f5;
      padding: 10px 11px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #405064;
      font-size: 12px;
      font-weight: 800;
    }
    tbody tr:hover { background: #f7fbff; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .rank {
      display: inline-grid;
      place-items: center;
      min-width: 30px;
      height: 28px;
      border-radius: 6px;
      background: #eaf2ff;
      color: var(--blue);
      font-weight: 850;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      min-width: 140px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 23px;
      border-radius: 999px;
      background: var(--chip);
      color: #415066;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }
    .chip.good { background: #e8f6ef; color: var(--green); }
    .chip.warn { background: #fff1db; color: var(--amber); }
    .chip.risk { background: #ffeceb; color: var(--red); }
    .meter {
      width: 110px;
      height: 8px;
      overflow: hidden;
      background: #e7edf4;
      border-radius: 999px;
      margin-top: 5px;
    }
    .meter > span {
      display: block;
      height: 100%;
      background: var(--blue);
      width: 0;
    }
    .muted { color: var(--muted); }
    .small { font-size: 12px; }
    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
    }
    .source-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .source-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }
    .segment-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .segment-toggle {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfe;
      color: #405064;
      padding: 6px 11px;
      font-size: 12px;
      font-weight: 800;
    }
    .segment-toggle input {
      width: 14px;
      height: 14px;
      accent-color: var(--blue);
    }
    .segment-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 10px;
    }
    .segment-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 12px;
      min-width: 0;
    }
    .segment-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 850;
      color: var(--ink);
    }
    .segment-count {
      color: var(--blue);
      font-variant-numeric: tabular-nums;
    }
    .segment-desc {
      margin-top: 5px;
      min-height: 16px;
      color: var(--muted);
      font-size: 12px;
    }
    .segment-stats {
      display: grid;
      gap: 6px;
      margin-top: 10px;
      font-size: 12px;
    }
    .segment-stat {
      display: grid;
      grid-template-columns: 58px 1fr 36px;
      gap: 7px;
      align-items: center;
    }
    .bar-bg {
      height: 9px;
      border-radius: 999px;
      background: #e7edf4;
      overflow: hidden;
    }
    .bar {
      height: 100%;
      border-radius: 999px;
      background: var(--green);
    }
    .segment-table-wrap {
      margin-top: 14px;
      max-height: 420px;
    }
    @media (max-width: 1120px) {
      .topbar, .layout, .input-grid, .visual-grid { grid-template-columns: 1fr; }
      .score-list { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
      .segment-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
    }
    @media (max-width: 680px) {
      .topbar, main { padding-left: 14px; padding-right: 14px; }
      .score-list { grid-template-columns: 1fr; }
      .segment-grid { grid-template-columns: 1fr; }
      .source-grid { grid-template-columns: 1fr; }
      th, td { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>AI 기반 공급망 의사결정 대시보드</h1>
        <div class="subtitle">세부품명번호 · 사업자등록번호 PK · 사회적 책임 지표 · 낙찰/제재 분석</div>
      </div>
      <div class="section-note">생성 <span id="generatedAt"></span> · 데이터 <span id="sourceUpdatedAt"></span> · 갱신 <span id="pipelineStatus"></span></div>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>발주계획 입력</h2>
        <div class="section-note">AI 순위는 고정하고 프리셋은 해석 기준으로만 적용합니다</div>
      </div>
      <div class="body-pad">
        <div class="input-grid">
          <label>세부품명번호
            <input id="itemCodeInput" list="itemCodeList" inputmode="numeric">
            <datalist id="itemCodeList"></datalist>
          </label>
          <label>추정금액
            <input id="budgetInput" inputmode="numeric" value="100000000">
          </label>
          <label>사업자등록번호
            <input id="bizInput" inputmode="numeric" placeholder="선택 입력">
          </label>
          <label>해석 프리셋
            <select id="presetSelect"></select>
          </label>
        </div>
      </div>
    </section>

    <div class="visual-grid">
      <section class="top-five-panel">
        <div class="section-head">
          <h2>AI Top 5 프리셋 해석</h2>
          <div class="section-note">AI 추천 상위 5개 후보를 선택한 실무 프리셋 기준으로 함께 해석</div>
        </div>
        <div class="viz-pad">
          <div id="scoreBars" class="score-list"></div>
          <div class="legend">
            <span><i class="dot seg ai"></i>AI</span>
            <span><i class="dot seg preset"></i>선택 프리셋</span>
          </div>
        </div>
      </section>

      <section class="top-five-panel">
        <div class="section-head">
          <h2>프리셋 Top 5 순위</h2>
          <div class="section-note">선택한 실무 프리셋이 우선 검토하는 후보</div>
        </div>
        <div class="viz-pad">
          <div id="presetBars" class="score-list"></div>
          <div class="legend">
            <span><i class="dot seg preset"></i>선택 프리셋</span>
            <span><i class="dot seg ai"></i>AI</span>
          </div>
        </div>
      </section>
    </div>

    <div class="layout">
      <div>
        <section>
          <div class="section-head">
            <h2>공급망 순위</h2>
            <div class="section-note" id="rankingNote"></div>
          </div>
          <div class="table-wrap ranking-wrap">
            <table>
              <thead>
                <tr>
                  <th>AI순위</th>
                  <th>업체명</th>
                  <th>사업자번호</th>
                  <th>대표/제조</th>
                  <th>주소</th>
                  <th>프리셋</th>
                  <th>지표합</th>
                  <th>지표명</th>
                  <th>낙찰 실적</th>
                  <th>예산 적합</th>
                  <th>제재</th>
                </tr>
              </thead>
              <tbody id="rankingBody"></tbody>
            </table>
          </div>
        </section>

        <section>
          <div class="section-head">
            <h2>기업별 검증 상세</h2>
            <div class="section-note">세부품명번호 + 사업자등록번호 기준</div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>candidate</th>
                  <th>업체</th>
                  <th>AI</th>
                  <th>프리셋</th>
                  <th>안정성</th>
                  <th>사회책임</th>
                  <th>공급등록</th>
                  <th>인증유효</th>
                  <th>최근낙찰일</th>
                  <th>평균낙찰금액</th>
                </tr>
              </thead>
              <tbody id="detailBody"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>

    <section>
      <div class="section-head">
        <h2>지표 기반 후보군 탐색</h2>
        <div class="section-note">선택 지표별 후보군과 관련 데이터를 확인합니다</div>
      </div>
      <div class="body-pad">
        <div class="segment-controls">
          <label class="segment-toggle"><input type="checkbox" class="segment-check" value="multi" checked>다중지표 보유</label>
          <label class="segment-toggle"><input type="checkbox" class="segment-check" value="awardCount" checked>낙찰건수 상위</label>
          <label class="segment-toggle"><input type="checkbox" class="segment-check" value="awardAmount">낙찰금액 상위</label>
          <label class="segment-toggle"><input type="checkbox" class="segment-check" value="sanction">부정당제재</label>
          <label class="segment-toggle"><input type="checkbox" class="segment-check" value="budgetFit">예산 적합</label>
        </div>
        <div id="segmentSummary" class="segment-grid"></div>
        <div class="table-wrap segment-table-wrap">
          <table>
            <thead>
              <tr>
                <th>후보군</th>
                <th>업체</th>
                <th>AI</th>
                <th>프리셋</th>
                <th>지표</th>
                <th>낙찰건수</th>
                <th>낙찰금액</th>
                <th>제재</th>
              </tr>
            </thead>
            <tbody id="segmentBody"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <script>
    const DATA = __PAYLOAD__;

    const fmt = new Intl.NumberFormat("ko-KR");
    const moneyFmt = new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 });
    const $ = (id) => document.getElementById(id);

    function asNumber(value) {
      const n = Number(String(value || "").replaceAll(",", "").replace(/[^\d.-]/g, ""));
      return Number.isFinite(n) ? n : 0;
    }

    function yes(value) { return value === true || value === 1 || value === "Y"; }
    function pct(value) { return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`; }
    function safeText(value, fallback = "-") { return value === undefined || value === null || value === "" ? fallback : value; }
    function shortAddress(value) { return value ? String(value).replace(/\s+/g, " ").slice(0, 56) : "-"; }

    function chips(labels, cls = "") {
      if (!labels.length) return `<span class="chip">없음</span>`;
      return `<div class="chips">${labels.map((label) => `<span class="chip ${cls}">${label}</span>`).join("")}</div>`;
    }

    function itemName(code) {
      const found = DATA.items.find((item) => item.code === code);
      return found ? found.name : code;
    }

    function selectedPresetName() {
      return $("presetSelect").value || DATA.presets?.[0] || "균형형";
    }

    function currentBaseRows() {
      const code = $("itemCodeInput").value.trim();
      const biz = $("bizInput").value.trim().replace(/\D/g, "");
      let rows = DATA.records.filter((row) => row.item_code === code);
      if (biz) rows = rows.filter((row) => String(row.biz_no).includes(biz));
      return rows;
    }

    function maxOf(rows, key) {
      return Math.max(1, ...rows.map((row) => Number(row[key] || 0)));
    }

    function budgetFit(avg, budget) {
      if (!budget || !avg) return 0.55;
      const ratio = avg / budget;
      if (ratio <= 1) return Math.max(0.35, 0.65 + ratio * 0.35);
      return Math.max(0, 1 - (ratio - 1) * 1.25);
    }

    function enrich(rows) {
      const budget = asNumber($("budgetInput").value);
      const presetName = selectedPresetName();
      const maxAward = maxOf(rows, "award_count");
      const maxRecent = maxOf(rows, "recent_award_count");
      const maxAmount = Math.max(1, ...rows.map((row) => Math.log1p(Number(row.award_amount_sum || 0))));
      const maxPreset = Math.max(1, ...rows.map((row) => Number(row.presets?.[presetName]?.score || 0)));
      return rows.map((row) => {
        const awardNorm = Number(row.award_count || 0) / maxAward;
        const recentNorm = Number(row.recent_award_count || 0) / maxRecent;
        const amountNorm = Math.log1p(Number(row.award_amount_sum || 0)) / maxAmount;
        const stability = Math.min(1, awardNorm * 0.38 + recentNorm * 0.28 + amountNorm * 0.18 + (row.manufacturer ? 0.10 : 0) + (row.representative_item ? 0.06 : 0));
        const social = Math.min(1, Number(row.indicator_count || 0) / 7);
        const fit = budgetFit(Number(row.avg_award_amount || 0), budget);
        const validity = (row.cert_valid ? 0.55 : 0.25) + (row.supply_registered_at ? 0.25 : 0) + (row.supply_changed_at ? 0.20 : 0);
        const risk = row.current_sanction ? 1 : row.sanction ? 0.55 : 0;
        const preset = row.presets?.[presetName] || {};
        const presetScore = Number(preset.score || 0);
        const presetRank = Number(preset.rank || 0);
        const aiRank = Number(row.ai_rank || 0);
        return {
          ...row,
          stability_score: stability,
          social_score: social,
          budget_fit: fit,
          validity_score: Math.min(1, validity),
          risk_score: risk,
          preset_name: presetName,
          preset_score: presetScore,
          preset_norm: presetScore / maxPreset,
          preset_rank: presetRank,
          preset_detail: preset.detail || "",
          rank_gap: presetRank && aiRank ? presetRank - aiRank : null,
        };
      }).sort((a, b) => (a.ai_rank || 9999) - (b.ai_rank || 9999) || Number(b.ai_score || 0) - Number(a.ai_score || 0));
    }

    function xmlText(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }

    function renderTopScoreBars(rows) {
      const topRows = rows.slice(0, 5);
      if (!topRows.length) {
        $("scoreBars").innerHTML = `<div class="empty">시각화할 후보가 없습니다.</div>`;
        return;
      }
      $("scoreBars").innerHTML = topRows.map((row, index) => `
        <div class="score-row">
          <div class="score-head">
            <div class="score-name">AI ${row.ai_rank || index + 1}위. ${row.company}</div>
            <div class="score-value">${Number(row.ai_score || 0).toFixed(3)}</div>
          </div>
          <div class="total-bar" title="AI 예측점수"><span class="seg ai" style="display:block;width:${pct(Number(row.ai_score || 0))}"></span></div>
          <div class="stack-bar" title="선택 프리셋 상대점수"><span class="seg preset" style="width:${pct(row.preset_norm)}"></span></div>
          <div class="score-meta">
            <div>프리셋 ${row.preset_rank || "-"}위 · ${row.preset_score ? row.preset_score.toFixed(1) : "-"}점 · <span class="rank-gap">${rankGapText(row)}</span></div>
            <div class="score-metrics">
              <div class="score-metric">안정성<b>${pct(row.stability_score)}</b></div>
              <div class="score-metric">사회책임<b>${pct(row.social_score)}</b></div>
              <div class="score-metric">예산<b>${pct(row.budget_fit)}</b></div>
            </div>
          </div>
        </div>
      `).join("");
    }

    function renderPresetScoreBars(rows) {
      const topRows = rows
        .filter((row) => Number(row.preset_rank || 0) > 0)
        .sort((a, b) => Number(a.preset_rank || 9999) - Number(b.preset_rank || 9999) || (a.ai_rank || 9999) - (b.ai_rank || 9999))
        .slice(0, 5);
      if (!topRows.length) {
        $("presetBars").innerHTML = `<div class="empty">선택 프리셋 순위가 없습니다.</div>`;
        return;
      }
      $("presetBars").innerHTML = topRows.map((row) => `
        <div class="score-row">
          <div class="score-head">
            <div class="score-name">프리셋 ${row.preset_rank}위. ${row.company}</div>
            <div class="score-value">${row.preset_score ? row.preset_score.toFixed(1) : "-"}</div>
          </div>
          <div class="total-bar" title="선택 프리셋 상대점수"><span class="seg preset" style="display:block;width:${pct(row.preset_norm)}"></span></div>
          <div class="stack-bar" title="AI 예측점수"><span class="seg ai" style="width:${pct(Number(row.ai_score || 0))}"></span></div>
          <div class="score-meta">
            <div>AI ${row.ai_rank || "-"}위 · AI ${Number(row.ai_score || 0).toFixed(3)} · <span class="rank-gap">${rankGapText(row)}</span></div>
            <div class="score-metrics">
              <div class="score-metric">안정성<b>${pct(row.stability_score)}</b></div>
              <div class="score-metric">사회책임<b>${pct(row.social_score)}</b></div>
              <div class="score-metric">예산<b>${pct(row.budget_fit)}</b></div>
            </div>
          </div>
        </div>
      `).join("");
    }

    function rankGapText(row) {
      if (row.rank_gap === null || row.rank_gap === undefined) return "순위차 없음";
      if (row.rank_gap === 0) return "AI와 프리셋 동일";
      if (row.rank_gap < 0) return `프리셋이 ${Math.abs(row.rank_gap)}위 높게 봄`;
      return `프리셋이 ${row.rank_gap}위 낮게 봄`;
    }

    function renderVisuals(rows) {
      renderTopScoreBars(rows);
      renderPresetScoreBars(rows);
    }

    function renderRanking(rows) {
      $("rankingNote").textContent = `${itemName($("itemCodeInput").value.trim())} · ${rows.length}개 후보`;
      if (!rows.length) {
        $("rankingBody").innerHTML = `<tr><td class="empty" colspan="11">조건에 맞는 후보가 없습니다.</td></tr>`;
        return;
      }
      $("rankingBody").innerHTML = rows.map((row, index) => {
        const rep = [row.representative_item ? "대표" : "", row.manufacturer ? "제조" : ""].filter(Boolean);
        const risk = row.current_sanction ? ["현재제재"] : row.sanction ? ["제재이력"] : ["정상"];
        const riskCls = row.current_sanction ? "risk" : row.sanction ? "warn" : "good";
        return `<tr>
          <td><span class="rank">${row.ai_rank || index + 1}</span></td>
          <td><b>${row.company}</b><br><span class="small muted">${safeText(row.representative_name, "대표자 미확인")}</span></td>
          <td>${row.biz_no}</td>
          <td>${chips(rep, "good")}</td>
          <td>${shortAddress(row.address)}</td>
          <td class="num">${row.preset_rank ? `${row.preset_rank}위<br><span class="small muted">${row.preset_score.toFixed(1)}점</span>` : "-"}</td>
          <td class="num"><b>${row.indicator_count}</b></td>
          <td>${chips(row.indicator_names, "good")}</td>
          <td class="num">${fmt.format(row.award_count || 0)}건<br><span class="small muted">최근 ${fmt.format(row.recent_award_count || 0)}건</span></td>
          <td class="num">${pct(row.budget_fit)}<div class="meter"><span style="width:${pct(row.budget_fit)}"></span></div></td>
          <td>${chips(risk, riskCls)}</td>
        </tr>`;
      }).join("");
    }

    function renderDetails(rows) {
      if (!rows.length) {
        $("detailBody").innerHTML = `<tr><td class="empty" colspan="10">검증 상세가 없습니다.</td></tr>`;
        return;
      }
      $("detailBody").innerHTML = rows.map((row) => `
        <tr>
          <td class="num">${row.candidate_id}</td>
          <td><b>${row.company}</b><br><span class="small muted">${row.biz_no}</span></td>
          <td class="num">${Number(row.ai_score || 0).toFixed(3)}<br><span class="small muted">HGB ${row.ai_rank || "-"}위</span></td>
          <td class="num" title="${xmlText(row.preset_detail)}">${row.preset_rank ? `${row.preset_rank}위<br><span class="small muted">${row.preset_score.toFixed(1)}점</span>` : "-"}</td>
          <td class="num">${pct(row.stability_score)}</td>
          <td class="num">${pct(row.social_score)}<br><span class="small muted">${row.indicator_count}개</span></td>
          <td>${safeText(row.supply_registered_at)}<br><span class="small muted">${safeText(row.supply_changed_at)}</span></td>
          <td>${row.cert_valid ? '<span class="chip good">Y</span>' : '<span class="chip">미확인</span>'}</td>
          <td>${safeText(row.last_award_date)}</td>
          <td class="num">${row.avg_award_amount ? `${moneyFmt.format(row.avg_award_amount)}원` : "-"}</td>
        </tr>
      `).join("");
    }

    function quantile(values, q) {
      const sorted = values.map(Number).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      if (!sorted.length) return 0;
      const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1));
      return sorted[index];
    }

    function activeSegmentIds() {
      return Array.from(document.querySelectorAll(".segment-check:checked")).map((input) => input.value);
    }

    function segmentDefinitions(rows) {
      const awardCountCut = quantile(rows.map((row) => Number(row.award_count || 0)), 0.8);
      const awardAmountCut = quantile(rows.map((row) => Number(row.award_amount_sum || 0)), 0.8);
      const multiLabels = (row) => {
        const labels = [];
        if (Number(row.indicator_count || 0) > 0) labels.push("사회책임");
        if (Number(row.award_count || 0) >= awardCountCut && Number(row.award_count || 0) > 0) labels.push("낙찰건수");
        if (Number(row.budget_fit || 0) >= 0.8) labels.push("예산적합");
        if (!row.current_sanction && !row.sanction) labels.push("제재없음");
        if (row.manufacturer) labels.push("제조업체");
        return labels;
      };
      rows.forEach((row) => { row.multi_metric_names = multiLabels(row); });
      return [
        {
          id: "multi",
          label: "다중지표 보유",
          desc: "사회책임·낙찰·예산·제재·제조 중 2개 이상",
          rows: rows.filter((row) => row.multi_metric_names.length >= 2),
        },
        {
          id: "awardCount",
          label: "낙찰건수 상위",
          desc: `${fmt.format(awardCountCut)}건 이상`,
          rows: rows.filter((row) => Number(row.award_count || 0) >= awardCountCut && Number(row.award_count || 0) > 0),
        },
        {
          id: "awardAmount",
          label: "낙찰금액 상위",
          desc: `${moneyFmt.format(awardAmountCut)}원 이상`,
          rows: rows.filter((row) => Number(row.award_amount_sum || 0) >= awardAmountCut && Number(row.award_amount_sum || 0) > 0),
        },
        {
          id: "sanction",
          label: "부정당제재",
          desc: "제재 이력 또는 현재 제재",
          rows: rows.filter((row) => row.current_sanction || row.sanction),
        },
        {
          id: "budgetFit",
          label: "예산 적합",
          desc: "추정금액 기준 80점 이상",
          rows: rows.filter((row) => Number(row.budget_fit || 0) >= 0.8),
        },
      ];
    }

    function segmentAvg(rows, key) {
      if (!rows.length) return 0;
      return rows.reduce((sum, row) => sum + Number(row[key] || 0), 0) / rows.length;
    }

    function renderSegments(rows) {
      const active = new Set(activeSegmentIds());
      const definitions = segmentDefinitions(rows).filter((segment) => active.has(segment.id));
      if (!definitions.length) {
        $("segmentSummary").innerHTML = `<div class="empty">확인할 지표를 선택하세요.</div>`;
        $("segmentBody").innerHTML = `<tr><td class="empty" colspan="8">선택된 후보군이 없습니다.</td></tr>`;
        return;
      }

      $("segmentSummary").innerHTML = definitions.map((segment) => {
        const topNames = segment.rows.slice(0, 3).map((row) => row.company).join(" · ") || "-";
        return `<div class="segment-card">
          <div class="segment-title"><span>${segment.label}</span><span class="segment-count">${fmt.format(segment.rows.length)}개</span></div>
          <div class="segment-desc">${segment.desc}</div>
          <div class="segment-stats">
            <div class="segment-stat"><span>AI</span><div class="bar-bg"><div class="bar" style="width:${pct(segmentAvg(segment.rows, "ai_score"))}"></div></div><span>${pct(segmentAvg(segment.rows, "ai_score"))}</span></div>
            <div class="segment-stat"><span>프리셋</span><div class="bar-bg"><div class="bar" style="width:${pct(segmentAvg(segment.rows, "preset_norm"))}"></div></div><span>${pct(segmentAvg(segment.rows, "preset_norm"))}</span></div>
          </div>
          <div class="small muted" style="margin-top:8px">${topNames}</div>
        </div>`;
      }).join("");

      const byCandidate = new Map();
      definitions.forEach((segment) => {
        segment.rows.forEach((row) => {
          const current = byCandidate.get(row.candidate_id) || { row, labels: [] };
          current.labels.push(segment.label);
          byCandidate.set(row.candidate_id, current);
        });
      });
      const segmentRows = Array.from(byCandidate.values())
        .sort((a, b) => (a.row.ai_rank || 9999) - (b.row.ai_rank || 9999) || Number(b.row.preset_score || 0) - Number(a.row.preset_score || 0));

      if (!segmentRows.length) {
        $("segmentBody").innerHTML = `<tr><td class="empty" colspan="8">선택 지표에 맞는 후보가 없습니다.</td></tr>`;
        return;
      }

      $("segmentBody").innerHTML = segmentRows.map(({ row, labels }) => {
        const risk = row.current_sanction ? ["현재제재"] : row.sanction ? ["제재이력"] : ["정상"];
        const riskCls = row.current_sanction ? "risk" : row.sanction ? "warn" : "good";
        const multiReason = row.multi_metric_names?.length ? `<br><span class="small muted">${row.multi_metric_names.join(" · ")}</span>` : "";
        return `<tr>
          <td>${chips(labels)}${multiReason}</td>
          <td><b>${row.company}</b><br><span class="small muted">${row.biz_no}</span></td>
          <td class="num">${Number(row.ai_score || 0).toFixed(3)}<br><span class="small muted">${row.ai_rank || "-"}위</span></td>
          <td class="num">${row.preset_rank ? `${row.preset_rank}위<br><span class="small muted">${row.preset_score.toFixed(1)}점</span>` : "-"}</td>
          <td>${chips(row.indicator_names, "good")}</td>
          <td class="num">${fmt.format(row.award_count || 0)}건<br><span class="small muted">최근 ${fmt.format(row.recent_award_count || 0)}건</span></td>
          <td class="num">${row.award_amount_sum ? `${moneyFmt.format(row.award_amount_sum)}원` : "-"}</td>
          <td>${chips(risk, riskCls)}</td>
        </tr>`;
      }).join("");
    }

    function pipelineStatusText() {
      const status = DATA.pipeline_status || {};
      if (!status.status) return "수동 생성";
      const labels = { success: "완료", failed: "실패", running: "실행 중", dry_run: "점검 완료" };
      const label = labels[status.status] || status.status;
      const trigger = status.trigger === "scheduled" ? "예약 갱신" : "수동 갱신";
      const when = status.finished_at || status.started_at || "";
      const schedule = status.schedule_label ? ` · ${status.schedule_label}` : "";
      return `${trigger} ${label}${when ? ` · ${when}` : ""}${schedule}`;
    }

    function render() {
      const rows = enrich(currentBaseRows());
      renderVisuals(rows);
      renderRanking(rows);
      renderDetails(rows);
      renderSegments(rows);
    }

    function init() {
      $("generatedAt").textContent = DATA.generated_at;
      $("sourceUpdatedAt").textContent = DATA.source_updated_at;
      $("pipelineStatus").textContent = pipelineStatusText();
      $("itemCodeList").innerHTML = DATA.items.map((item) => `<option value="${item.code}">${item.name}</option>`).join("");
      $("presetSelect").innerHTML = (DATA.presets || ["균형형"]).map((preset) => `<option value="${xmlText(preset)}">${xmlText(preset)}</option>`).join("");
      $("presetSelect").value = (DATA.presets || []).includes("균형형") ? "균형형" : DATA.presets?.[0] || "균형형";
      $("itemCodeInput").value = DATA.items[0]?.code || "";
      ["itemCodeInput", "budgetInput", "bizInput", "presetSelect"].forEach((id) => $(id).addEventListener("input", render));
      $("presetSelect").addEventListener("change", render);
      document.querySelectorAll(".segment-check").forEach((input) => input.addEventListener("change", render));
      render();
    }

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    print(f"dashboard written: {DASHBOARD_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
