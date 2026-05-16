#!/usr/bin/env python
"""Build official 150-row feature input files from candidate list, teammate files, and APIs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "02_processed"
RAW = ROOT / "01_raw_data"
FEATURE_DIR = PROCESSED / "feature_inputs"
CANDIDATES = PROCESSED / "후보업체_목록.csv"
YEONWOO_SOURCE = ROOT / "kec_김연우" / "연우_낙찰위험_features.csv"
AWARD_SOURCE = PROCESSED / "한국어_표준데이터셋" / "조달청_나라장터_낙찰정보서비스_물품.csv"
SANCTION_SOURCE = PROCESSED / "한국어_표준데이터셋" / "조달청_나라장터_사용자정보서비스_부정당제재업체정보조회.csv"
AWARD_START_YEAR = 2022
AWARD_END_YEAR = 2025
RECENT_AWARD_YEAR = 2025

STARTUP_URL = "http://apis.data.go.kr/B552735/kisedCertService/getCorporateInformation"
DISABLED_URL = "http://apis.data.go.kr/B552583/comp/comp_auth"
SMPP_DIRECT_URL = "http://apis.data.go.kr/B550598/smppCertInfo/getDPrductList"

BASE_COLUMNS = ["candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명", "대표물품", "제조업체"]


def norm_digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def norm_name(value: str | None) -> str:
    text = (value or "").lower()
    for token in [
        "주식회사",
        "(주)",
        "㈜",
        "유한회사",
        "합자회사",
        "합명회사",
        "사단법인",
        "재단법인",
        "사회적협동조합",
        "협동조합",
        "보호작업장",
        "사업단",
        " ",
    ]:
        text = text.replace(token, "")
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def hyphen_bizno(value: str) -> str:
    digits = norm_digits(value)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return value


def parse_date(value: str | None) -> dt.date | None:
    text = (value or "")[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def parse_number(value: Any) -> float:
    text = re.sub(r"[^0-9.\-]", "", str(value or "").replace(",", ""))
    if not text or text in {"-", ".", "-."}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def first_value(row: dict[str, str], columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_url(url: str, params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        # Public data service keys can already contain URL escapes, so avoid double-encoding them.
        if key in {"serviceKey", "ServiceKey"}:
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{urllib.parse.quote_plus(key)}={urllib.parse.quote_plus(str(value))}")
    return f"{url}?{'&'.join(parts)}"


def fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "kec-feature-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def xml_text(root: ET.Element, tag: str) -> str:
    found = root.find(f".//{tag}")
    return (found.text or "").strip() if found is not None else ""


def xml_items(root: ET.Element) -> list[dict[str, str]]:
    result = []
    for item in root.findall(".//item"):
        result.append({child.tag: (child.text or "").strip() for child in list(item)})
    return result


def query_startup(service_key: str, bizno: str) -> dict[str, Any]:
    url = build_url(
        STARTUP_URL,
        {
            "serviceKey": service_key,
            "page": 1,
            "perPage": 10,
            "returnType": "json",
            "cond[brno::EQ]": bizno,
        },
    )
    data = parse_json(fetch_text(url))
    rows = data.get("data") if isinstance(data, dict) else []
    exact_rows = [row for row in rows or [] if norm_digits(str(row.get("brno", ""))) == bizno]
    return {"matched": bool(exact_rows), "rows": exact_rows, "raw_count": len(rows or []), "raw": data}


def query_disabled(service_key: str, bizno: str) -> dict[str, Any]:
    url = build_url(
        DISABLED_URL,
        {
            "serviceKey": service_key,
            "pageNo": 1,
            "numOfRows": 10,
            "comp_biz_no": hyphen_bizno(bizno),
        },
    )
    text = fetch_text(url)
    root = ET.fromstring(text)
    items = xml_items(root)
    exact_items = []
    for item in items:
        values = " ".join(item.values())
        if bizno in norm_digits(values):
            exact_items.append(item)
    return {"matched": bool(exact_items), "rows": exact_items, "raw_count": len(items), "raw": text[:1000]}


def query_smpp_direct(service_key: str, bizno: str, item_code: str, date_yyyymmdd: str) -> dict[str, Any]:
    url = build_url(
        SMPP_DIRECT_URL,
        {
            "ServiceKey": service_key,
            "stdrDate": date_yyyymmdd,
            "bsnmNo": bizno,
            "detailPrdnmNo": item_code,
            "numOfRows": 10,
            "pageNo": 1,
        },
    )
    text = fetch_text(url)
    root = ET.fromstring(text)
    items = xml_items(root)
    result_code = xml_text(root, "resultCode")
    exact_items = []
    for item in items:
        item_digits = norm_digits(" ".join(item.values()))
        if item_code in item_digits:
            exact_items.append(item)
    matched = result_code == "00" and bool(exact_items)
    return {"matched": matched, "rows": exact_items, "raw_count": len(items), "result_code": result_code, "raw": text[:1000]}


def load_local_policy_sets() -> dict[str, set[str]]:
    startup = {
        norm_digits(row.get("사업자번호") or row.get("brno"))
        for row in read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "창업진흥원_창업기업확인서발급기업정보_조회서비스.csv")
    }
    disabled = {
        norm_digits(row.get("사업자번호") or row.get("compBizNo"))
        for row in read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "한국장애인고용공단_장애인_표준사업장_실시간_조회.csv")
    }
    public_purchase = {
        norm_digits(row.get("조회사업자번호") or row.get("searched_bsnmNo"))
        for row in read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "주식회사한국중소벤처기업유통원_공공구매종합정보망_인증서_정보_제공_서비스.csv")
    }
    return {
        "startup": {value for value in startup if value},
        "disabled": {value for value in disabled if value},
        "public_purchase": {value for value in public_purchase if value},
    }


def load_social_sources() -> dict[str, Any]:
    # Social-value sources use mixed identifiers; keep both business numbers and conservative name keys.
    social_enterprise_rows = read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "사회적기업_조회.csv")
    social_enterprise_biz = {norm_digits(row.get("사업자등록번호")) for row in social_enterprise_rows}
    social_enterprise_names = {norm_name(row.get("기업명")) for row in social_enterprise_rows if norm_name(row.get("기업명"))}

    coop_rows = read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "한국사회적기업진흥원_사회적협동조합_설립현황.csv")
    social_coop_names = {
        norm_name(row.get("협동조합명"))
        for row in coop_rows
        if (row.get("구분") or "").strip() == "사회적협동조합" and norm_name(row.get("협동조합명"))
    }

    veteran_rows = read_csv_rows(PROCESSED / "한국어_표준데이터셋" / "국가보훈부_자활용사촌_복지공장_생산품목.csv")
    veteran_names = {norm_name(row.get("복지공장(법인)명칭")) for row in veteran_rows if norm_name(row.get("복지공장(법인)명칭"))}
    veteran_item_pairs = {
        (norm_name(row.get("복지공장(법인)명칭")), norm_digits(row.get("물품목록번호(G2B분류번호)")))
        for row in veteran_rows
    }

    return {
        "social_enterprise_biz": {value for value in social_enterprise_biz if value},
        "social_enterprise_names": social_enterprise_names,
        "social_coop_names": social_coop_names,
        "veteran_names": veteran_names,
        "veteran_item_pairs": veteran_item_pairs,
    }


def load_award_stats(target_biznos: set[str]) -> dict[str, dict[str, Any]]:
    # Recompute award/risk features from source data for the current candidate list.
    # This avoids stale candidate_id matches when new items are appended.
    stats = {
        bizno: {
            "count": 0,
            "recent_count": 0,
            "amount_sum": 0.0,
            "last_award_date": "",
        }
        for bizno in target_biznos
    }
    if not AWARD_SOURCE.exists():
        return stats

    for row in read_csv_rows(AWARD_SOURCE):
        bizno = norm_digits(first_value(row, ["사업자번호", "낙찰업체사업자번호", "bidwinnrBizno"]))
        if bizno not in stats:
            continue
        date = parse_date(first_value(row, ["최종낙찰일자", "개찰일시", "등록일시", "fnlSucsfDate", "rlOpengDt"]))
        if not date or not (AWARD_START_YEAR <= date.year <= AWARD_END_YEAR):
            continue
        amount = parse_number(first_value(row, ["낙찰금액", "sucsfbidAmt"]))
        stat = stats[bizno]
        stat["count"] += 1
        stat["amount_sum"] += amount
        if date.year >= RECENT_AWARD_YEAR:
            stat["recent_count"] += 1
        date_text = date.isoformat()
        if date_text > stat["last_award_date"]:
            stat["last_award_date"] = date_text

    for stat in stats.values():
        count = int(stat["count"])
        stat["avg_amount"] = round(stat["amount_sum"] / count) if count else 0
    return stats


def load_sanction_stats(target_biznos: set[str], today: dt.date) -> dict[str, dict[str, Any]]:
    # Sanction status is evaluated as of today's run, while history is kept as a risk indicator.
    stats = {
        bizno: {
            "sanction": False,
            "count": 0,
            "current": False,
            "end_date": "",
        }
        for bizno in target_biznos
    }
    if not SANCTION_SOURCE.exists():
        return stats

    for row in read_csv_rows(SANCTION_SOURCE):
        bizno = norm_digits(first_value(row, ["사업자번호", "사업자등록번호", "업체사업자번호"]))
        if bizno not in stats:
            continue
        start = parse_date(row.get("제재시작일"))
        end = parse_date(row.get("제재종료일"))
        stat = stats[bizno]
        stat["sanction"] = True
        stat["count"] += 1
        if start and start <= today and (end is None or today <= end):
            stat["current"] = True
        if end and end.isoformat() > stat["end_date"]:
            stat["end_date"] = end.isoformat()
    return stats


def sync_yeonwoo(candidates: list[dict[str, str]]) -> None:
    source_rows = read_csv_rows(YEONWOO_SOURCE) if YEONWOO_SOURCE.exists() else []
    # Old teammate files are fallback evidence; source-derived stats below take priority.
    by_pair = {
        (norm_digits(row.get("물품번호")), norm_digits(row.get("사업자번호_정규화") or row.get("사업자번호"))): row
        for row in source_rows
    }
    by_biz = {
        norm_digits(row.get("사업자번호_정규화") or row.get("사업자번호")): row
        for row in source_rows
    }
    target_biznos = {candidate["사업자번호_정규화"] for candidate in candidates if candidate.get("사업자번호_정규화")}
    award_stats = load_award_stats(target_biznos)
    sanction_stats = load_sanction_stats(target_biznos, dt.date.today())
    extra_columns = [
        "낙찰건수",
        "최근낙찰건수",
        "최근낙찰일",
        "평균낙찰금액",
        "낙찰금액합계",
        "부정당제재",
        "제재횟수",
        "현재제재여부",
        "제재종료일",
        "매칭방식",
        "비고",
    ]
    rows = []
    for candidate in candidates:
        bizno = candidate["사업자번호_정규화"]
        item_code = candidate["물품번호"]
        old = by_pair.get((norm_digits(item_code), bizno), by_biz.get(bizno, {}))
        award = award_stats.get(bizno, {})
        sanction = sanction_stats.get(bizno, {})
        row = {column: candidate.get(column, "") for column in BASE_COLUMNS}
        count = int(award.get("count", 0) or 0)
        amount_sum = float(award.get("amount_sum", 0.0) or 0.0)
        sanction_count = int(sanction.get("count", 0) or 0)
        row.update({
            "낙찰건수": str(count or old.get("낙찰건수", "")),
            "최근낙찰건수": str(int(award.get("recent_count", 0) or 0) or old.get("최근낙찰건수", "")),
            "최근낙찰일": award.get("last_award_date") or old.get("최근낙찰일", ""),
            "평균낙찰금액": str(int(award.get("avg_amount", 0) or 0) or old.get("평균낙찰금액", "")),
            "낙찰금액합계": str(round(amount_sum, 2) if amount_sum else old.get("낙찰금액합계", "")),
            "부정당제재": "Y" if sanction.get("sanction") else old.get("부정당제재", ""),
            "제재횟수": str(sanction_count or old.get("제재횟수", "")),
            "현재제재여부": "Y" if sanction.get("current") else old.get("현재제재여부", ""),
            "제재종료일": sanction.get("end_date") or old.get("제재종료일", ""),
            "매칭방식": "원천_사업자번호집계" if count or sanction_count else old.get("매칭방식", "원천조회_미매칭"),
            "비고": "" if count or sanction_count else old.get("비고", "낙찰/제재 원천 매칭 없음"),
        })
        rows.append(row)
    write_csv(FEATURE_DIR / "연우_낙찰위험_features.csv", rows, BASE_COLUMNS + extra_columns)


def build_policy(candidates: list[dict[str, str]], api_results: dict[str, dict[str, Any]]) -> None:
    local = load_local_policy_sets()
    columns = BASE_COLUMNS + ["창업기업", "장애인표준사업장", "공공구매인증서", "인증개수", "인증유효여부", "매칭방식", "비고"]
    rows = []
    for candidate in candidates:
        bizno = candidate["사업자번호_정규화"]
        item_code = candidate["물품번호"]
        startup_match = bizno in local["startup"] or api_results.get("startup", {}).get(bizno, {}).get("matched", False)
        disabled_match = bizno in local["disabled"] or api_results.get("disabled", {}).get(bizno, {}).get("matched", False)
        public_match = bizno in local["public_purchase"] or api_results.get("smpp", {}).get(f"{bizno}:{item_code}", {}).get("matched", False)
        count = sum([startup_match, disabled_match, public_match])
        row = {column: candidate.get(column, "") for column in BASE_COLUMNS}
        row.update({
            "창업기업": "Y" if startup_match else "",
            "장애인표준사업장": "Y" if disabled_match else "",
            "공공구매인증서": "Y" if public_match else "",
            "인증개수": str(count) if count else "",
            "인증유효여부": "Y" if count else "",
            "매칭방식": "API/원천_사업자번호매칭" if count else "API/원천조회_미매칭",
            "비고": "" if count else "현재 후보 목록 기준 정책인증 확정 매칭 없음",
        })
        rows.append(row)
    write_csv(FEATURE_DIR / "연수_정책인증_features.csv", rows, columns)


def build_social(candidates: list[dict[str, str]]) -> None:
    sources = load_social_sources()
    columns = BASE_COLUMNS + ["사회적기업", "사회적협동조합", "자활용사촌복지공장", "생산품목매칭", "매칭방식", "비고"]
    rows = []
    for candidate in candidates:
        bizno = candidate["사업자번호_정규화"]
        item_code = candidate["물품번호"]
        name = candidate["업체명"]
        normalized_name = norm_name(name)
        social_enterprise = bizno in sources["social_enterprise_biz"]
        social_coop = False
        veteran = False
        product_match = False
        notes = []

        # Name-only matches are noted but not scored unless the legal form is explicit enough.
        if not social_enterprise and normalized_name in sources["social_enterprise_names"]:
            notes.append("사회적기업 이름 유사 후보 있음; 사업자번호 불일치로 점수 미반영")
        if normalized_name in sources["social_coop_names"]:
            if "사회적협동조합" in name:
                social_coop = True
            else:
                notes.append("사회적협동조합 이름 유사 후보 있음; 사업자번호 부재로 점수 미반영")
        if normalized_name in sources["veteran_names"]:
            veteran = True
            product_match = (normalized_name, item_code) in sources["veteran_item_pairs"]

        count = sum([social_enterprise, social_coop, veteran, product_match])
        row = {column: candidate.get(column, "") for column in BASE_COLUMNS}
        row.update({
            "사회적기업": "Y" if social_enterprise else "",
            "사회적협동조합": "Y" if social_coop else "",
            "자활용사촌복지공장": "Y" if veteran else "",
            "생산품목매칭": "Y" if product_match else "",
            "매칭방식": "사업자번호/보수적업체명매칭" if count else "원천조회_미매칭",
            "비고": "; ".join(notes) if notes else ("" if count else "현재 후보 목록 기준 사회적가치 확정 매칭 없음"),
        })
        rows.append(row)
    write_csv(FEATURE_DIR / "준우_사회적가치_features.csv", rows, columns)


def collect_api_results(
    candidates: list[dict[str, str]],
    service_key: str,
    today: str,
    use_smpp: bool,
    api_delay: float,
    smpp_delay: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {"startup": {}, "disabled": {}, "smpp": {}}
    unique_biz = sorted({candidate["사업자번호_정규화"] for candidate in candidates})
    for index, bizno in enumerate(unique_biz, 1):
        try:
            result["startup"][bizno] = query_startup(service_key, bizno)
        except Exception as exc:  # noqa: BLE001
            result["startup"][bizno] = {"matched": False, "error": str(exc)}
        try:
            result["disabled"][bizno] = query_disabled(service_key, bizno)
        except Exception as exc:  # noqa: BLE001
            result["disabled"][bizno] = {"matched": False, "error": str(exc)}
        if index % 10 == 0:
            print(f"API 조회 진행: 사업자번호 {index}/{len(unique_biz)}")
        time.sleep(api_delay)

    if use_smpp:
        for index, candidate in enumerate(candidates, 1):
            bizno = candidate["사업자번호_정규화"]
            item_code = candidate["물품번호"]
            key = f"{bizno}:{item_code}"
            try:
                result["smpp"][key] = query_smpp_direct(service_key, bizno, item_code, today)
            except Exception as exc:  # noqa: BLE001
                result["smpp"][key] = {"matched": False, "error": str(exc)}
                if "429" in str(exc):
                    time.sleep(max(smpp_delay * 5, 5.0))
            if index % 10 == 0:
                print(f"공공구매 API 조회 진행: 후보 {index}/{len(candidates)}")
            time.sleep(smpp_delay)
    return result


def write_api_summary(api_results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for group, results in api_results.items():
        matched = sum(1 for row in results.values() if row.get("matched"))
        errored = sum(1 for row in results.values() if row.get("error"))
        rows.append({"구분": group, "조회건수": len(results), "매칭건수": matched, "오류건수": errored})
    write_csv(PROCESSED / "feature_inputs" / "API조회_요약.csv", rows, ["구분", "조회건수", "매칭건수", "오류건수"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official feature input CSVs for the fixed 150 candidates.")
    parser.add_argument("--use-api", action="store_true", help="Query public APIs using DATA_GO_KR_SERVICE_KEY")
    parser.add_argument("--use-smpp", action="store_true", help="Also query SMPP direct production API; may consume more quota")
    parser.add_argument("--today", default="20260508", help="Reference date for public purchase API, YYYYMMDD")
    parser.add_argument("--api-delay", type=float, default=0.1, help="Delay seconds between startup/disabled API calls")
    parser.add_argument("--smpp-delay", type=float, default=1.0, help="Delay seconds between SMPP public-purchase API calls")
    args = parser.parse_args()

    candidates = read_csv_rows(CANDIDATES)
    api_results: dict[str, dict[str, Any]] = {"startup": {}, "disabled": {}, "smpp": {}}
    if args.use_api:
        service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
        if not service_key:
            raise SystemExit("DATA_GO_KR_SERVICE_KEY 환경변수가 필요합니다.")
        api_results = collect_api_results(candidates, service_key, args.today, args.use_smpp, args.api_delay, args.smpp_delay)
        write_api_summary(api_results)

    sync_yeonwoo(candidates)
    build_policy(candidates, api_results)
    build_social(candidates)
    print("official feature inputs saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
