#!/usr/bin/env python
"""
Create human-facing Korean-column datasets from raw API/source files.

The raw files keep provider/API column names for reproducibility. This script
creates analysis-ready CSV files whose file names and column names follow the
Korean dataset names used in the task PDF.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_raw_data"
OUT = ROOT / "02_processed" / "한국어_표준데이터셋"
AWARDS_DIR = RAW / "나라장터_낙찰정보서비스"
SHARED_YEARS = {str(year) for year in range(2022, 2026)}


def is_shared_year_file(path: Path) -> bool:
    # Keep the shared submission dataset focused on the 2022-2025 analysis window.
    years = set(re.findall(r"20\d{2}", str(path)))
    return not years or years <= SHARED_YEARS


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def read_many_csvs(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            if "manifest" in path.name.lower():
                continue
            rows.extend(read_csv_rows(path))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pick(row: dict[str, str], *candidates: str) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value not in (None, ""):
            return value
    return ""


def normalize(rows: list[dict[str, str]], mapping: list[tuple[str, tuple[str, ...]]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        # Each Korean column can accept several provider/API aliases.
        result.append({korean: pick(row, *sources) for korean, sources in mapping})
    return result


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    award_files = [
        path
        for path in sorted(AWARDS_DIR.glob("20??/[0-1][0-9]/나라장터_낙찰정보서비스_물품_*.csv"))
        if is_shared_year_file(path)
    ]
    awards = read_many_csvs(award_files)
    award_mapping = [
        ("입찰공고번호", ("bidNtceNo",)),
        ("입찰공고차수", ("bidNtceOrd",)),
        ("입찰분류번호", ("bidClsfcNo",)),
        ("재입찰번호", ("rbidNo",)),
        ("입찰공고명", ("bidNtceNm",)),
        ("참가업체수", ("prtcptCnum",)),
        ("낙찰업체명", ("bidwinnrNm",)),
        ("사업자번호", ("bidwinnrBizno", "business_no")),
        ("대표자명", ("bidwinnrCeoNm",)),
        ("업체주소", ("bidwinnrAdrs",)),
        ("업체전화번호", ("bidwinnrTelNo",)),
        ("낙찰금액", ("sucsfbidAmt", "award_amount")),
        ("낙찰률", ("sucsfbidRate",)),
        ("개찰일시", ("rlOpengDt",)),
        ("수요기관코드", ("dminsttCd",)),
        ("수요기관명", ("dminsttNm",)),
        ("등록일시", ("rgstDt",)),
        ("최종낙찰일자", ("fnlSucsfDate", "award_date")),
        ("최종낙찰업체담당자", ("fnlSucsfCorpOfcl",)),
        ("연계기관명", ("linkInsttNm",)),
    ]
    write_csv(
        OUT / "조달청_나라장터_낙찰정보서비스_물품.csv",
        normalize(awards, award_mapping),
        [name for name, _ in award_mapping],
    )

    user_base = RAW / "나라장터_사용자정보서비스"
    basic = read_many_csvs((user_base / "2. 조달업체기본정보조회").glob("*.csv"))
    basic_mapping = [
        ("사업자번호", ("bizno",)),
        ("업체명", ("corpNm",)),
        ("영문업체명", ("engCorpNm",)),
        ("개업일자", ("opbizDt",)),
        ("지역코드", ("rgnCd",)),
        ("지역명", ("rgnNm",)),
        ("우편번호", ("zip",)),
        ("주소", ("adrs",)),
        ("상세주소", ("dtlAdrs",)),
        ("전화번호", ("telNo",)),
        ("팩스번호", ("faxNo",)),
        ("국가명", ("cntryNm",)),
        ("홈페이지주소", ("hmpgAdrs",)),
        ("제조구분명", ("mnfctDivNm",)),
        ("종업원수", ("emplyeNum",)),
        ("업체업무구분명", ("corpBsnsDivNm",)),
        ("본지사구분명", ("hdoffceDivNm",)),
        ("등록일시", ("rgstDt",)),
        ("변경일시", ("chgDt",)),
        ("대표자명", ("ceoNm",)),
    ]
    write_csv(
        OUT / "조달청_나라장터_사용자정보서비스_조달업체기본정보조회.csv",
        normalize(basic, basic_mapping),
        [name for name, _ in basic_mapping],
    )

    industries = read_many_csvs((user_base / "3. 조달업체업종정보조회").glob("*.csv"))
    industry_mapping = [
        ("사업자번호", ("bizno",)),
        ("업종명", ("indstrytyNm",)),
        ("업종코드", ("indstrytyCd",)),
        ("등록일시", ("rgstDt",)),
        ("유효기간만료일", ("vldPrdExprtDt",)),
        ("업종상태명", ("indstrytyStatsNm",)),
        ("대표업종여부", ("rprsntIndstrytyYn",)),
        ("시스템등록일시", ("systmRgstDt",)),
        ("시스템변경일시", ("systmChgDt",)),
    ]
    write_csv(
        OUT / "조달청_나라장터_사용자정보서비스_조달업체업종정보조회.csv",
        normalize(industries, industry_mapping),
        [name for name, _ in industry_mapping],
    )

    supply_files = [
        path
        for path in sorted((user_base / "4. 조달업체공급물품정보조회").rglob("*.csv"))
        if is_shared_year_file(path)
    ]
    # Supply records define which suppliers can be candidates for each item.
    supplies = read_many_csvs(supply_files)
    supply_mapping = [
        ("사업자번호", ("bizno",)),
        ("세부품명", ("dtilPrdctClsfcNoNm",)),
        ("세부품명번호", ("dtilPrdctClsfcNo",)),
        ("등록일시", ("rgstDt",)),
        ("변경일시", ("chgDt",)),
        ("대표물품여부", ("rprsntPrdctClsfcNoNmYn",)),
        ("제조여부", ("mnfctYn",)),
    ]
    write_csv(
        OUT / "조달청_나라장터_사용자정보서비스_조달업체공급물품정보조회.csv",
        normalize(supplies, supply_mapping),
        [name for name, _ in supply_mapping],
    )

    sanctions = read_many_csvs((user_base / "5. 부정당제재업체정보조회").glob("*.csv"))
    sanction_mapping = [
        ("제재문서명", ("unptRsttDocNm",)),
        ("사업자번호", ("bizno",)),
        ("업체명", ("corpNm",)),
        ("제재시작일", ("rsttBgnDate",)),
        ("제재종료일", ("rsttEndDate",)),
        ("기관코드", ("insttCd",)),
        ("기관명", ("insttNm",)),
        ("법령명", ("lawordNm",)),
        ("집행사유명", ("enfcPrvNm",)),
        ("통보일자", ("ntfcnDt",)),
        ("제재진행명", ("rsttProgrsNm",)),
    ]
    write_csv(
        OUT / "조달청_나라장터_사용자정보서비스_부정당제재업체정보조회.csv",
        normalize(sanctions, sanction_mapping),
        [name for name, _ in sanction_mapping],
    )

    startup = read_csv_rows(RAW / "경진대회_공공데이터" / "창업진흥원_창업기업확인서발급기업정보_조회서비스.csv")
    startup_mapping = [
        ("사업자번호", ("brno",)),
        ("확인서만료일자", ("confmdoc_expr_dt",)),
        ("확인서발급일자", ("confmdoc_isu_dt",)),
        ("확인서발급번호", ("confmdoc_isu_no",)),
        ("법인등록번호", ("crno",)),
        ("기업명", ("ntrp_nm",)),
        ("기업유형명", ("ntrp_type_nm",)),
        ("대표자명", ("repr_nm",)),
        ("공동대표자명", ("unin_repr_nm",)),
    ]
    write_csv(
        OUT / "창업진흥원_창업기업확인서발급기업정보_조회서비스.csv",
        normalize(startup, startup_mapping),
        [name for name, _ in startup_mapping],
    )

    disabled = read_csv_rows(RAW / "경진대회_공공데이터" / "한국장애인고용공단_장애인 표준사업장 실시간 조회.csv")
    disabled_mapping = [
        ("주소", ("address",)),
        ("인증일자", ("authDate",)),
        ("인증ID", ("compAuthId",)),
        ("사업자번호", ("compBizNo",)),
        ("사업체명", ("compName",)),
        ("인증번호", ("compRegNo",)),
        ("전화번호", ("compTel",)),
        ("사업체유형명", ("compTypeNm",)),
        ("대표자명", ("presidentName",)),
        ("생산품", ("product",)),
        ("순번", ("rnum",)),
        ("관리번호", ("compMgrNo",)),
    ]
    write_csv(
        OUT / "한국장애인고용공단_장애인_표준사업장_실시간_조회.csv",
        normalize(disabled, disabled_mapping),
        [name for name, _ in disabled_mapping],
    )

    cert = read_csv_rows(RAW / "경진대회_공공데이터" / "주식회사한국중소벤처기업유통원_공공구매종합정보망 인증서 정보 제공 서비스.csv")
    cert_mapping = [
        ("인증구분코드", ("certSeCode",)),
        ("유효기간시작일", ("validPdBeginDe",)),
        ("유효기간종료일", ("validPdEndDe",)),
        ("인증일자", ("certfcDe",)),
        ("세부품명번호", ("detailPrdnmNo",)),
        ("조회사업자번호", ("searched_bsnmNo",)),
    ]
    write_csv(
        OUT / "주식회사한국중소벤처기업유통원_공공구매종합정보망_인증서_정보_제공_서비스.csv",
        normalize(cert, cert_mapping),
        [name for name, _ in cert_mapping],
    )

    copy_csv(
        RAW / "사회적책임_외부데이터" / "사회적기업_조회_20260428.csv",
        OUT / "사회적기업_조회.csv",
    )

    coops = read_csv_rows(RAW / "사회적책임_외부데이터" / "한국사회적기업진흥원_사회적협동조합_설립현황_20250414.csv")
    social_coops = [
        row for row in coops
        if "사회적" in (row.get("구분") or "") or "사회적협동조합" in (row.get("협동조합명") or "")
    ]
    coop_columns = ["구분", "협동조합명", "대표자", "업종", "유형", "소재지", "수리인가일", "주소"]
    write_csv(
        OUT / "한국사회적기업진흥원_사회적협동조합_설립현황.csv",
        [{column: row.get(column, "") for column in coop_columns} for row in social_coops],
        coop_columns,
    )

    copy_csv(
        RAW / "사회적책임_외부데이터" / "국가보훈부_자활용사촌_복지공장_생산품목_20250502.csv",
        OUT / "국가보훈부_자활용사촌_복지공장_생산품목.csv",
    )

    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
