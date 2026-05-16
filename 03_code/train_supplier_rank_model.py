#!/usr/bin/env python
"""Train a Random Forest supplier-ranking model from G2B item/supply data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import pickle
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    RandomForestClassifier = None
    joblib = None


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "02_processed"
STANDARD = PROCESSED / "한국어_표준데이터셋"
OUTPUT_DIR = PROCESSED / "model_outputs"
SUPPLY_PATH = STANDARD / "조달청_나라장터_사용자정보서비스_조달업체공급물품정보조회.csv"
AWARD_PATH = STANDARD / "조달청_나라장터_낙찰정보서비스_물품.csv"
CANDIDATES_PATH = PROCESSED / "후보업체_목록.csv"

# Features are computed at the award event date so future records do not leak
# into the train/test ranking task.
FEATURE_COLUMNS = [
    "대표물품",
    "제조업체",
    "공급품목등록후_경과일수",
    "공급품목등록후_경과일수_log",
    "업체_등록품목수",
    "업체_동일대분류등록수",
    "업체_동일분류공급품목수",
    "업체_동일중분류등록수",
    "업체_동일소분류등록수",
    "업체_대표물품등록수",
    "업체_제조품목등록수",
    "품목_공급업체수",
    "품목_대표물품업체수",
    "품목_제조업체수",
    "품목_제조업체비율",
    "품목_대표물품비율",
    "품목_평균참가업체수",
    "품목_평균참가업체수_log",
    "품목_경쟁강도",
    "이전전체낙찰건수",
    "이전동일품목낙찰건수",
    "해당물품_과거낙찰건수",
    "이전동일대분류낙찰건수",
    "최근365일낙찰건수",
    "최근낙찰일_경과일수",
    "최근낙찰일_경과일수_log",
    "이전낙찰금액합계_log",
    "참가업체수_log",
    "물품번호_prefix2",
    "물품번호_prefix4",
    "물품번호_prefix6",
    "창업기업",
    "장애인표준사업장",
    "사회적기업",
    "부정당제재",
    "제재종료후_경과일수",
    "제재종료후_경과일수_log",
]

BASE_OUTPUT_COLUMNS = [
    "dataset",
    "event_id",
    "입찰공고번호",
    "개찰일자",
    "연도",
    "물품번호",
    "물품명",
    "사업자번호",
    "업체명",
    "label",
]


def norm_digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def parse_date(value: str | None) -> dt.date | None:
    text = (value or "")[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def parse_float(value: str | None) -> float:
    text = re.sub(r"[^0-9.\-]", "", value or "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


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
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class AhoCorasickMatcher:
    def __init__(self, patterns: list[str]) -> None:
        self.trie: list[dict[str, int]] = [{}]
        self.out: list[list[str]] = [[]]
        self.fail: list[int] = [0]
        for pattern in patterns:
            self.add(pattern)
        self.build()

    def add(self, pattern: str) -> None:
        node = 0
        for char in pattern:
            nxt = self.trie[node].get(char)
            if nxt is None:
                nxt = len(self.trie)
                self.trie[node][char] = nxt
                self.trie.append({})
                self.out.append([])
                self.fail.append(0)
            node = nxt
        self.out[node].append(pattern)

    def build(self) -> None:
        queue: deque[int] = deque(self.trie[0].values())
        while queue:
            current = queue.popleft()
            for char, nxt in self.trie[current].items():
                queue.append(nxt)
                fallback = self.fail[current]
                while fallback and char not in self.trie[fallback]:
                    fallback = self.fail[fallback]
                self.fail[nxt] = self.trie[fallback].get(char, 0)
                self.out[nxt].extend(self.out[self.fail[nxt]])

    def find(self, text: str) -> list[str]:
        node = 0
        found: set[str] = set()
        for char in text:
            while node and char not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(char, 0)
            if self.out[node]:
                found.update(self.out[node])
        # Prefer the most specific item name if a shorter match is contained
        # inside a longer matched item name.
        return [name for name in found if not any(name != other and name in other for other in found)]


@dataclass
class SupplyIndex:
    name_to_code: dict[str, str]
    code_to_name: dict[str, str]
    suppliers_by_code: dict[str, list[str]]
    first_reg: dict[tuple[str, str], dt.date]
    flags_by_pair: dict[tuple[str, str], tuple[int, int]]
    registrations_by_biz: dict[str, list[tuple[dt.date, str, int, int]]]
    biz_name: dict[str, str]


def build_supply_index() -> SupplyIndex:
    # The supply index defines the eligible supplier pool for each item and
    # keeps registration/representative/manufacturer flags by item-supplier pair.
    name_to_codes: dict[str, set[str]] = defaultdict(set)
    code_to_name: dict[str, str] = {}
    suppliers_by_code_sets: dict[str, set[str]] = defaultdict(set)
    first_reg: dict[tuple[str, str], dt.date] = {}
    flags_by_pair: dict[tuple[str, str], tuple[int, int]] = {}
    registrations_by_biz: dict[str, list[tuple[dt.date, str, int, int]]] = defaultdict(list)

    for row in read_csv_rows(SUPPLY_PATH):
        code = (row.get("세부품명번호") or "").strip()
        name = (row.get("세부품명") or "").strip()
        bizno = norm_digits(row.get("사업자번호"))
        reg_date = parse_date(row.get("등록일시"))
        representative = 1 if (row.get("대표물품여부") or "").strip().upper() == "Y" else 0
        manufacturer = 1 if (row.get("제조여부") or "").strip().upper() == "Y" else 0
        if not code or not name or not bizno or not reg_date:
            continue
        if len(name) >= 3:
            name_to_codes[name].add(code)
        code_to_name.setdefault(code, name)
        suppliers_by_code_sets[code].add(bizno)

        pair = (code, bizno)
        if pair not in first_reg or reg_date < first_reg[pair]:
            first_reg[pair] = reg_date
        old_rep, old_manu = flags_by_pair.get(pair, (0, 0))
        flags_by_pair[pair] = (max(old_rep, representative), max(old_manu, manufacturer))
        registrations_by_biz[bizno].append((reg_date, code, representative, manufacturer))

    name_to_code = {name: next(iter(codes)) for name, codes in name_to_codes.items() if len(codes) == 1}
    suppliers_by_code = {code: sorted(values) for code, values in suppliers_by_code_sets.items()}
    for registrations in registrations_by_biz.values():
        registrations.sort(key=lambda item: item[0])

    biz_name = load_biz_names()
    return SupplyIndex(name_to_code, code_to_name, suppliers_by_code, first_reg, flags_by_pair, registrations_by_biz, biz_name)


def load_biz_names() -> dict[str, str]:
    result: dict[str, str] = {}
    path = STANDARD / "조달청_나라장터_사용자정보서비스_조달업체기본정보조회.csv"
    if not path.exists():
        return result
    for row in read_csv_rows(path):
        bizno = norm_digits(row.get("사업자등록번호") or row.get("사업자번호"))
        name = row.get("업체명") or row.get("상호명") or row.get("supplierNm") or ""
        if bizno and name:
            result[bizno] = name
    return result


def load_binary_sets() -> dict[str, Any]:
    def values_from(path: Path, columns: list[str]) -> set[str]:
        if not path.exists():
            return set()
        result = set()
        for row in read_csv_rows(path):
            for column in columns:
                value = norm_digits(row.get(column))
                if value:
                    result.add(value)
                    break
        return result

    sanction_periods: dict[str, list[tuple[dt.date | None, dt.date | None]]] = defaultdict(list)
    sanction_path = STANDARD / "조달청_나라장터_사용자정보서비스_부정당제재업체정보조회.csv"
    if sanction_path.exists():
        for row in read_csv_rows(sanction_path):
            bizno = norm_digits(row.get("사업자번호") or row.get("사업자등록번호"))
            if bizno:
                sanction_periods[bizno].append((parse_date(row.get("제재시작일")), parse_date(row.get("제재종료일"))))

    return {
        "startup": values_from(STANDARD / "창업진흥원_창업기업확인서발급기업정보_조회서비스.csv", ["사업자번호", "brno"]),
        "disabled": values_from(STANDARD / "한국장애인고용공단_장애인_표준사업장_실시간_조회.csv", ["사업자번호", "compBizNo"]),
        "social": values_from(STANDARD / "사회적기업_조회.csv", ["사업자등록번호", "사업자번호"]),
        "sanctioned": set(sanction_periods),
        "sanction_periods": dict(sanction_periods),
    }


def sanction_features(binary_sets: dict[str, Any], bizno: str, date: dt.date) -> dict[str, float]:
    active = 0.0
    days_since_end = 0.0
    ended_days: list[int] = []
    for start, end in binary_sets.get("sanction_periods", {}).get(bizno, []):
        if start and start <= date and (end is None or date <= end):
            active = 1.0
        if end and end < date:
            ended_days.append((date - end).days)
    if ended_days:
        days_since_end = float(min(ended_days))
    return {
        "부정당제재": active,
        "제재종료후_경과일수": days_since_end,
        "제재종료후_경과일수_log": math.log1p(days_since_end),
    }


@dataclass
class PriorAwardStats:
    # Accumulated only with awards that happened before the current event.
    total_count: Counter[str]
    item_count: Counter[tuple[str, str]]
    prefix4_count: Counter[tuple[str, str]]
    recent_dates: dict[str, list[dt.date]]
    amount_sum: Counter[str]
    participant_sum_by_item: Counter[str]
    participant_count_by_item: Counter[str]
    participant_sum_by_prefix4: Counter[str]
    participant_count_by_prefix4: Counter[str]

    @classmethod
    def create(cls) -> "PriorAwardStats":
        return cls(Counter(), Counter(), Counter(), defaultdict(list), Counter(), Counter(), Counter(), Counter(), Counter())

    def update(self, bizno: str, item_code: str | None, date: dt.date, amount: float, participant_count: float = 0.0) -> None:
        if not bizno:
            return
        self.total_count[bizno] += 1
        self.recent_dates[bizno].append(date)
        self.amount_sum[bizno] += amount
        if item_code:
            self.item_count[(bizno, item_code)] += 1
            self.prefix4_count[(bizno, item_code[:4])] += 1
            if participant_count > 0:
                self.participant_sum_by_item[item_code] += participant_count
                self.participant_count_by_item[item_code] += 1
                self.participant_sum_by_prefix4[item_code[:4]] += participant_count
                self.participant_count_by_prefix4[item_code[:4]] += 1

    def recent_365(self, bizno: str, date: dt.date) -> int:
        cutoff = date - dt.timedelta(days=365)
        return sum(1 for old_date in self.recent_dates.get(bizno, []) if cutoff <= old_date < date)

    def days_since_last_award(self, bizno: str, date: dt.date) -> float:
        prior_dates = [old_date for old_date in self.recent_dates.get(bizno, []) if old_date < date]
        if not prior_dates:
            return 9999.0
        return float((date - max(prior_dates)).days)

    def avg_participants(self, item_code: str) -> float:
        if self.participant_count_by_item[item_code]:
            return float(self.participant_sum_by_item[item_code] / self.participant_count_by_item[item_code])
        prefix4 = item_code[:4]
        if self.participant_count_by_prefix4[prefix4]:
            return float(self.participant_sum_by_prefix4[prefix4] / self.participant_count_by_prefix4[prefix4])
        return 0.0


def supplier_features(index: SupplyIndex, code: str, bizno: str, date: dt.date) -> dict[str, float]:
    representative, manufacturer = index.flags_by_pair.get((code, bizno), (0, 0))
    regs = [row for row in index.registrations_by_biz.get(bizno, []) if row[0] <= date]
    pair_reg_date = index.first_reg.get((code, bizno))
    registration_days = float((date - pair_reg_date).days) if pair_reg_date and pair_reg_date <= date else 0.0
    prefix2 = code[:2]
    prefix4 = code[:4]
    prefix6 = code[:6]
    same_prefix4_count = float(sum(1 for _, registered_code, _, _ in regs if registered_code.startswith(prefix4)))
    return {
        "대표물품": float(representative),
        "제조업체": float(manufacturer),
        "공급품목등록후_경과일수": registration_days,
        "공급품목등록후_경과일수_log": math.log1p(registration_days),
        "업체_등록품목수": float(len({registered_code for _, registered_code, _, _ in regs})),
        "업체_동일대분류등록수": float(sum(1 for _, registered_code, _, _ in regs if registered_code.startswith(prefix2))),
        "업체_동일분류공급품목수": same_prefix4_count,
        "업체_동일중분류등록수": same_prefix4_count,
        "업체_동일소분류등록수": float(sum(1 for _, registered_code, _, _ in regs if registered_code.startswith(prefix6))),
        "업체_대표물품등록수": float(sum(rep for _, _, rep, _ in regs)),
        "업체_제조품목등록수": float(sum(manu for _, _, _, manu in regs)),
    }


def item_pool(index: SupplyIndex, code: str, date: dt.date) -> list[str]:
    result = []
    for bizno in index.suppliers_by_code.get(code, []):
        reg_date = index.first_reg.get((code, bizno))
        if reg_date and reg_date <= date:
            result.append(bizno)
    return result


def item_features(index: SupplyIndex, code: str, pool: list[str]) -> dict[str, float]:
    representative_count = 0
    manufacturer_count = 0
    for bizno in pool:
        representative, manufacturer = index.flags_by_pair.get((code, bizno), (0, 0))
        representative_count += representative
        manufacturer_count += manufacturer
    count = len(pool)
    return {
        "품목_공급업체수": float(count),
        "품목_대표물품업체수": float(representative_count),
        "품목_제조업체수": float(manufacturer_count),
        "품목_제조업체비율": manufacturer_count / count if count else 0.0,
        "품목_대표물품비율": representative_count / count if count else 0.0,
    }


def build_feature_row(
    *,
    index: SupplyIndex,
    binary_sets: dict[str, Any],
    prior: PriorAwardStats,
    code: str,
    bizno: str,
    date: dt.date,
    pool: list[str],
    participant_count: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    values.update(supplier_features(index, code, bizno, date))
    values.update(item_features(index, code, pool))
    avg_participants = prior.avg_participants(code)
    days_since_last = prior.days_since_last_award(bizno, date)
    same_item_count = float(prior.item_count[(bizno, code)])
    values.update({
        "이전전체낙찰건수": float(prior.total_count[bizno]),
        "이전동일품목낙찰건수": same_item_count,
        "해당물품_과거낙찰건수": same_item_count,
        "이전동일대분류낙찰건수": float(prior.prefix4_count[(bizno, code[:4])]),
        "최근365일낙찰건수": float(prior.recent_365(bizno, date)),
        "최근낙찰일_경과일수": days_since_last,
        "최근낙찰일_경과일수_log": math.log1p(days_since_last),
        "이전낙찰금액합계_log": math.log1p(prior.amount_sum[bizno]),
        "참가업체수_log": math.log1p(participant_count),
        "품목_평균참가업체수": avg_participants,
        "품목_평균참가업체수_log": math.log1p(avg_participants),
        "품목_경쟁강도": avg_participants / max(len(pool), 1),
        "물품번호_prefix2": float(int(code[:2])) if code[:2].isdigit() else 0.0,
        "물품번호_prefix4": float(int(code[:4])) if code[:4].isdigit() else 0.0,
        "물품번호_prefix6": float(int(code[:6])) if code[:6].isdigit() else 0.0,
        "창업기업": 1.0 if bizno in binary_sets["startup"] else 0.0,
        "장애인표준사업장": 1.0 if bizno in binary_sets["disabled"] else 0.0,
        "사회적기업": 1.0 if bizno in binary_sets["social"] else 0.0,
    })
    values.update(sanction_features(binary_sets, bizno, date))
    return values


@dataclass
class AwardEvent:
    row: dict[str, str]
    date: dt.date
    year: str
    item_code: str | None
    item_name: str | None
    winner_bizno: str
    event_id: str


def load_award_events(index: SupplyIndex, matcher: AhoCorasickMatcher) -> list[AwardEvent]:
    events: list[AwardEvent] = []
    for sequence, row in enumerate(read_csv_rows(AWARD_PATH), 1):
        date = parse_date(row.get("개찰일시") or row.get("등록일시"))
        if not date:
            continue
        year = str(date.year)
        title = row.get("입찰공고명") or ""
        matched_names = matcher.find(title)
        item_code = None
        item_name = None
        # Use title matching only when one item name is identified unambiguously.
        if len(matched_names) == 1:
            item_name = matched_names[0]
            item_code = index.name_to_code[item_name]
        bid_no = row.get("입찰공고번호") or f"row{sequence}"
        event_id = f"{bid_no}_{row.get('입찰공고차수') or ''}_{row.get('입찰분류번호') or ''}_{row.get('재입찰번호') or ''}_{sequence}"
        events.append(AwardEvent(row, date, year, item_code, item_name, norm_digits(row.get("사업자번호")), event_id))
    events.sort(key=lambda event: (event.date, event.event_id))
    return events


def build_dataset(
    index: SupplyIndex,
    binary_sets: dict[str, Any],
    matcher: AhoCorasickMatcher,
    train_negative_samples: int,
    test_negative_samples: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], PriorAwardStats]:
    rng = random.Random(random_seed)
    prior = PriorAwardStats.create()
    rows: list[dict[str, Any]] = []
    summary = Counter()

    events = load_award_events(index, matcher)
    for event in events:
        amount = parse_float(event.row.get("낙찰금액"))
        participant_count = parse_float(event.row.get("참가업체수"))

        if event.year in {"2022", "2023", "2024", "2025"} and event.item_code and event.winner_bizno:
            pool = item_pool(index, event.item_code, event.date)
            if event.winner_bizno in pool and len(pool) > 1:
                # One actual winner is paired with sampled non-winners from the
                # same eligible item pool to train a supplier ranking model.
                dataset = "train" if event.year in {"2022", "2023", "2024"} else "test"
                negative_count = train_negative_samples if dataset == "train" else test_negative_samples
                negatives = [bizno for bizno in pool if bizno != event.winner_bizno]
                if len(negatives) > negative_count:
                    negatives = rng.sample(negatives, negative_count)
                sampled = [(event.winner_bizno, 1)] + [(bizno, 0) for bizno in negatives]
                for bizno, label in sampled:
                    features = build_feature_row(
                        index=index,
                        binary_sets=binary_sets,
                        prior=prior,
                        code=event.item_code,
                        bizno=bizno,
                        date=event.date,
                        pool=pool,
                        participant_count=participant_count,
                    )
                    output_row: dict[str, Any] = {
                        "dataset": dataset,
                        "event_id": event.event_id,
                        "입찰공고번호": event.row.get("입찰공고번호", ""),
                        "개찰일자": event.date.isoformat(),
                        "연도": event.year,
                        "물품번호": event.item_code,
                        "물품명": event.item_name or index.code_to_name.get(event.item_code, ""),
                        "사업자번호": bizno,
                        "업체명": event.row.get("낙찰업체명", "") if label else index.biz_name.get(bizno, ""),
                        "label": label,
                    }
                    output_row.update(features)
                    rows.append(output_row)
                summary[f"{dataset}_positive_events"] += 1
                summary[f"{dataset}_rows"] += len(sampled)
            else:
                summary["matched_but_winner_not_in_prior_supply_pool"] += 1
        elif event.item_code:
            summary["matched_item_awards_outside_years_or_missing_biz"] += 1
        else:
            summary["unmatched_item_awards"] += 1

        # Award history is updated after feature extraction to avoid target leakage.
        prior.update(event.winner_bizno, event.item_code, event.date, amount, participant_count)

    return rows, dict(summary), prior


class SimpleDecisionTree:
    def __init__(
        self,
        *,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        random_state: random.Random,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree: dict[str, Any] | None = None
        self.feature_importance_: Counter[int] = Counter()

    @staticmethod
    def gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = float(np.mean(y))
        return 2.0 * p * (1.0 - p)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        indices = np.arange(len(y))
        self.tree = self._build(x, y, indices, 0)

    def _build(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> dict[str, Any]:
        labels = y[indices]
        probability = float(np.mean(labels)) if len(labels) else 0.0
        if (
            depth >= self.max_depth
            or len(indices) < self.min_samples_leaf * 2
            or probability <= 0.0
            or probability >= 1.0
        ):
            return {"leaf": True, "prob": probability}

        feature_count = x.shape[1]
        feature_indices = self.random_state.sample(range(feature_count), min(self.max_features, feature_count))
        best_feature = -1
        best_threshold = 0.0
        best_score = float("inf")
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        parent_gini = self.gini(labels)
        for feature in feature_indices:
            values = x[indices, feature]
            unique_values = np.unique(values)
            if len(unique_values) <= 1:
                continue
            if len(unique_values) > 16:
                thresholds = np.quantile(unique_values, np.linspace(0.1, 0.9, 9))
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in np.unique(thresholds):
                left_mask = values <= threshold
                left = indices[left_mask]
                right = indices[~left_mask]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                score = (len(left) * self.gini(y[left]) + len(right) * self.gini(y[right])) / len(indices)
                if score < best_score:
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_score = float(score)
                    best_left = left
                    best_right = right

        if best_feature < 0 or best_left is None or best_right is None:
            return {"leaf": True, "prob": probability}

        gain = max(0.0, parent_gini - best_score)
        self.feature_importance_[best_feature] += gain * len(indices)
        return {
            "leaf": False,
            "prob": probability,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(x, y, best_left, depth + 1),
            "right": self._build(x, y, best_right, depth + 1),
        }

    def predict_one(self, row: np.ndarray) -> float:
        assert self.tree is not None
        node = self.tree
        while not node["leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["prob"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(row) for row in x], dtype=float)


class SimpleRandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 40,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        max_features: int | None = None,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.trees: list[SimpleDecisionTree] = []
        self.feature_importance_: Counter[int] = Counter()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = random.Random(self.random_seed)
        self.trees = []
        sample_size = len(y)
        max_features = self.max_features or max(1, int(math.sqrt(x.shape[1])))
        for _ in range(self.n_estimators):
            indices = np.array([rng.randrange(sample_size) for _ in range(sample_size)], dtype=int)
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=random.Random(rng.randrange(1_000_000_000)),
            )
            tree.fit(x[indices], y[indices])
            self.trees.append(tree)
            for feature, value in tree.feature_importance_.items():
                self.feature_importance_[feature] += value

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            return np.zeros(len(x), dtype=float)
        predictions = np.vstack([tree.predict_proba(x) for tree in self.trees])
        return predictions.mean(axis=0)


def rows_to_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row.get(column, 0.0) or 0.0) for column in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return x, y


def predict_positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]
    return np.asarray(probabilities, dtype=float)


def build_random_forest(args: argparse.Namespace) -> Any:
    use_sklearn = args.engine == "sklearn" or (args.engine == "auto" and RandomForestClassifier is not None)
    if use_sklearn:
        if RandomForestClassifier is None:
            raise SystemExit("scikit-learn이 설치되어 있지 않습니다. --engine simple을 쓰거나 scikit-learn을 설치하세요.")
        print("모델 엔진: scikit-learn RandomForestClassifier")
        return RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
    print("모델 엔진: 내장 SimpleRandomForestClassifier")
    return SimpleRandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_seed=args.seed,
    )


def feature_importance_rows(model: Any) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        importances = list(enumerate(np.asarray(model.feature_importances_, dtype=float)))
        importances.sort(key=lambda item: item[1], reverse=True)
        total = float(sum(value for _, value in importances)) or 1.0
        return [
            {
                "feature": FEATURE_COLUMNS[index],
                "importance": round(float(value) / total, 6),
                "raw_importance": round(float(value), 6),
            }
            for index, value in importances
        ]

    total = sum(model.feature_importance_.values()) or 1.0
    return [
        {
            "feature": FEATURE_COLUMNS[index],
            "importance": round(value / total, 6),
            "raw_importance": round(value, 6),
        }
        for index, value in model.feature_importance_.most_common()
    ]


def evaluate_predictions(rows: list[dict[str, Any]], probabilities: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_event: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        by_event[str(row["event_id"])].append((row, float(probability)))

    prediction_rows: list[dict[str, Any]] = []
    hit1 = hit3 = hit5 = 0
    mrr_sum = 0.0
    event_count = 0
    for event_id, event_rows in by_event.items():
        event_rows.sort(key=lambda item: item[1], reverse=True)
        positive_rank = None
        for rank, (row, probability) in enumerate(event_rows, 1):
            if rank <= 10:
                prediction_rows.append({
                    "event_id": event_id,
                    "rank": rank,
                    "AI예측점수": round(probability, 6),
                    "label": row["label"],
                    "물품번호": row["물품번호"],
                    "물품명": row["물품명"],
                    "사업자번호": row["사업자번호"],
                    "업체명": row.get("업체명", ""),
                    "개찰일자": row["개찰일자"],
                    "입찰공고번호": row["입찰공고번호"],
                })
            if int(row["label"]) == 1:
                positive_rank = rank
        if positive_rank is None:
            continue
        event_count += 1
        hit1 += 1 if positive_rank <= 1 else 0
        hit3 += 1 if positive_rank <= 3 else 0
        hit5 += 1 if positive_rank <= 5 else 0
        mrr_sum += 1.0 / positive_rank

    metrics = {
        "test_events": float(event_count),
        "hit_at_1": hit1 / event_count if event_count else 0.0,
        "hit_at_3": hit3 / event_count if event_count else 0.0,
        "hit_at_5": hit5 / event_count if event_count else 0.0,
        "mrr": mrr_sum / event_count if event_count else 0.0,
    }
    return prediction_rows, metrics


def score_official_candidates(
    index: SupplyIndex,
    binary_sets: dict[str, Any],
    prior: PriorAwardStats,
    model: Any,
) -> list[dict[str, Any]]:
    today = dt.date(2026, 5, 9)
    rows = []
    candidates = read_csv_rows(CANDIDATES_PATH)
    feature_rows = []
    # Current 후보업체_목록.csv is scored with the trained model for dashboard use.
    for candidate in candidates:
        code = (candidate.get("물품번호") or "").strip()
        bizno = norm_digits(candidate.get("사업자번호_정규화") or candidate.get("사업자번호"))
        pool = item_pool(index, code, today)
        features = build_feature_row(
            index=index,
            binary_sets=binary_sets,
            prior=prior,
            code=code,
            bizno=bizno,
            date=today,
            pool=pool,
            participant_count=0.0,
        )
        feature_rows.append(features)
    x = np.array([[float(row.get(column, 0.0) or 0.0) for column in FEATURE_COLUMNS] for row in feature_rows], dtype=float)
    probabilities = predict_positive_proba(model, x)
    for candidate, features, probability in zip(candidates, feature_rows, probabilities):
        row = {key: candidate.get(key, "") for key in ["candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명"]}
        row["AI예측점수"] = round(float(probability), 6)
        for column in FEATURE_COLUMNS:
            row[column] = features.get(column, 0.0)
        rows.append(row)

    global_ranked = sorted(rows, key=lambda item: float(item["AI예측점수"]), reverse=True)
    for rank, row in enumerate(global_ranked, 1):
        row["전체_AI순위"] = rank

    rows_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_item[str(row["물품번호"])].append(row)
    for item_rows in rows_by_item.values():
        item_rows.sort(key=lambda item: float(item["AI예측점수"]), reverse=True)
        for rank, row in enumerate(item_rows, 1):
            row["물품별_AI순위"] = rank

    return sorted(rows, key=lambda item: (str(item["물품번호"]), int(item["물품별_AI순위"])))


def write_ai_rule_comparison(official_rows: list[dict[str, Any]]) -> None:
    rule_path = PROCESSED / "scored_outputs" / "후보업체_목록" / "업체추천_점수_균형형.csv"
    if not rule_path.exists():
        return

    ai_by_id = {str(row["candidate_id"]): row for row in official_rows}
    rule_rows = read_csv_rows(rule_path)
    rule_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rule_rows:
        rule_by_item[str(row.get("물품번호", ""))].append(row)
    for item_rows in rule_by_item.values():
        item_rows.sort(key=lambda row: parse_float(row.get("프리셋점수")), reverse=True)
        for rank, row in enumerate(item_rows, 1):
            row["규칙기반_균형형_물품별순위"] = str(rank)

    rows = []
    for rule in rule_rows:
        ai = ai_by_id.get(str(rule.get("candidate_id", "")), {})
        rows.append({
            "candidate_id": rule.get("candidate_id", ""),
            "물품번호": rule.get("물품번호", ""),
            "물품명": rule.get("물품명", ""),
            "사업자번호_정규화": rule.get("사업자번호_정규화", ""),
            "업체명": rule.get("업체명", ""),
            "물품별_AI순위": ai.get("물품별_AI순위", ""),
            "전체_AI순위": ai.get("전체_AI순위", ""),
            "AI예측점수": ai.get("AI예측점수", ""),
            "규칙기반_균형형_물품별순위": rule.get("규칙기반_균형형_물품별순위", ""),
            "규칙기반_균형형_전체순위": rule.get("순위", ""),
            "규칙기반_균형형점수": rule.get("프리셋점수", ""),
            "대표물품": rule.get("대표물품", ""),
            "제조업체": rule.get("제조업체", ""),
            "낙찰건수": rule.get("낙찰건수", ""),
            "최근낙찰건수": rule.get("최근낙찰건수", ""),
            "부정당제재": rule.get("부정당제재", ""),
            "현재제재여부": rule.get("현재제재여부", ""),
            "창업기업": rule.get("창업기업", ""),
            "장애인표준사업장": rule.get("장애인표준사업장", ""),
        })

    rows.sort(key=lambda row: (str(row["물품번호"]), int(row["물품별_AI순위"] or 999999)))
    columns = [
        "candidate_id",
        "물품번호",
        "물품명",
        "사업자번호_정규화",
        "업체명",
        "물품별_AI순위",
        "전체_AI순위",
        "AI예측점수",
        "규칙기반_균형형_물품별순위",
        "규칙기반_균형형_전체순위",
        "규칙기반_균형형점수",
        "대표물품",
        "제조업체",
        "낙찰건수",
        "최근낙찰건수",
        "부정당제재",
        "현재제재여부",
        "창업기업",
        "장애인표준사업장",
    ]
    write_csv(OUTPUT_DIR / "후보업체_AI_규칙점수_비교.csv", rows, columns)
    write_csv(
        OUTPUT_DIR / "후보업체_AI_TOP5_물품별.csv",
        [row for row in rows if int(row["물품별_AI순위"] or 999999) <= 5],
        columns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Random Forest supplier-ranking prototype.")
    parser.add_argument("--train-negatives", type=int, default=10)
    parser.add_argument("--test-negatives", type=int, default=30)
    parser.add_argument("--trees", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--engine", choices=["auto", "sklearn", "simple"], default="auto")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("공급물품 인덱스 생성 중...")
    index = build_supply_index()
    print(f"품목명 패턴 {len(index.name_to_code):,}개, 공급업체-품목 조합 {len(index.first_reg):,}개")
    matcher = AhoCorasickMatcher(list(index.name_to_code))
    binary_sets = load_binary_sets()

    print("학습/평가 데이터셋 생성 중...")
    rows, summary, prior = build_dataset(
        index=index,
        binary_sets=binary_sets,
        matcher=matcher,
        train_negative_samples=args.train_negatives,
        test_negative_samples=args.test_negatives,
        random_seed=args.seed,
    )
    train_rows = [row for row in rows if row["dataset"] == "train"]
    test_rows = [row for row in rows if row["dataset"] == "test"]
    print(f"train rows={len(train_rows):,}, test rows={len(test_rows):,}")

    if args.save_dataset:
        write_csv(OUTPUT_DIR / "supplier_rank_training_dataset.csv", rows, BASE_OUTPUT_COLUMNS + FEATURE_COLUMNS)

    train_x, train_y = rows_to_matrix(train_rows)
    test_x, test_y = rows_to_matrix(test_rows)
    model = build_random_forest(args)
    print("랜덤포레스트 학습 중...")
    model.fit(train_x, train_y)

    train_probs = predict_positive_proba(model, train_x)
    test_probs = predict_positive_proba(model, test_x)
    prediction_rows, metrics = evaluate_predictions(test_rows, test_probs)

    # Simple row-level diagnostics, separate from ranking metrics.
    row_metrics = {
        "train_positive_rate": float(np.mean(train_y)) if len(train_y) else 0.0,
        "test_positive_rate": float(np.mean(test_y)) if len(test_y) else 0.0,
        "train_positive_score_mean": float(np.mean(train_probs[train_y == 1])) if np.any(train_y == 1) else 0.0,
        "train_negative_score_mean": float(np.mean(train_probs[train_y == 0])) if np.any(train_y == 0) else 0.0,
        "test_positive_score_mean": float(np.mean(test_probs[test_y == 1])) if np.any(test_y == 1) else 0.0,
        "test_negative_score_mean": float(np.mean(test_probs[test_y == 0])) if np.any(test_y == 0) else 0.0,
    }

    evaluation_rows = [{"metric": key, "value": round(value, 6)} for key, value in {**metrics, **row_metrics}.items()]
    for key, value in summary.items():
        evaluation_rows.append({"metric": key, "value": value})
    evaluation_rows.extend([
        {"metric": "train_rows", "value": len(train_rows)},
        {"metric": "test_rows", "value": len(test_rows)},
        {"metric": "train_negative_samples_per_event", "value": args.train_negatives},
        {"metric": "test_negative_samples_per_event", "value": args.test_negatives},
    ])
    write_csv(OUTPUT_DIR / "random_forest_evaluation.csv", evaluation_rows, ["metric", "value"])
    write_csv(
        OUTPUT_DIR / "random_forest_predictions_2025_top10.csv",
        prediction_rows,
        ["event_id", "rank", "AI예측점수", "label", "물품번호", "물품명", "사업자번호", "업체명", "개찰일자", "입찰공고번호"],
    )

    write_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", feature_importance_rows(model), ["feature", "importance", "raw_importance"])

    official_rows = score_official_candidates(index, binary_sets, prior, model)
    write_csv(
        OUTPUT_DIR / "후보업체_AI예측점수.csv",
        official_rows,
        ["물품별_AI순위", "전체_AI순위", "candidate_id", "물품번호", "물품명", "사업자번호", "사업자번호_정규화", "업체명", "AI예측점수"] + FEATURE_COLUMNS,
    )
    write_ai_rule_comparison(official_rows)

    with (OUTPUT_DIR / "supplier_rank_random_forest.pkl").open("wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLUMNS, "summary": summary, "metrics": metrics}, f)
    if joblib is not None:
        joblib.dump({"model": model, "features": FEATURE_COLUMNS, "summary": summary, "metrics": metrics}, OUTPUT_DIR / "supplier_rank_random_forest.joblib")

    print("완료")
    print(f"Hit@1={metrics['hit_at_1']:.4f}, Hit@3={metrics['hit_at_3']:.4f}, Hit@5={metrics['hit_at_5']:.4f}, MRR={metrics['mrr']:.4f}")
    print(f"결과 폴더: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
