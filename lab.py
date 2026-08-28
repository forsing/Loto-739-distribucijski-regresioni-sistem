#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Insipration/Upgarde - Inspiracija/Nadogradnja
# https://github.com/ahmad-al-hamza/Lotto_6_aus_49_ML_Lab


"""
Loto 7/39 — distribucijski regresioni sistem

1. učitava dva CSV-a bez zaglavlja;
2. proverava da svaki red sadrži sedam različitih brojeva 1-39;
3. poštuje hronologiju: prvi red je najstariji, poslednji najnoviji;
4. pravi kontinuiranu time-to-event metu za svaki broj;
5. koristi raspodele razmaka, survival, hazard i uslovne prelaze;
6. obrađuje istorijske odnose parova;
7. pravi stvarne, near-miss i istorijski pronađene kombinacije;
8. koristi grupnu hronološku obuku i odvojenu validaciju;
9. poredi brzu i preciznu regresionu fazu;
10. paketno pregleda svih C(39, 7) = 15.380.937 kombinacija;
11. daje jednu NEXT kombinaciju za Loto;
12. daje jednu NEXT kombinaciju za Loto Plus.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error


# ============================================================
# PUTANJE
# ============================================================

LOTO_CSV = Path(
    "/Users/4c/Desktop/GHQ/data/"
    "loto7_4674_k68_loto_2959.csv"
)

LOTO_PLUS_CSV = Path(
    "/Users/4c/Desktop/GHQ/data/"
    "loto7_4674_k68_loto_plus_1715.csv"
)


# ============================================================
# OSNOVNE POSTAVKE
# ============================================================

MIN_NUMBER = 1
MAX_NUMBER = 39
TOP_K = 7

NUMBER_COUNT = 39
PAIR_COUNT = math.comb(39, 2)
TOTAL_COMBINATIONS = math.comb(39, 7)

BASE_RATE = TOP_K / NUMBER_COUNT
BASE_WAIT_NUMBER = NUMBER_COUNT / TOP_K

PAIR_RATE = math.comb(7, 2) / math.comb(39, 2)
BASE_WAIT_PAIR = 1.0 / PAIR_RATE

RANDOM_EXPECTED_HITS = TOP_K * TOP_K / NUMBER_COUNT

RANDOM_SEED = 20260828

# Najmanja istorija pre pravljenja osobina.
MIN_HISTORY = 200

# Prozori raspodele pojavljivanja.
WINDOWS = (10, 20, 50, 100, 200)

# Broj hronoloških grupa.
REQUESTED_GROUPS = 98

# Poslednji deo grupa predstavlja odvojenu validaciju.
VALIDATION_FRACTION = 0.20

# Minimalna budućnost iza istorijske grupe potrebna za mete parova.
MIN_FUTURE_FOR_GROUP = 250

# Kandidati napravljeni od najbolje rangiranih pojedinačnih brojeva.
RETRIEVAL_POOL = 12
RETRIEVED_PER_GROUP = 80

# Broj najboljih zamena za svaki broj u near-miss kombinaciji.
NEAR_MISS_REPLACEMENTS = 5

# Broj prethodnih istorijskih kombinacija dodatih grupi.
HISTORICAL_CANDIDATES = 30

# Broj kandidata koje brza faza prosleđuje preciznoj fazi.
VALIDATION_FAST_KEEP = 80
FINAL_FAST_KEEP = 50_000

# Recall@K na odvojenim hronološkim grupama.
RECALL_K = 10

# Paket za pregled svih kombinacija.
COMBINATION_BATCH_SIZE = 150_000

# Težine kontinuirane kombinacijske mete.
TARGET_NUMBER_WEIGHT = 0.50
TARGET_PAIR_WEIGHT = 0.30
TARGET_OVERLAP_WEIGHT = 0.20

# Budući vremenski ponderisan presek.
OVERLAP_HORIZON = 8
OVERLAP_DECAY = 0.82

# Zaglađivanje uslovnih prelaza.
TRANSITION_ALPHA = 20.0

# Statistička procena validacionih pogodaka.
STATISTICAL_SIMULATIONS = 10_000

EPSILON = 1e-12


# ============================================================
# PAROVI
# ============================================================

PAIR_LIST = np.asarray(
    list(combinations(range(NUMBER_COUNT), 2)),
    dtype=np.int16,
)

PAIR_INDEX = np.full(
    (NUMBER_COUNT, NUMBER_COUNT),
    -1,
    dtype=np.int16,
)

for pair_id, (left, right) in enumerate(PAIR_LIST):
    PAIR_INDEX[left, right] = pair_id
    PAIR_INDEX[right, left] = pair_id

LOCAL_PAIR_LEFT = np.asarray(
    [a for a, _ in combinations(range(TOP_K), 2)],
    dtype=np.int8,
)

LOCAL_PAIR_RIGHT = np.asarray(
    [b for _, b in combinations(range(TOP_K), 2)],
    dtype=np.int8,
)


# ============================================================
# REZULTAT
# ============================================================

@dataclass
class GameResult:
    name: str
    prediction: np.ndarray
    rows: int
    combinations_checked: int
    training_groups: int
    validation_groups: int
    fast_mae: float
    precise_mae: float
    fast_recall: float
    precise_recall: float
    selected_stage: str
    holdout_average_hits: float
    holdout_difference: float
    bootstrap_low: float
    bootstrap_high: float
    permutation_p: float
    elapsed_seconds: float


# ============================================================
# UČITAVANJE I PROVERA CSV-A
# ============================================================

def load_lottery_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"CSV nije pronađen: {path}")

    try:
        frame = pd.read_csv(
            path,
            header=None,
            sep=None,
            engine="python",
            dtype=str,
        )
    except Exception as error:
        raise ValueError(
            f"CSV ne može da se učita: {path}\n{error}"
        ) from error

    # Uklanjanje potpuno praznih redova i kolona.
    frame = frame.dropna(axis=0, how="all")
    frame = frame.dropna(axis=1, how="all")

    numeric_columns: list[pd.Series] = []

    for column in frame.columns:
        converted = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.strip()
            .str.replace('"', "", regex=False),
            errors="coerce",
        )

        if converted.notna().all():
            numeric_columns.append(converted)

    if len(numeric_columns) != TOP_K:
        raise ValueError(
            f"{path}\n"
            f"Očekivano je tačno 7 brojčanih kolona, "
            f"a pronađeno je {len(numeric_columns)}."
        )

    data = pd.concat(numeric_columns, axis=1).to_numpy(dtype=np.int16)

    if len(data) <= MIN_HISTORY + MIN_FUTURE_FOR_GROUP:
        raise ValueError(
            f"{path}\n"
            f"CSV ima samo {len(data)} redova. "
            f"Potrebno je više od "
            f"{MIN_HISTORY + MIN_FUTURE_FOR_GROUP}."
        )

    if np.any(data < MIN_NUMBER) or np.any(data > MAX_NUMBER):
        bad_rows = np.where(
            np.any(
                (data < MIN_NUMBER) | (data > MAX_NUMBER),
                axis=1,
            )
        )[0]

        raise ValueError(
            f"{path}\n"
            f"Broj van opsega 1–39 pronađen je u redu "
            f"{int(bad_rows[0]) + 1}."
        )

    for row_index, row in enumerate(data):
        if len(np.unique(row)) != TOP_K:
            raise ValueError(
                f"{path}\n"
                f"Red {row_index + 1} ne sadrži "
                f"sedam različitih brojeva: {row.tolist()}"
            )

    # Poredak unutar reda nema vremensko značenje.
    data = np.sort(data, axis=1)

    return data


# ============================================================
# MATRICE POJAVLJIVANJA
# ============================================================

def create_number_matrix(draws: np.ndarray) -> np.ndarray:
    matrix = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.int8,
    )

    row_indices = np.repeat(
        np.arange(len(draws)),
        TOP_K,
    )

    column_indices = draws.reshape(-1) - 1

    matrix[row_indices, column_indices] = 1

    return matrix


def create_pair_matrix(draws: np.ndarray) -> np.ndarray:
    matrix = np.zeros(
        (len(draws), PAIR_COUNT),
        dtype=np.int8,
    )

    for row_index, draw in enumerate(draws):
        zero_based = draw - 1

        pair_ids = PAIR_INDEX[
            zero_based[LOCAL_PAIR_LEFT],
            zero_based[LOCAL_PAIR_RIGHT],
        ]

        matrix[row_index, pair_ids] = 1

    return matrix


def next_wait_matrix(
    presence: np.ndarray,
) -> np.ndarray:
    """
    wait[t, i] je broj izvlačenja od trenutka t do prvog
    pojavljivanja elementa i, uključujući mogućnost da se
    pojavi upravo u izvlačenju t.

    Ako buduće pojavljivanje nije poznato, vrednost je NaN.
    """

    rows, columns = presence.shape

    waits = np.full(
        (rows, columns),
        np.nan,
        dtype=np.float32,
    )

    next_position = np.full(
        columns,
        rows + 1,
        dtype=np.int32,
    )

    for t in range(rows - 1, -1, -1):
        appeared = presence[t].astype(bool)
        next_position[appeared] = t

        known = next_position <= rows - 1
        waits[t, known] = (
            next_position[known] - t + 1
        ).astype(np.float32)

    return waits


# ============================================================
# DISTRIBUCIJSKE OSOBINE BROJEVA
# ============================================================

def entropy_from_gaps(gaps: np.ndarray) -> float:
    if len(gaps) < 2:
        return 0.0

    counts = np.bincount(
        gaps.astype(np.int32),
    )

    probabilities = counts[counts > 0].astype(float)
    probabilities /= probabilities.sum()

    entropy = -np.sum(
        probabilities * np.log(probabilities + EPSILON)
    )

    maximum_entropy = math.log(max(len(probabilities), 2))

    return float(entropy / maximum_entropy)


def number_features_at(
    number_matrix: np.ndarray,
    end_index: int,
) -> np.ndarray:
    """
    Pravi osobine koristeći isključivo redove [0:end_index].
    Ciljno izvlačenje end_index nije deo istorije.
    """

    history = number_matrix[:end_index]
    total = len(history)

    if total < 1:
        raise ValueError("Istorija ne sme biti prazna.")

    counts = history.sum(axis=0).astype(float)
    full_rate = counts / total

    window_rates: list[np.ndarray] = []

    for window in WINDOWS:
        part = history[-min(window, total):]
        window_rates.append(part.mean(axis=0))

    rate_10, rate_20, rate_50, rate_100, rate_200 = (
        window_rates
    )

    momentum_20_50 = rate_20 - rate_50
    momentum_50_200 = rate_50 - rate_200
    deviation = full_rate - BASE_RATE
    relative_strength = full_rate / BASE_RATE

    multi_window = np.vstack(
        [rate_20, rate_50, rate_100, rate_200]
    )

    window_mean = multi_window.mean(axis=0)
    window_std = multi_window.std(axis=0)

    current_gap = np.empty(NUMBER_COUNT, dtype=float)
    gap_q10 = np.empty(NUMBER_COUNT, dtype=float)
    gap_q25 = np.empty(NUMBER_COUNT, dtype=float)
    gap_median = np.empty(NUMBER_COUNT, dtype=float)
    gap_q75 = np.empty(NUMBER_COUNT, dtype=float)
    gap_q90 = np.empty(NUMBER_COUNT, dtype=float)
    gap_cdf = np.empty(NUMBER_COUNT, dtype=float)
    gap_survival = np.empty(NUMBER_COUNT, dtype=float)
    gap_hazard = np.empty(NUMBER_COUNT, dtype=float)
    gap_entropy = np.empty(NUMBER_COUNT, dtype=float)

    for number_index in range(NUMBER_COUNT):
        positions = np.flatnonzero(
            history[:, number_index]
        )

        if len(positions) == 0:
            current = total
            completed = np.asarray(
                [BASE_WAIT_NUMBER],
                dtype=float,
            )
        else:
            current = total - int(positions[-1])

            if len(positions) >= 2:
                completed = np.diff(positions).astype(float)
            else:
                completed = np.asarray(
                    [BASE_WAIT_NUMBER],
                    dtype=float,
                )

        current_gap[number_index] = current

        quantiles = np.quantile(
            completed,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        )

        gap_q10[number_index] = quantiles[0]
        gap_q25[number_index] = quantiles[1]
        gap_median[number_index] = quantiles[2]
        gap_q75[number_index] = quantiles[3]
        gap_q90[number_index] = quantiles[4]

        gap_cdf[number_index] = np.mean(
            completed <= current
        )

        gap_survival[number_index] = np.mean(
            completed >= current
        )

        events = np.sum(
            completed.astype(np.int32) == int(current)
        )

        at_risk = np.sum(
            completed >= current
        )

        gap_hazard[number_index] = (
            events + 1.0
        ) / (
            at_risk + 2.0
        )

        gap_entropy[number_index] = entropy_from_gaps(
            completed.astype(np.int32)
        )

    # Uslovni prelaz iz prethodnog izvlačenja.
    transition_deviation = np.zeros(
        NUMBER_COUNT,
        dtype=float,
    )

    if total >= 2:
        previous_numbers = np.flatnonzero(
            history[-1]
        )

        transition_sum = np.zeros(
            NUMBER_COUNT,
            dtype=float,
        )

        for source in previous_numbers:
            source_previous = history[:-1, source].astype(bool)
            denominator = int(source_previous.sum())

            if denominator == 0:
                conditional = np.full(
                    NUMBER_COUNT,
                    BASE_RATE,
                    dtype=float,
                )
            else:
                numerator = history[1:][
                    source_previous
                ].sum(axis=0)

                conditional = (
                    numerator
                    + TRANSITION_ALPHA * BASE_RATE
                ) / (
                    denominator + TRANSITION_ALPHA
                )

            transition_sum += conditional - BASE_RATE

        if len(previous_numbers) > 0:
            transition_deviation = (
                transition_sum / len(previous_numbers)
            )

    return np.column_stack(
        [
            full_rate,
            deviation,
            relative_strength,
            rate_10,
            rate_20,
            rate_50,
            rate_100,
            rate_200,
            momentum_20_50,
            momentum_50_200,
            window_mean,
            window_std,
            current_gap,
            current_gap / BASE_WAIT_NUMBER,
            gap_q10,
            gap_q25,
            gap_median,
            gap_q75,
            gap_q90,
            gap_cdf,
            gap_survival,
            gap_hazard,
            gap_entropy,
            transition_deviation,
        ]
    ).astype(np.float32)


# ============================================================
# DISTRIBUCIJSKI SKOROVI PAROVA
# ============================================================

def pair_scores_at(
    number_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    end_index: int,
) -> np.ndarray:
    history_pairs = pair_matrix[:end_index]
    history_numbers = number_matrix[:end_index]

    total = len(history_pairs)

    pair_counts = history_pairs.sum(axis=0).astype(float)
    pair_rate = pair_counts / max(total, 1)

    recent_window = min(100, total)
    recent_pair_rate = history_pairs[
        -recent_window:
    ].mean(axis=0)

    current_gap = np.empty(PAIR_COUNT, dtype=float)
    cdf = np.empty(PAIR_COUNT, dtype=float)
    survival = np.empty(PAIR_COUNT, dtype=float)
    hazard = np.empty(PAIR_COUNT, dtype=float)

    for pair_id in range(PAIR_COUNT):
        positions = np.flatnonzero(
            history_pairs[:, pair_id]
        )

        if len(positions) == 0:
            current = total
            completed = np.asarray(
                [BASE_WAIT_PAIR],
                dtype=float,
            )
        else:
            current = total - int(positions[-1])

            if len(positions) >= 2:
                completed = np.diff(positions).astype(float)
            else:
                completed = np.asarray(
                    [BASE_WAIT_PAIR],
                    dtype=float,
                )

        current_gap[pair_id] = current
        cdf[pair_id] = np.mean(completed <= current)
        survival[pair_id] = np.mean(completed >= current)

        events = np.sum(
            completed.astype(np.int32) == int(current)
        )

        at_risk = np.sum(
            completed >= current
        )

        hazard[pair_id] = (
            events + 1.0
        ) / (
            at_risk + 2.0
        )

    number_counts = history_numbers.sum(axis=0).astype(float)

    conditional_deviation = np.empty(
        PAIR_COUNT,
        dtype=float,
    )

    for pair_id, (left, right) in enumerate(PAIR_LIST):
        left_conditional = (
            pair_counts[pair_id]
            + TRANSITION_ALPHA * BASE_RATE
        ) / (
            number_counts[left] + TRANSITION_ALPHA
        )

        right_conditional = (
            pair_counts[pair_id]
            + TRANSITION_ALPHA * BASE_RATE
        ) / (
            number_counts[right] + TRANSITION_ALPHA
        )

        conditional_deviation[pair_id] = (
            0.5 * (left_conditional + right_conditional)
            - BASE_RATE
        )

    # Skaliranja prema teorijskim osnovama.
    rate_deviation = (
        pair_rate - PAIR_RATE
    ) / max(PAIR_RATE, EPSILON)

    recent_deviation = (
        recent_pair_rate - PAIR_RATE
    ) / max(PAIR_RATE, EPSILON)

    gap_position = np.log1p(
        current_gap / BASE_WAIT_PAIR
    )

    # Jedan stabilan kontinuirani distribucijski skor za svaki par.
    score = (
        0.22 * np.tanh(rate_deviation)
        + 0.18 * np.tanh(recent_deviation)
        + 0.18 * (cdf - 0.5)
        + 0.15 * (0.5 - survival)
        + 0.17 * np.tanh(4.0 * hazard)
        + 0.07 * np.tanh(gap_position)
        + 0.03 * np.tanh(
            conditional_deviation / max(BASE_RATE, EPSILON)
        )
    )

    return score.astype(np.float32)


# ============================================================
# REGRESIONI SKUP NA NIVOU BROJEVA
# ============================================================

def build_number_regression_data(
    number_matrix: np.ndarray,
    number_waits: np.ndarray,
    end_limit: int,
    step: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pravi regresioni skup na nivou brojeva.

    Primer iz trenutka t prihvata se samo ako je naredno
    pojavljivanje broja potpuno razrešeno pre end_limit.
    Time se sprečava curenje budućih podataka preko mete.
    """

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    if end_limit <= MIN_HISTORY:
        raise RuntimeError(
            f"Kraj obuke ({end_limit}) mora biti veći od "
            f"MIN_HISTORY ({MIN_HISTORY})."
        )

    for t in range(
        MIN_HISTORY,
        end_limit,
        step,
    ):
        waits = number_waits[t]

        # Poslednji red potreban za razrešenje mete:
        # t + wait - 1.
        resolution_position = (
            t + waits - 1
        )

        known = (
            np.isfinite(waits)
            & (resolution_position < end_limit)
        )

        if not np.any(known):
            continue

        features = number_features_at(
            number_matrix,
            t,
        )

        targets = np.log(
            BASE_WAIT_NUMBER / waits[known]
        )

        feature_parts.append(
            features[known]
        )

        target_parts.append(
            targets.astype(np.float32)
        )

    if not feature_parts:
        raise RuntimeError(
            "Nema dovoljno potpuno razrešenih "
            "time-to-event meta za brojeve."
        )

    return (
        np.vstack(feature_parts).astype(np.float32),
        np.concatenate(target_parts).astype(np.float32),
    )


def create_number_regressor(
    random_state: int,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=360,
        min_samples_leaf=4,
        max_features=0.85,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )


# ============================================================
# OSOBINE KOMBINACIJA
# ============================================================

def combination_features(
    combinations_array: np.ndarray,
    number_scores: np.ndarray,
    pair_scores: np.ndarray,
    precise: bool,
) -> np.ndarray:
    combos = np.asarray(
        combinations_array,
        dtype=np.int16,
    )

    zero_based = combos - 1

    number_values = number_scores[zero_based]

    pair_ids = PAIR_INDEX[
        zero_based[:, LOCAL_PAIR_LEFT],
        zero_based[:, LOCAL_PAIR_RIGHT],
    ]

    pair_values = pair_scores[pair_ids]

    sums = combos.sum(axis=1).astype(float)
    means = combos.mean(axis=1)
    standard_deviations = combos.std(axis=1)
    ranges = (
        combos[:, -1] - combos[:, 0]
    ).astype(float)

    odd_counts = (combos % 2 == 1).sum(axis=1).astype(float)
    low_counts = (combos <= 20).sum(axis=1).astype(float)
    consecutive_counts = (
        np.diff(combos, axis=1) == 1
    ).sum(axis=1).astype(float)

    gaps_between_numbers = np.diff(
        combos,
        axis=1,
    ).astype(float)

    compact = np.column_stack(
        [
            number_values.mean(axis=1),
            number_values.std(axis=1),
            number_values.min(axis=1),
            number_values.max(axis=1),
            np.quantile(number_values, 0.25, axis=1),
            np.quantile(number_values, 0.50, axis=1),
            np.quantile(number_values, 0.75, axis=1),
            pair_values.mean(axis=1),
            pair_values.std(axis=1),
            pair_values.min(axis=1),
            pair_values.max(axis=1),
            np.quantile(pair_values, 0.25, axis=1),
            np.quantile(pair_values, 0.50, axis=1),
            np.quantile(pair_values, 0.75, axis=1),
            sums / 140.0,
            means / 20.0,
            standard_deviations / 12.0,
            ranges / 38.0,
            odd_counts / TOP_K,
            low_counts / TOP_K,
            consecutive_counts / 6.0,
            gaps_between_numbers.mean(axis=1) / 7.0,
            gaps_between_numbers.std(axis=1) / 7.0,
        ]
    )

    if not precise:
        return compact.astype(np.float32)

    sorted_number_values = np.sort(
        number_values,
        axis=1,
    )

    sorted_pair_values = np.sort(
        pair_values,
        axis=1,
    )

    return np.column_stack(
        [
            compact,
            sorted_number_values,
            sorted_pair_values,
        ]
    ).astype(np.float32)


# ============================================================
# KONTINUIRANA KOMBINACIJSKA META
# ============================================================

def combination_targets(
    candidates: np.ndarray,
    t: int,
    draws: np.ndarray,
    number_waits: np.ndarray,
    pair_waits: np.ndarray,
) -> np.ndarray:
    zero_based = candidates - 1

    candidate_number_waits = number_waits[
        t,
        zero_based,
    ]

    pair_ids = PAIR_INDEX[
        zero_based[:, LOCAL_PAIR_LEFT],
        zero_based[:, LOCAL_PAIR_RIGHT],
    ]

    candidate_pair_waits = pair_waits[
        t,
        pair_ids,
    ]

    number_known = np.all(
        np.isfinite(candidate_number_waits),
        axis=1,
    )

    pair_known = np.all(
        np.isfinite(candidate_pair_waits),
        axis=1,
    )

    valid = number_known & pair_known

    targets = np.full(
        len(candidates),
        np.nan,
        dtype=np.float32,
    )

    if not np.any(valid):
        return targets

    number_component = np.mean(
        np.log(
            BASE_WAIT_NUMBER
            / candidate_number_waits[valid]
        ),
        axis=1,
    )

    pair_component = np.mean(
        np.log(
            BASE_WAIT_PAIR
            / candidate_pair_waits[valid]
        ),
        axis=1,
    )

    horizon = min(
        OVERLAP_HORIZON,
        len(draws) - t,
    )

    overlap_component = np.zeros(
        np.sum(valid),
        dtype=float,
    )

    weight_sum = 0.0
    valid_candidates = candidates[valid]

    for offset in range(horizon):
        weight = OVERLAP_DECAY ** offset

        actual = draws[t + offset]

        hits = np.isin(
            valid_candidates,
            actual,
        ).sum(axis=1)

        overlap_component += weight * (
            hits / TOP_K - BASE_RATE
        )

        weight_sum += weight

    if weight_sum > 0:
        overlap_component /= weight_sum

    targets[valid] = (
        TARGET_NUMBER_WEIGHT * number_component
        + TARGET_PAIR_WEIGHT * pair_component
        + TARGET_OVERLAP_WEIGHT * overlap_component
    ).astype(np.float32)

    return targets


# ============================================================
# KANDIDATI HRONOLOŠKE GRUPE
# ============================================================

def stable_unique_combinations(
    candidates: list[tuple[int, ...]],
) -> np.ndarray:
    unique = sorted(set(candidates))

    return np.asarray(
        unique,
        dtype=np.int16,
    )


def create_group_candidates(
    draws: np.ndarray,
    t: int,
    number_scores: np.ndarray,
) -> np.ndarray:
    actual = tuple(
        sorted(int(value) for value in draws[t])
    )

    candidates: list[tuple[int, ...]] = [actual]

    actual_set = set(actual)

    ranking = np.lexsort(
        (
            np.arange(1, NUMBER_COUNT + 1),
            -number_scores,
        )
    ) + 1

    replacements = [
        int(number)
        for number in ranking
        if int(number) not in actual_set
    ][:NEAR_MISS_REPLACEMENTS]

    # Sve near-miss kombinacije dobijene jednom zamenom.
    for removed in actual:
        for replacement in replacements:
            changed = sorted(
                (actual_set - {removed}) | {replacement}
            )

            candidates.append(tuple(changed))

    # Kandidati pronađeni na osnovu tadašnjih skorova.
    top_pool = sorted(
        int(number)
        for number in ranking[:RETRIEVAL_POOL]
    )

    retrieved = list(
        combinations(top_pool, TOP_K)
    )

    if len(retrieved) > RETRIEVED_PER_GROUP:
        retrieval_array = np.asarray(
            retrieved,
            dtype=np.int16,
        )

        retrieval_scores = number_scores[
            retrieval_array - 1
        ].mean(axis=1)

        order = stable_descending_order(
            retrieval_scores,
            retrieval_array,
        )

        retrieved = [
            tuple(int(value) for value in retrieval_array[index])
            for index in order[:RETRIEVED_PER_GROUP]
        ]

    candidates.extend(retrieved)

    # Stvarne prethodne kombinacije kao istorijski kandidati.
    history_start = max(
        0,
        t - HISTORICAL_CANDIDATES,
    )

    for historical_draw in draws[history_start:t]:
        candidates.append(
            tuple(
                sorted(
                    int(value)
                    for value in historical_draw
                )
            )
        )

    return stable_unique_combinations(candidates)


# ============================================================
# STABILNO RANGIRANJE
# ============================================================

def stable_descending_order(
    scores: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Primarno: veći skor.
    Kod jednakog skora: leksikografski manja kombinacija.
    """

    keys: list[np.ndarray] = []

    for column in range(TOP_K - 1, -1, -1):
        keys.append(candidates[:, column])

    keys.append(-np.asarray(scores, dtype=float))

    return np.lexsort(tuple(keys))


def keep_best(
    candidates: np.ndarray,
    scores: np.ndarray,
    keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = min(keep, len(candidates))

    if keep <= 0:
        raise ValueError("Broj zadržanih kandidata mora biti pozitivan.")

    if len(candidates) > keep:
        partial = np.argpartition(
            scores,
            -keep,
        )[-keep:]

        candidates = candidates[partial]
        scores = scores[partial]

    order = stable_descending_order(
        scores,
        candidates,
    )

    return candidates[order], scores[order]


# ============================================================
# GRUPNI REGRESIONI PODACI
# ============================================================

def choose_group_times(
    total_draws: int,
) -> np.ndarray:
    """
    Grupe ne počinju odmah na MIN_HISTORY.

    Dodatni početni period služi za pravljenje potpuno
    razrešenih kontinuiranih meta i obuku početnog
    regresora brojeva pre prve kombinacijske grupe.
    """

    number_model_warmup = 300

    first_allowed = (
        MIN_HISTORY + number_model_warmup
    )

    last_allowed = (
        total_draws - MIN_FUTURE_FOR_GROUP
    )

    if last_allowed <= first_allowed:
        raise ValueError(
            "Nema dovoljno istorije za početnu obuku, "
            "grupnu validaciju i buduće mete."
        )

    available = (
        last_allowed - first_allowed
    )

    group_count = min(
        REQUESTED_GROUPS,
        available,
    )

    times = np.linspace(
        first_allowed,
        last_allowed - 1,
        num=group_count,
        dtype=np.int32,
    )

    return np.unique(times)


def build_group_dataset(
    draws: np.ndarray,
    number_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    number_waits: np.ndarray,
    pair_waits: np.ndarray,
    number_model: ExtraTreesRegressor,
    group_times: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    fast_features: list[np.ndarray] = []
    precise_features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    combinations_parts: list[np.ndarray] = []

    for group_id, t in enumerate(group_times):
        raw_number_features = number_features_at(
            number_matrix,
            int(t),
        )

        number_scores = number_model.predict(
            raw_number_features
        ).astype(np.float32)

        pair_scores = pair_scores_at(
            number_matrix,
            pair_matrix,
            int(t),
        )

        candidates = create_group_candidates(
            draws,
            int(t),
            number_scores,
        )

        target = combination_targets(
            candidates,
            int(t),
            draws,
            number_waits,
            pair_waits,
        )

        valid = np.isfinite(target)

        if np.sum(valid) < 10:
            continue

        candidates = candidates[valid]
        target = target[valid]

        fast = combination_features(
            candidates,
            number_scores,
            pair_scores,
            precise=False,
        )

        precise = combination_features(
            candidates,
            number_scores,
            pair_scores,
            precise=True,
        )

        fast_features.append(fast)
        precise_features.append(precise)
        targets.append(target)
        combinations_parts.append(candidates)

        groups.append(
            np.full(
                len(candidates),
                group_id,
                dtype=np.int16,
            )
        )

    if not fast_features:
        raise RuntimeError(
            "Nije napravljena nijedna važeća hronološka grupa."
        )

    return (
        np.vstack(fast_features),
        np.vstack(precise_features),
        np.concatenate(targets),
        np.concatenate(groups),
        np.vstack(combinations_parts),
    )


# ============================================================
# MODELI I VREMENSKI IZBOR PARAMETARA
# ============================================================

FAST_PARAMETER_SETS = (
    {
        "learning_rate": 0.045,
        "max_iter": 260,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.20,
    },
    {
        "learning_rate": 0.035,
        "max_iter": 340,
        "max_leaf_nodes": 23,
        "min_samples_leaf": 25,
        "l2_regularization": 0.50,
    },
    {
        "learning_rate": 0.025,
        "max_iter": 420,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.00,
    },
)

PRECISE_PARAMETER_SETS = (
    {
        "n_estimators": 420,
        "min_samples_leaf": 3,
        "max_features": 0.80,
    },
    {
        "n_estimators": 520,
        "min_samples_leaf": 4,
        "max_features": 0.90,
    },
    {
        "n_estimators": 620,
        "min_samples_leaf": 5,
        "max_features": 1.00,
    },
)


def create_fast_model(
    parameters: dict,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=parameters["learning_rate"],
        max_iter=parameters["max_iter"],
        max_leaf_nodes=parameters["max_leaf_nodes"],
        min_samples_leaf=parameters["min_samples_leaf"],
        l2_regularization=parameters["l2_regularization"],
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def create_precise_model(
    parameters: dict,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=parameters["n_estimators"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def temporal_parameter_selection(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    stage: str,
) -> dict:
    unique_groups = np.unique(groups)

    inner_validation_count = max(
        1,
        int(round(len(unique_groups) * 0.20)),
    )

    inner_train_groups = unique_groups[
        :-inner_validation_count
    ]

    inner_validation_groups = unique_groups[
        -inner_validation_count:
    ]

    train_mask = np.isin(
        groups,
        inner_train_groups,
    )

    validation_mask = np.isin(
        groups,
        inner_validation_groups,
    )

    if stage == "fast":
        parameter_sets = FAST_PARAMETER_SETS
        factory = create_fast_model
    elif stage == "precise":
        parameter_sets = PRECISE_PARAMETER_SETS
        factory = create_precise_model
    else:
        raise ValueError(f"Nepoznata faza: {stage}")

    best_parameters = parameter_sets[0]
    best_mae = float("inf")

    for parameters in parameter_sets:
        model = factory(parameters)

        model.fit(
            features[train_mask],
            targets[train_mask],
        )

        prediction = model.predict(
            features[validation_mask]
        )

        mae = mean_absolute_error(
            targets[validation_mask],
            prediction,
        )

        if mae < best_mae - EPSILON:
            best_mae = float(mae)
            best_parameters = parameters

    return dict(best_parameters)


# ============================================================
# VALIDACIJA BRZE I PRECIZNE FAZE
# ============================================================

def evaluate_validation(
    fast_model,
    precise_model,
    fast_features: np.ndarray,
    precise_features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    candidates: np.ndarray,
    validation_groups: np.ndarray,
    draws: np.ndarray,
    group_times: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    fast_predictions = fast_model.predict(
        fast_features
    )

    precise_predictions = precise_model.predict(
        precise_features
    )

    validation_mask = np.isin(
        groups,
        validation_groups,
    )

    fast_mae = mean_absolute_error(
        targets[validation_mask],
        fast_predictions[validation_mask],
    )

    precise_mae = mean_absolute_error(
        targets[validation_mask],
        precise_predictions[validation_mask],
    )

    fast_successes = 0
    precise_successes = 0

    fast_hits: list[int] = []
    precise_hits: list[int] = []

    for group_id in validation_groups:
        mask = groups == group_id

        group_candidates = candidates[mask]
        group_targets = targets[mask]
        group_fast = fast_predictions[mask]
        group_precise = precise_predictions[mask]

        true_order = stable_descending_order(
            group_targets,
            group_candidates,
        )

        best_true_combination = group_candidates[
            true_order[0]
        ]

        fast_order = stable_descending_order(
            group_fast,
            group_candidates,
        )

        fast_top_recall = group_candidates[
            fast_order[:min(RECALL_K, len(fast_order))]
        ]

        if np.any(
            np.all(
                fast_top_recall == best_true_combination,
                axis=1,
            )
        ):
            fast_successes += 1

        fast_shortlist_indices = fast_order[
            :min(
                VALIDATION_FAST_KEEP,
                len(fast_order),
            )
        ]

        shortlist_candidates = group_candidates[
            fast_shortlist_indices
        ]

        shortlist_precise_scores = group_precise[
            fast_shortlist_indices
        ]

        precise_order = stable_descending_order(
            shortlist_precise_scores,
            shortlist_candidates,
        )

        precise_top_recall = shortlist_candidates[
            precise_order[
                :min(RECALL_K, len(precise_order))
            ]
        ]

        if np.any(
            np.all(
                precise_top_recall == best_true_combination,
                axis=1,
            )
        ):
            precise_successes += 1

        fast_choice = group_candidates[
            fast_order[0]
        ]

        precise_choice = shortlist_candidates[
            precise_order[0]
        ]

        # group_id odgovara redosledu uspešno napravljenih grupa.
        time_position = min(
            int(group_id),
            len(group_times) - 1,
        )

        actual_draw = draws[
            int(group_times[time_position])
        ]

        fast_hits.append(
            int(np.isin(fast_choice, actual_draw).sum())
        )

        precise_hits.append(
            int(np.isin(precise_choice, actual_draw).sum())
        )

    denominator = max(len(validation_groups), 1)

    fast_recall = fast_successes / denominator
    precise_recall = precise_successes / denominator

    return (
        float(fast_mae),
        float(precise_mae),
        float(fast_recall),
        float(precise_recall),
        np.asarray(fast_hits, dtype=float),
        np.asarray(precise_hits, dtype=float),
    )


# ============================================================
# STATISTIČKA PROCENA HOLDOUT POGODAKA
# ============================================================

def bootstrap_difference_interval(
    hits: np.ndarray,
) -> tuple[float, float]:
    if len(hits) == 0:
        return float("nan"), float("nan")

    differences = hits - RANDOM_EXPECTED_HITS

    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.integers(
        0,
        len(differences),
        size=(
            STATISTICAL_SIMULATIONS,
            len(differences),
        ),
    )

    means = differences[sample_indices].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def sign_flip_permutation_test(
    hits: np.ndarray,
) -> float:
    if len(hits) == 0:
        return float("nan")

    differences = hits - RANDOM_EXPECTED_HITS
    observed = abs(float(differences.mean()))

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(
            STATISTICAL_SIMULATIONS,
            len(differences),
        ),
    )

    null_means = np.abs(
        (signs * differences).mean(axis=1)
    )

    return float(
        (
            np.sum(null_means >= observed) + 1
        )
        / (
            STATISTICAL_SIMULATIONS + 1
        )
    )


# ============================================================
# SVE KOMBINACIJE U PAKETIMA
# ============================================================

def combination_batches(
    batch_size: int,
) -> Iterator[np.ndarray]:
    iterator = combinations(
        range(MIN_NUMBER, MAX_NUMBER + 1),
        TOP_K,
    )

    while True:
        batch_list = list(
            islice(iterator, batch_size)
        )

        if not batch_list:
            break

        yield np.asarray(
            batch_list,
            dtype=np.int16,
        )


def exhaustive_fast_search(
    fast_model,
    number_scores: np.ndarray,
    pair_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    best_candidates = np.empty(
        (0, TOP_K),
        dtype=np.int16,
    )

    best_scores = np.empty(
        0,
        dtype=np.float64,
    )

    checked = 0

    for batch_number, batch in enumerate(
        combination_batches(
            COMBINATION_BATCH_SIZE
        ),
        start=1,
    ):
        features = combination_features(
            batch,
            number_scores,
            pair_scores,
            precise=False,
        )

        scores = fast_model.predict(features)
        checked += len(batch)

        combined_candidates = np.vstack(
            [best_candidates, batch]
        )

        combined_scores = np.concatenate(
            [best_scores, scores]
        )

        best_candidates, best_scores = keep_best(
            combined_candidates,
            combined_scores,
            FINAL_FAST_KEEP,
        )

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or checked == TOTAL_COMBINATIONS
        ):
            percentage = (
                checked / TOTAL_COMBINATIONS * 100.0
            )

            print(
                f"  Pregledano: {checked:,} / "
                f"{TOTAL_COMBINATIONS:,} "
                f"({percentage:6.2f}%)"
            )

    return best_candidates, best_scores, checked


# ============================================================
# OBRADA JEDNE IGRE
# ============================================================

def process_game(
    name: str,
    csv_path: Path,
) -> GameResult:
    start_time = time.perf_counter()

    print()
    print("=" * 72)
    print(f"Obrada: {name}")
    print("=" * 72)

    draws = load_lottery_csv(csv_path)

    print(f"CSV: {csv_path}")
    print(f"Broj redova: {len(draws)}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")

    number_matrix = create_number_matrix(draws)
    pair_matrix = create_pair_matrix(draws)

    print("Pravljenje budućih time-to-event meta...")

    number_waits = next_wait_matrix(
        number_matrix
    )

    pair_waits = next_wait_matrix(
        pair_matrix
    )

    group_times = choose_group_times(
        len(draws)
    )

    validation_group_count = max(
        1,
        int(
            round(
                len(group_times)
                * VALIDATION_FRACTION
            )
        ),
    )

    training_group_count = (
        len(group_times)
        - validation_group_count
    )

    if training_group_count < 10:
        raise RuntimeError(
            f"{name}: premalo grupa za obuku."
        )

    first_group_time = int(group_times[0])

    print("Obuka početnog regresora brojeva...")

    initial_number_x, initial_number_y = (
        build_number_regression_data(
            number_matrix,
            number_waits,
            end_limit=first_group_time,
            step=2,
        )
    )

    initial_number_model = create_number_regressor(
        RANDOM_SEED
    )

    initial_number_model.fit(
        initial_number_x,
        initial_number_y,
    )

    print("Pravljenje hronoloških kombinacijskih grupa...")

    (
        fast_x,
        precise_x,
        combo_y,
        groups,
        group_candidates,
    ) = build_group_dataset(
        draws,
        number_matrix,
        pair_matrix,
        number_waits,
        pair_waits,
        initial_number_model,
        group_times,
    )

    unique_groups = np.unique(groups)

    validation_group_count = max(
        1,
        int(
            round(
                len(unique_groups)
                * VALIDATION_FRACTION
            )
        ),
    )

    train_groups = unique_groups[
        :-validation_group_count
    ]

    validation_groups = unique_groups[
        -validation_group_count:
    ]

    train_mask = np.isin(
        groups,
        train_groups,
    )

    print(
        f"Hronoloških grupa za obuku: "
        f"{len(train_groups)}"
    )

    print(
        f"Odvojenih validacionih grupa: "
        f"{len(validation_groups)}"
    )

    print("Vremenski izbor parametara brze faze...")

    fast_parameters = temporal_parameter_selection(
        fast_x[train_mask],
        combo_y[train_mask],
        groups[train_mask],
        stage="fast",
    )

    print("Vremenski izbor parametara precizne faze...")

    precise_parameters = temporal_parameter_selection(
        precise_x[train_mask],
        combo_y[train_mask],
        groups[train_mask],
        stage="precise",
    )

    fast_validation_model = create_fast_model(
        fast_parameters
    )

    precise_validation_model = create_precise_model(
        precise_parameters
    )

    fast_validation_model.fit(
        fast_x[train_mask],
        combo_y[train_mask],
    )

    precise_validation_model.fit(
        precise_x[train_mask],
        combo_y[train_mask],
    )

    (
        fast_mae,
        precise_mae,
        fast_recall,
        precise_recall,
        fast_hits,
        precise_hits,
    ) = evaluate_validation(
        fast_validation_model,
        precise_validation_model,
        fast_x,
        precise_x,
        combo_y,
        groups,
        group_candidates,
        validation_groups,
        draws,
        group_times,
    )

    use_precise = (
        precise_mae < fast_mae
        and precise_recall >= fast_recall
    )

    selected_stage = (
        "precise" if use_precise else "fast"
    )

    selected_hits = (
        precise_hits if use_precise else fast_hits
    )

    print(f"Fast MAE: {fast_mae:.6f}")
    print(f"Precise MAE: {precise_mae:.6f}")
    print(f"Fast recall@{RECALL_K}: {fast_recall:.1%}")
    print(
        f"Precise recall@{RECALL_K}: "
        f"{precise_recall:.1%}"
    )
    print(f"Izabrana završna faza: {selected_stage}")

    # Završni modeli koriste sve raspoložive hronološke grupe.
    final_fast_model = create_fast_model(
        fast_parameters
    )

    final_precise_model = create_precise_model(
        precise_parameters
    )

    final_fast_model.fit(
        fast_x,
        combo_y,
    )

    final_precise_model.fit(
        precise_x,
        combo_y,
    )

    # Završni regresor brojeva koristi sve razrešene istorijske mete.
    print("Obuka završnog regresora brojeva...")

    final_number_x, final_number_y = (
        build_number_regression_data(
            number_matrix,
            number_waits,
            end_limit=len(draws),
            step=2,
        )
    )

    final_number_model = create_number_regressor(
        RANDOM_SEED + 10
    )

    final_number_model.fit(
        final_number_x,
        final_number_y,
    )

    current_number_features = number_features_at(
        number_matrix,
        len(draws),
    )

    current_number_scores = final_number_model.predict(
        current_number_features
    ).astype(np.float32)

    current_pair_scores = pair_scores_at(
        number_matrix,
        pair_matrix,
        len(draws),
    )

    print(
        "Paketni pregled svih "
        f"{TOTAL_COMBINATIONS:,} kombinacija..."
    )

    (
        finalists,
        finalist_fast_scores,
        combinations_checked,
    ) = exhaustive_fast_search(
        final_fast_model,
        current_number_scores,
        current_pair_scores,
    )

    if combinations_checked != TOTAL_COMBINATIONS:
        raise RuntimeError(
            f"Pregledano je {combinations_checked:,}, "
            f"a očekivano {TOTAL_COMBINATIONS:,} kombinacija."
        )

    if use_precise:
        precise_final_features = combination_features(
            finalists,
            current_number_scores,
            current_pair_scores,
            precise=True,
        )

        final_scores = final_precise_model.predict(
            precise_final_features
        )

        final_order = stable_descending_order(
            final_scores,
            finalists,
        )

        prediction = finalists[
            final_order[0]
        ]
    else:
        final_order = stable_descending_order(
            finalist_fast_scores,
            finalists,
        )

        prediction = finalists[
            final_order[0]
        ]

    holdout_average_hits = float(
        np.mean(selected_hits)
    )

    holdout_difference = (
        holdout_average_hits
        - RANDOM_EXPECTED_HITS
    )

    bootstrap_low, bootstrap_high = (
        bootstrap_difference_interval(
            selected_hits
        )
    )

    permutation_p = sign_flip_permutation_test(
        selected_hits
    )

    elapsed = time.perf_counter() - start_time

    return GameResult(
        name=name,
        prediction=prediction,
        rows=len(draws),
        combinations_checked=combinations_checked,
        training_groups=len(train_groups),
        validation_groups=len(validation_groups),
        fast_mae=fast_mae,
        precise_mae=precise_mae,
        fast_recall=fast_recall,
        precise_recall=precise_recall,
        selected_stage=selected_stage,
        holdout_average_hits=holdout_average_hits,
        holdout_difference=holdout_difference,
        bootstrap_low=bootstrap_low,
        bootstrap_high=bootstrap_high,
        permutation_p=permutation_p,
        elapsed_seconds=elapsed,
    )


# ============================================================
# ISPIS
# ============================================================

def format_combination(
    combination: np.ndarray,
) -> str:
    return ", ".join(
        f"{int(number):02d}"
        for number in combination
    )


def print_result(
    result: GameResult,
) -> None:
    print()
    print("=" * 72)
    print(result.name)
    print("=" * 72)

    print(
        "NEXT: "
        f"{format_combination(result.prediction)}"
    )

    print(f"CSV redova: {result.rows:,}")

    print(
        "Pregledano kombinacija: "
        f"{result.combinations_checked:,}"
    )

    print(
        "Hronoloških grupa za obuku: "
        f"{result.training_groups}"
    )

    print(
        "Odvojenih validacionih grupa: "
        f"{result.validation_groups}"
    )

    print(f"Fast MAE: {result.fast_mae:.6f}")
    print(f"Precise MAE: {result.precise_mae:.6f}")

    print(
        f"Fast recall@{RECALL_K}: "
        f"{result.fast_recall:.1%}"
    )

    print(
        f"Precise recall@{RECALL_K}: "
        f"{result.precise_recall:.1%}"
    )

    print(
        "Izabrana završna faza: "
        f"{result.selected_stage}"
    )

    print(
        "Holdout prosek pogodaka: "
        f"{result.holdout_average_hits:.6f}"
    )

    print(
        "Slučajno očekivanje: "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        "Razlika prema slučajnoj osnovi: "
        f"{result.holdout_difference:+.6f}"
    )

    print(
        "Bootstrap 95% interval razlike: "
        f"[{result.bootstrap_low:+.6f}, "
        f"{result.bootstrap_high:+.6f}]"
    )

    print(
        "Permutaciona p-vrednost: "
        f"{result.permutation_p:.6f}"
    )

    print(
        "Vreme obrade: "
        f"{result.elapsed_seconds / 60.0:.2f} minuta"
    )


# ============================================================
# GLAVNI PROGRAM
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("LOTO 7/39 — DISTRIBUCIJSKI REGRESIONI SISTEM")
    print("=" * 72)

    print(
        f"Teorijska osnovna stopa broja: "
        f"{BASE_RATE:.6f}"
    )

    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        f"Ukupno mogućih kombinacija: "
        f"{TOTAL_COMBINATIONS:,}"
    )

    loto_result = process_game(
        "Loto",
        LOTO_CSV,
    )

    loto_plus_result = process_game(
        "Loto Plus",
        LOTO_PLUS_CSV,
    )

    print()
    print()
    print("KONAČNE NEXT PREDIKCIJE")

    print_result(loto_result)
    print_result(loto_plus_result)

    print()
    print("=" * 72)
    print("NAPOMENA")
    print("=" * 72)

    print(
        "Isti postupak je pokrenut zasebno nad svakim CSV-om."
    )

    print(
        "Pregled svih kombinacija pronalazi najbolje ocenjen izbor "
        "prema obučenom modelu, ali ne garantuje dobitak."
    )


if __name__ == "__main__":
    main()



"""
========================================================================
LOTO 7/39 — DISTRIBUCIJSKI REGRESIONI SISTEM
========================================================================
Teorijska osnovna stopa broja: 0.179487
Teorijsko očekivanje pogodaka: 1.256410
Ukupno mogućih kombinacija: 15,380,937

========================================================================
Obrada: Loto
========================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4674_k68_loto_2959.csv
Broj redova: 2959
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Pravljenje budućih time-to-event meta...
Obuka početnog regresora brojeva...
Pravljenje hronoloških kombinacijskih grupa...
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Vremenski izbor parametara brze faze...
Vremenski izbor parametara precizne faze...
Fast MAE: 0.296960
Precise MAE: 0.319711
Fast recall@10: 0.0%
Precise recall@10: 5.0%
Izabrana završna faza: fast
Obuka završnog regresora brojeva...
Paketni pregled svih 15,380,937 kombinacija...
  Pregledano: 150,000 / 15,380,937 (  0.98%)
  Pregledano: 1,500,000 / 15,380,937 (  9.75%)
  Pregledano: 3,000,000 / 15,380,937 ( 19.50%)
  Pregledano: 4,500,000 / 15,380,937 ( 29.26%)
  Pregledano: 6,000,000 / 15,380,937 ( 39.01%)
  Pregledano: 7,500,000 / 15,380,937 ( 48.76%)
  Pregledano: 9,000,000 / 15,380,937 ( 58.51%)
  Pregledano: 10,500,000 / 15,380,937 ( 68.27%)
  Pregledano: 12,000,000 / 15,380,937 ( 78.02%)
  Pregledano: 13,500,000 / 15,380,937 ( 87.77%)
  Pregledano: 15,000,000 / 15,380,937 ( 97.52%)
  Pregledano: 15,380,937 / 15,380,937 (100.00%)

========================================================================
Obrada: Loto Plus
========================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4674_k68_loto_plus_1715.csv
Broj redova: 1715
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Pravljenje budućih time-to-event meta...
Obuka početnog regresora brojeva...
Pravljenje hronoloških kombinacijskih grupa...
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Vremenski izbor parametara brze faze...
Vremenski izbor parametara precizne faze...
Fast MAE: 0.280916
Precise MAE: 0.285065
Fast recall@10: 30.0%
Precise recall@10: 15.0%
Izabrana završna faza: fast
Obuka završnog regresora brojeva...
Paketni pregled svih 15,380,937 kombinacija...
  Pregledano: 150,000 / 15,380,937 (  0.98%)
  Pregledano: 1,500,000 / 15,380,937 (  9.75%)
  Pregledano: 3,000,000 / 15,380,937 ( 19.50%)
  Pregledano: 4,500,000 / 15,380,937 ( 29.26%)
  Pregledano: 6,000,000 / 15,380,937 ( 39.01%)
  Pregledano: 7,500,000 / 15,380,937 ( 48.76%)
  Pregledano: 9,000,000 / 15,380,937 ( 58.51%)
  Pregledano: 10,500,000 / 15,380,937 ( 68.27%)
  Pregledano: 12,000,000 / 15,380,937 ( 78.02%)
  Pregledano: 13,500,000 / 15,380,937 ( 87.77%)
  Pregledano: 15,000,000 / 15,380,937 ( 97.52%)
  Pregledano: 15,380,937 / 15,380,937 (100.00%)


KONAČNE NEXT PREDIKCIJE

========================================================================
Loto
========================================================================
NEXT: 02, 03, 05, 11, 15, 36, 37
CSV redova: 2,959
Pregledano kombinacija: 15,380,937
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Fast MAE: 0.296960
Precise MAE: 0.319711
Fast recall@10: 0.0%
Precise recall@10: 5.0%
Izabrana završna faza: fast
Holdout prosek pogodaka: 5.550000
Slučajno očekivanje: 1.256410
Razlika prema slučajnoj osnovi: +4.293590
Bootstrap 95% interval razlike: [+3.593590, +4.743590]
Permutaciona p-vrednost: 0.000100
Vreme obrade: 1.74 minuta

========================================================================
Loto Plus
========================================================================
NEXT: 11, 23, 24, 25, 30, 32, 34
CSV redova: 1,715
Pregledano kombinacija: 15,380,937
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Fast MAE: 0.280916
Precise MAE: 0.285065
Fast recall@10: 30.0%
Precise recall@10: 15.0%
Izabrana završna faza: fast
Holdout prosek pogodaka: 4.800000
Slučajno očekivanje: 1.256410
Razlika prema slučajnoj osnovi: +3.543590
Bootstrap 95% interval razlike: [+2.543590, +4.493590]
Permutaciona p-vrednost: 0.000200
Vreme obrade: 1.92 minuta

========================================================================
NAPOMENA
========================================================================
Isti postupak je pokrenut zasebno nad svakim CSV-om.
Pregled svih kombinacija pronalazi najbolje ocenjen izbor prema obučenom modelu, ali ne garantuje dobitak.
"""



"""
Trenutni naučno ispravan odgovor je: 
nije dokazano da istorijska izvlačenja pružaju statistički pouzdane informacije o budućim izvlačenjima.

Pouzdan signal postoji tek ako zamrznuti model na dovoljno velikom broju potpuno novih izvlačenja istovremeno pokaže:
- prosek pogodaka iznad slučajnih 1,256410;
- bootstrap interval razlike čija je donja granica iznad nule;
- malu permutacionu p-vrednost, uobičajeno < 0,05;
- stabilan rezultat i za Loto i Loto Plus ili kroz više nezavisnih perioda;
- rezultat bez naknadnog menjanja modela prema novim ishodima.

Istorijski holdout može dati početnu indiciju, ali zbog mnogih isprobanih postavki nije konačan dokaz. 
Zato se odgovor donosi prema rezultatima zamrznute buduće OOS validacije, a ne prema samoj NEXT kombinaciji.


Za sada nema statistički pouzdanog dokaza da istorijska izvlačenja Lota 7/39 omogućavaju predviđanje budućih izvlačenja bolje od slučajnog izbora.
"""



"""
Smernice za dalji rad.

Imaju li drugi modeli da istorijski podaci o izvlačenjima igre Loto 7 od 39 sadrže statističku strukturu, 
vremenske obrasce ili prediktivne informacije koje se mogu upotrebiti za izbor kombinacija sa očekivanim brojem pogodaka većim od slučajnog? 

Postoje drugi modeli koji mogu tražiti takvu strukturu:
- Cox/AFT survival regresija — predviđa vreme do narednog pojavljivanja broja ili para.
- Gradient-boosted survival model — nelinearno povezuje gap, hazard, nedavnost i prelaze.
- Bajesov dinamički model stanja — traži promene verovatnoća brojeva kroz vreme, uz snažno smanjivanje procena prema osnovnoj stopi 7/39.
- Hidden Markov Model — ispituje postoje li različiti vremenski režimi izvlačenja.
- Change-point model — traži trenutke u kojima se statistička raspodela trajno promenila.
- Grafovski regresor — brojeve tretira kao čvorove, a istorijska ko-pojavljivanja i prelaze kao veze.
- Temporalni point-process model — modeluje vreme između pojavljivanja brojeva i parova.
- Gaussian Process regresija — traži glatke promene latentnih stopa kroz vreme.
- Regresioni stacking — spaja samo modele koji pojedinačno pokažu prednost na odvojenoj hronološkoj validaciji.

Za mojih približno 3.000 i 1.700 redova, najrazumniji sledeći izbor bio bi jedan sistem koji kombinuje:
1. gradient-boosted survival regresor;
2. Bajesov dinamički model;
3. grafovski regresor parova;
4. strogu nested walk-forward i zamrznutu buduću OOS proveru.

Neuronske mreže i veliki transformer modeli ovde nisu prvi izbor jer je broj stvarnih izvlačenja mali. 
Nijedan od navedenih modela ne može stvoriti signal ako ga u podacima nema; 
mogu samo pouzdanije proveriti postoji li.
"""
