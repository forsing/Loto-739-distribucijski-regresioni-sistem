#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Insipration/Upgarde - Inspiracija/Nadogradnja
# https://github.com/ahmad-al-hamza/Lotto_6_aus_49_ML_Lab


"""
LOTO 7/39 — FINALNI DISTRIBUCIJSKI REGRESIONI SISTEM

Implementirano:

1. Kontinuirana time-to-event meta za brojeve.
2. Kontinuirana time-to-event meta za parove.
3. Empirijske gap, survival i hazard raspodele.
4. Uslovni prelazi i odstupanje od osnovne stope 7/39.
5. Expanding walk-forward obuka regresora brojeva.
6. Expanding walk-forward obuka regresora parova.
7. Stvarne, near-miss i istorijski pronađene kombinacije.
8. Grupna hronološka obuka i odvojeni holdout.
9. Nested vremenski izbor parametara.
10. Ispravna procena precizne faze samo nad kandidatima
    koje je zadržala brza faza.
11. Paketni pregled svih 15.380.937 kombinacija.
12. Jedna NEXT predikcija za Loto i jedna za Loto Plus.
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
# CSV PUTANJE
# ============================================================

LOTO_CSV = Path(
    "/data/"
    "loto7_4674_k68_loto_2959.csv"
)

LOTO_PLUS_CSV = Path(
    "/data/"
    "loto7_4674_k68_loto_plus_1715.csv"
)


# ============================================================
# OSNOVNE POSTAVKE
# ============================================================

MIN_NUMBER = 1
MAX_NUMBER = 39
TOP_K = 7

NUMBER_COUNT = 39
PAIR_COUNT = math.comb(NUMBER_COUNT, 2)
TOTAL_COMBINATIONS = math.comb(NUMBER_COUNT, TOP_K)

BASE_RATE = TOP_K / NUMBER_COUNT
BASE_WAIT_NUMBER = NUMBER_COUNT / TOP_K

PAIR_RATE = (
    math.comb(TOP_K, 2)
    / math.comb(NUMBER_COUNT, 2)
)

BASE_WAIT_PAIR = 1.0 / PAIR_RATE

RANDOM_EXPECTED_HITS = (
    TOP_K * TOP_K / NUMBER_COUNT
)

RANDOM_SEED = 20260828

MIN_HISTORY = 200
NUMBER_MODEL_WARMUP = 300
MIN_FUTURE_FOR_GROUP = 250

WINDOWS = (10, 20, 50, 100, 200)

REQUESTED_GROUPS = 98
VALIDATION_FRACTION = 0.20

# Expanding modeli se ponovo obučavaju u svakoj grupi.
NUMBER_TRAINING_STEP = 5
PAIR_TRAINING_STEP = 8

RETRIEVAL_POOL = 12
RETRIEVED_PER_GROUP = 80

NEAR_MISS_REPLACEMENTS = 5
HISTORICAL_CANDIDATES = 30

VALIDATION_FAST_KEEP = 80
FINAL_FAST_KEEP = 50_000

RECALL_K = 10

COMBINATION_BATCH_SIZE = 150_000

TARGET_NUMBER_WEIGHT = 0.50
TARGET_PAIR_WEIGHT = 0.30
TARGET_OVERLAP_WEIGHT = 0.20

OVERLAP_HORIZON = 8
OVERLAP_DECAY = 0.82

TRANSITION_ALPHA = 20.0

STATISTICAL_SIMULATIONS = 10_000

EPSILON = 1e-12


# ============================================================
# PAROVI
# ============================================================

PAIR_LIST = np.asarray(
    list(
        combinations(
            range(NUMBER_COUNT),
            2,
        )
    ),
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
    [
        left
        for left, _ in combinations(
            range(TOP_K),
            2,
        )
    ],
    dtype=np.int8,
)

LOCAL_PAIR_RIGHT = np.asarray(
    [
        right
        for _, right in combinations(
            range(TOP_K),
            2,
        )
    ],
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
# UČITAVANJE CSV-A
# ============================================================

def load_lottery_csv(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV nije pronađen: {path}"
        )

    frame = pd.read_csv(
        path,
        header=None,
        sep=None,
        engine="python",
        dtype=str,
    )

    frame = frame.dropna(
        axis=0,
        how="all",
    )

    frame = frame.dropna(
        axis=1,
        how="all",
    )

    numeric_columns: list[pd.Series] = []

    for column in frame.columns:
        converted = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.strip()
            .str.replace(
                '"',
                "",
                regex=False,
            ),
            errors="coerce",
        )

        if converted.notna().all():
            numeric_columns.append(converted)

    if len(numeric_columns) != TOP_K:
        raise ValueError(
            f"{path}\n"
            f"Očekivano je 7 brojčanih kolona, "
            f"a pronađeno je {len(numeric_columns)}."
        )

    data = pd.concat(
        numeric_columns,
        axis=1,
    ).to_numpy(
        dtype=np.int16
    )

    minimum_required = (
        MIN_HISTORY
        + NUMBER_MODEL_WARMUP
        + MIN_FUTURE_FOR_GROUP
    )

    if len(data) <= minimum_required:
        raise ValueError(
            f"{path}\n"
            f"CSV ima {len(data)} redova, a potrebno je "
            f"više od {minimum_required}."
        )

    if (
        np.any(data < MIN_NUMBER)
        or np.any(data > MAX_NUMBER)
    ):
        bad_rows = np.flatnonzero(
            np.any(
                (data < MIN_NUMBER)
                | (data > MAX_NUMBER),
                axis=1,
            )
        )

        raise ValueError(
            f"Broj van opsega 1–39 pronađen je "
            f"u redu {int(bad_rows[0]) + 1}."
        )

    for row_index, row in enumerate(data):
        if len(np.unique(row)) != TOP_K:
            raise ValueError(
                f"Red {row_index + 1} nema sedam "
                f"različitih brojeva: {row.tolist()}"
            )

    return np.sort(
        data,
        axis=1,
    )


# ============================================================
# MATRICE POJAVLJIVANJA
# ============================================================

def create_number_matrix(
    draws: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.int8,
    )

    rows = np.repeat(
        np.arange(len(draws)),
        TOP_K,
    )

    columns = draws.reshape(-1) - 1

    matrix[rows, columns] = 1

    return matrix


def create_pair_matrix(
    draws: np.ndarray,
) -> np.ndarray:
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


def create_next_wait_matrix(
    presence: np.ndarray,
) -> np.ndarray:
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

    for t in range(
        rows - 1,
        -1,
        -1,
    ):
        appeared = presence[t].astype(bool)

        next_position[appeared] = t

        known = next_position < rows

        waits[t, known] = (
            next_position[known] - t + 1
        )

    return waits


# ============================================================
# POMOĆNE DISTRIBUCIJSKE FUNKCIJE
# ============================================================

def normalized_entropy(
    values: np.ndarray,
) -> float:
    if len(values) < 2:
        return 0.0

    integer_values = np.maximum(
        values.astype(np.int32),
        0,
    )

    counts = np.bincount(
        integer_values
    )

    probabilities = counts[
        counts > 0
    ].astype(float)

    probabilities /= probabilities.sum()

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities + EPSILON
        )
    )

    maximum = math.log(
        max(
            len(probabilities),
            2,
        )
    )

    return float(
        entropy / maximum
    )


def distribution_features(
    presence: np.ndarray,
    end_index: int,
    base_rate: float,
    base_wait: float,
) -> np.ndarray:
    """
    Opšte distribucijske osobine za brojeve ili parove.

    Koristi samo redove [0:end_index].
    """

    history = presence[:end_index]
    total = len(history)
    entity_count = history.shape[1]

    if total < 1:
        raise ValueError(
            "Istorija ne sme biti prazna."
        )

    full_rate = (
        history.mean(axis=0)
    )

    window_rates: list[np.ndarray] = []

    for window in WINDOWS:
        part = history[
            -min(window, total):
        ]

        window_rates.append(
            part.mean(axis=0)
        )

    (
        rate_10,
        rate_20,
        rate_50,
        rate_100,
        rate_200,
    ) = window_rates

    window_matrix = np.vstack(
        [
            rate_20,
            rate_50,
            rate_100,
            rate_200,
        ]
    )

    window_mean = window_matrix.mean(
        axis=0
    )

    window_std = window_matrix.std(
        axis=0
    )

    current_gap = np.empty(
        entity_count,
        dtype=float,
    )

    q10 = np.empty(entity_count)
    q25 = np.empty(entity_count)
    q50 = np.empty(entity_count)
    q75 = np.empty(entity_count)
    q90 = np.empty(entity_count)

    cdf = np.empty(entity_count)
    survival = np.empty(entity_count)
    hazard = np.empty(entity_count)
    entropy = np.empty(entity_count)

    for entity in range(entity_count):
        positions = np.flatnonzero(
            history[:, entity]
        )

        if len(positions) == 0:
            current = total

            completed_gaps = np.asarray(
                [base_wait],
                dtype=float,
            )
        else:
            current = (
                total - int(positions[-1])
            )

            if len(positions) >= 2:
                completed_gaps = np.diff(
                    positions
                ).astype(float)
            else:
                completed_gaps = np.asarray(
                    [base_wait],
                    dtype=float,
                )

        current_gap[entity] = current

        quantiles = np.quantile(
            completed_gaps,
            [
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ],
        )

        q10[entity] = quantiles[0]
        q25[entity] = quantiles[1]
        q50[entity] = quantiles[2]
        q75[entity] = quantiles[3]
        q90[entity] = quantiles[4]

        cdf[entity] = np.mean(
            completed_gaps <= current
        )

        survival[entity] = np.mean(
            completed_gaps >= current
        )

        events = np.sum(
            completed_gaps.astype(np.int32)
            == int(current)
        )

        at_risk = np.sum(
            completed_gaps >= current
        )

        hazard[entity] = (
            events + 1.0
        ) / (
            at_risk + 2.0
        )

        entropy[entity] = normalized_entropy(
            completed_gaps
        )

    return np.column_stack(
        [
            full_rate,
            full_rate - base_rate,
            full_rate / max(
                base_rate,
                EPSILON,
            ),
            rate_10,
            rate_20,
            rate_50,
            rate_100,
            rate_200,
            rate_20 - rate_50,
            rate_50 - rate_200,
            window_mean,
            window_std,
            current_gap,
            current_gap / base_wait,
            q10,
            q25,
            q50,
            q75,
            q90,
            cdf,
            survival,
            hazard,
            entropy,
        ]
    ).astype(np.float32)


# ============================================================
# OSOBINE BROJEVA
# ============================================================

def number_features_at(
    number_matrix: np.ndarray,
    end_index: int,
) -> np.ndarray:
    basic = distribution_features(
        number_matrix,
        end_index,
        BASE_RATE,
        BASE_WAIT_NUMBER,
    )

    history = number_matrix[:end_index]

    transition_deviation = np.zeros(
        NUMBER_COUNT,
        dtype=float,
    )

    if len(history) >= 2:
        previous_numbers = np.flatnonzero(
            history[-1]
        )

        accumulated = np.zeros(
            NUMBER_COUNT,
            dtype=float,
        )

        for source in previous_numbers:
            source_mask = history[
                :-1,
                source,
            ].astype(bool)

            source_count = int(
                source_mask.sum()
            )

            if source_count == 0:
                conditional = np.full(
                    NUMBER_COUNT,
                    BASE_RATE,
                    dtype=float,
                )
            else:
                following_counts = history[
                    1:
                ][source_mask].sum(axis=0)

                conditional = (
                    following_counts
                    + TRANSITION_ALPHA
                    * BASE_RATE
                ) / (
                    source_count
                    + TRANSITION_ALPHA
                )

            accumulated += (
                conditional - BASE_RATE
            )

        if len(previous_numbers) > 0:
            transition_deviation = (
                accumulated
                / len(previous_numbers)
            )

    return np.column_stack(
        [
            basic,
            transition_deviation,
        ]
    ).astype(np.float32)


# ============================================================
# OSOBINE PAROVA
# ============================================================

def pair_features_at(
    number_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    end_index: int,
) -> np.ndarray:
    basic = distribution_features(
        pair_matrix,
        end_index,
        PAIR_RATE,
        BASE_WAIT_PAIR,
    )

    history_numbers = number_matrix[
        :end_index
    ]

    history_pairs = pair_matrix[
        :end_index
    ]

    number_counts = history_numbers.sum(
        axis=0
    ).astype(float)

    pair_counts = history_pairs.sum(
        axis=0
    ).astype(float)

    conditional_left = np.empty(
        PAIR_COUNT,
        dtype=float,
    )

    conditional_right = np.empty(
        PAIR_COUNT,
        dtype=float,
    )

    cooccurrence_deviation = np.empty(
        PAIR_COUNT,
        dtype=float,
    )

    for pair_id, (left, right) in enumerate(
        PAIR_LIST
    ):
        left_probability = (
            pair_counts[pair_id]
            + TRANSITION_ALPHA
            * BASE_RATE
        ) / (
            number_counts[left]
            + TRANSITION_ALPHA
        )

        right_probability = (
            pair_counts[pair_id]
            + TRANSITION_ALPHA
            * BASE_RATE
        ) / (
            number_counts[right]
            + TRANSITION_ALPHA
        )

        conditional_left[pair_id] = (
            left_probability
            - BASE_RATE
        )

        conditional_right[pair_id] = (
            right_probability
            - BASE_RATE
        )

        cooccurrence_deviation[pair_id] = (
            0.5
            * (
                left_probability
                + right_probability
            )
            - BASE_RATE
        )

    return np.column_stack(
        [
            basic,
            conditional_left,
            conditional_right,
            cooccurrence_deviation,
        ]
    ).astype(np.float32)


# ============================================================
# REGRESIONI PODACI BROJEVA I PAROVA
# ============================================================

def build_entity_regression_data(
    feature_function,
    waits: np.ndarray,
    end_limit: int,
    base_wait: float,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for t in range(
        MIN_HISTORY,
        end_limit,
        step,
    ):
        current_waits = waits[t]

        resolution_position = (
            t + current_waits - 1
        )

        known = (
            np.isfinite(current_waits)
            & (
                resolution_position
                < end_limit
            )
        )

        if not np.any(known):
            continue

        features = feature_function(t)

        targets = np.log(
            base_wait
            / current_waits[known]
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
            "kontinuiranih meta."
        )

    return (
        np.vstack(
            feature_parts
        ).astype(np.float32),
        np.concatenate(
            target_parts
        ).astype(np.float32),
    )


def create_entity_regressor(
    random_state: int,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.045,
        max_iter=260,
        max_leaf_nodes=23,
        min_samples_leaf=30,
        l2_regularization=0.50,
        early_stopping=False,
        random_state=random_state,
    )


def train_expanding_entity_models(
    number_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    number_waits: np.ndarray,
    pair_waits: np.ndarray,
    end_limit: int,
    random_state: int,
):
    number_x, number_y = (
        build_entity_regression_data(
            feature_function=lambda t: (
                number_features_at(
                    number_matrix,
                    t,
                )
            ),
            waits=number_waits,
            end_limit=end_limit,
            base_wait=BASE_WAIT_NUMBER,
            step=NUMBER_TRAINING_STEP,
        )
    )

    pair_x, pair_y = (
        build_entity_regression_data(
            feature_function=lambda t: (
                pair_features_at(
                    number_matrix,
                    pair_matrix,
                    t,
                )
            ),
            waits=pair_waits,
            end_limit=end_limit,
            base_wait=BASE_WAIT_PAIR,
            step=PAIR_TRAINING_STEP,
        )
    )

    number_model = create_entity_regressor(
        random_state
    )

    pair_model = create_entity_regressor(
        random_state + 1
    )

    number_model.fit(
        number_x,
        number_y,
    )

    pair_model.fit(
        pair_x,
        pair_y,
    )

    return number_model, pair_model


# ============================================================
# STABILNO RANGIRANJE
# ============================================================

def stable_descending_order(
    scores: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    keys: list[np.ndarray] = []

    for column in range(
        TOP_K - 1,
        -1,
        -1,
    ):
        keys.append(
            candidates[:, column]
        )

    keys.append(
        -np.asarray(
            scores,
            dtype=float,
        )
    )

    return np.lexsort(
        tuple(keys)
    )


def keep_best(
    candidates: np.ndarray,
    scores: np.ndarray,
    keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = min(
        keep,
        len(candidates),
    )

    if len(candidates) > keep:
        indices = np.argpartition(
            scores,
            -keep,
        )[-keep:]

        candidates = candidates[indices]
        scores = scores[indices]

    order = stable_descending_order(
        scores,
        candidates,
    )

    return (
        candidates[order],
        scores[order],
    )


# ============================================================
# KANDIDATI
# ============================================================

def stable_unique_combinations(
    candidates: list[tuple[int, ...]],
) -> np.ndarray:
    return np.asarray(
        sorted(
            set(candidates)
        ),
        dtype=np.int16,
    )


def create_group_candidates(
    draws: np.ndarray,
    t: int,
    number_scores: np.ndarray,
) -> np.ndarray:
    actual = tuple(
        int(value)
        for value in draws[t]
    )

    actual_set = set(actual)

    candidates: list[tuple[int, ...]] = [
        actual
    ]

    ranking = np.lexsort(
        (
            np.arange(
                MIN_NUMBER,
                MAX_NUMBER + 1,
            ),
            -number_scores,
        )
    ) + 1

    replacements = [
        int(number)
        for number in ranking
        if int(number) not in actual_set
    ][
        :NEAR_MISS_REPLACEMENTS
    ]

    for removed in actual:
        for replacement in replacements:
            changed = tuple(
                sorted(
                    (
                        actual_set
                        - {removed}
                    )
                    | {replacement}
                )
            )

            candidates.append(changed)

    retrieval_pool = sorted(
        int(number)
        for number in ranking[
            :RETRIEVAL_POOL
        ]
    )

    retrieved = np.asarray(
        list(
            combinations(
                retrieval_pool,
                TOP_K,
            )
        ),
        dtype=np.int16,
    )

    if len(retrieved) > 0:
        retrieved_scores = number_scores[
            retrieved - 1
        ].mean(axis=1)

        retrieved_order = stable_descending_order(
            retrieved_scores,
            retrieved,
        )

        for index in retrieved_order[
            :RETRIEVED_PER_GROUP
        ]:
            candidates.append(
                tuple(
                    int(value)
                    for value in retrieved[index]
                )
            )

    history_start = max(
        0,
        t - HISTORICAL_CANDIDATES,
    )

    for historical_draw in draws[
        history_start:t
    ]:
        candidates.append(
            tuple(
                int(value)
                for value in historical_draw
            )
        )

    return stable_unique_combinations(
        candidates
    )


# ============================================================
# OSOBINE KOMBINACIJA
# ============================================================

def combination_features(
    candidates: np.ndarray,
    number_scores: np.ndarray,
    pair_scores: np.ndarray,
    precise: bool,
) -> np.ndarray:
    candidates = np.asarray(
        candidates,
        dtype=np.int16,
    )

    zero_based = candidates - 1

    number_values = number_scores[
        zero_based
    ]

    pair_ids = PAIR_INDEX[
        zero_based[:, LOCAL_PAIR_LEFT],
        zero_based[:, LOCAL_PAIR_RIGHT],
    ]

    pair_values = pair_scores[
        pair_ids
    ]

    differences = np.diff(
        candidates,
        axis=1,
    ).astype(float)

    sums = candidates.sum(
        axis=1
    ).astype(float)

    standard_deviations = candidates.std(
        axis=1
    )

    ranges = (
        candidates[:, -1]
        - candidates[:, 0]
    ).astype(float)

    odd_counts = (
        candidates % 2 == 1
    ).sum(axis=1).astype(float)

    low_counts = (
        candidates <= 20
    ).sum(axis=1).astype(float)

    consecutive_counts = (
        differences == 1
    ).sum(axis=1).astype(float)

    compact = np.column_stack(
        [
            number_values.mean(axis=1),
            number_values.std(axis=1),
            number_values.min(axis=1),
            number_values.max(axis=1),
            np.quantile(
                number_values,
                0.25,
                axis=1,
            ),
            np.quantile(
                number_values,
                0.50,
                axis=1,
            ),
            np.quantile(
                number_values,
                0.75,
                axis=1,
            ),
            pair_values.mean(axis=1),
            pair_values.std(axis=1),
            pair_values.min(axis=1),
            pair_values.max(axis=1),
            np.quantile(
                pair_values,
                0.25,
                axis=1,
            ),
            np.quantile(
                pair_values,
                0.50,
                axis=1,
            ),
            np.quantile(
                pair_values,
                0.75,
                axis=1,
            ),
            sums / 140.0,
            standard_deviations / 12.0,
            ranges / 38.0,
            odd_counts / TOP_K,
            low_counts / TOP_K,
            consecutive_counts / 6.0,
            differences.mean(axis=1) / 7.0,
            differences.std(axis=1) / 7.0,
        ]
    )

    if not precise:
        return compact.astype(
            np.float32
        )

    return np.column_stack(
        [
            compact,
            np.sort(
                number_values,
                axis=1,
            ),
            np.sort(
                pair_values,
                axis=1,
            ),
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

    valid = (
        np.all(
            np.isfinite(
                candidate_number_waits
            ),
            axis=1,
        )
        & np.all(
            np.isfinite(
                candidate_pair_waits
            ),
            axis=1,
        )
    )

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

    valid_candidates = candidates[
        valid
    ]

    overlap_component = np.zeros(
        len(valid_candidates),
        dtype=float,
    )

    horizon = min(
        OVERLAP_HORIZON,
        len(draws) - t,
    )

    total_weight = 0.0

    for offset in range(horizon):
        weight = (
            OVERLAP_DECAY ** offset
        )

        hits = np.isin(
            valid_candidates,
            draws[t + offset],
        ).sum(axis=1)

        overlap_component += weight * (
            hits / TOP_K - BASE_RATE
        )

        total_weight += weight

    if total_weight > 0:
        overlap_component /= total_weight

    targets[valid] = (
        TARGET_NUMBER_WEIGHT
        * number_component
        + TARGET_PAIR_WEIGHT
        * pair_component
        + TARGET_OVERLAP_WEIGHT
        * overlap_component
    ).astype(np.float32)

    return targets


# ============================================================
# HRONOLOŠKE GRUPE
# ============================================================

def choose_group_times(
    total_draws: int,
) -> np.ndarray:
    first_allowed = (
        MIN_HISTORY
        + NUMBER_MODEL_WARMUP
    )

    last_allowed = (
        total_draws
        - MIN_FUTURE_FOR_GROUP
    )

    if last_allowed <= first_allowed:
        raise ValueError(
            "Nema dovoljno podataka za obuku, "
            "validaciju i buduće mete."
        )

    group_count = min(
        REQUESTED_GROUPS,
        last_allowed - first_allowed,
    )

    return np.unique(
        np.linspace(
            first_allowed,
            last_allowed - 1,
            num=group_count,
            dtype=np.int32,
        )
    )


def build_expanding_group_dataset(
    draws: np.ndarray,
    number_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    number_waits: np.ndarray,
    pair_waits: np.ndarray,
    group_times: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, int],
]:
    fast_parts: list[np.ndarray] = []
    precise_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []

    group_time_map: dict[int, int] = {}

    for position, t_value in enumerate(
        group_times,
        start=1,
    ):
        t = int(t_value)

        print(
            f"  Expanding grupa "
            f"{position}/{len(group_times)} "
            f"(istorija do reda {t})"
        )

        number_model, pair_model = (
            train_expanding_entity_models(
                number_matrix,
                pair_matrix,
                number_waits,
                pair_waits,
                end_limit=t,
                random_state=(
                    RANDOM_SEED + position * 10
                ),
            )
        )

        current_number_features = (
            number_features_at(
                number_matrix,
                t,
            )
        )

        current_pair_features = (
            pair_features_at(
                number_matrix,
                pair_matrix,
                t,
            )
        )

        number_scores = number_model.predict(
            current_number_features
        ).astype(np.float32)

        pair_scores = pair_model.predict(
            current_pair_features
        ).astype(np.float32)

        candidates = create_group_candidates(
            draws,
            t,
            number_scores,
        )

        targets = combination_targets(
            candidates,
            t,
            draws,
            number_waits,
            pair_waits,
        )

        valid = np.isfinite(targets)

        if np.sum(valid) < 10:
            continue

        candidates = candidates[valid]
        targets = targets[valid]

        group_id = len(group_time_map)

        group_time_map[group_id] = t

        fast_parts.append(
            combination_features(
                candidates,
                number_scores,
                pair_scores,
                precise=False,
            )
        )

        precise_parts.append(
            combination_features(
                candidates,
                number_scores,
                pair_scores,
                precise=True,
            )
        )

        target_parts.append(targets)

        candidate_parts.append(
            candidates
        )

        group_parts.append(
            np.full(
                len(candidates),
                group_id,
                dtype=np.int16,
            )
        )

    if not fast_parts:
        raise RuntimeError(
            "Nije napravljena nijedna "
            "važeća hronološka grupa."
        )

    return (
        np.vstack(fast_parts),
        np.vstack(precise_parts),
        np.concatenate(target_parts),
        np.concatenate(group_parts),
        np.vstack(candidate_parts),
        group_time_map,
    )


# ============================================================
# KOMBINACIJSKI REGRESORI
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
):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=parameters[
            "learning_rate"
        ],
        max_iter=parameters[
            "max_iter"
        ],
        max_leaf_nodes=parameters[
            "max_leaf_nodes"
        ],
        min_samples_leaf=parameters[
            "min_samples_leaf"
        ],
        l2_regularization=parameters[
            "l2_regularization"
        ],
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def create_precise_model(
    parameters: dict,
):
    return ExtraTreesRegressor(
        n_estimators=parameters[
            "n_estimators"
        ],
        min_samples_leaf=parameters[
            "min_samples_leaf"
        ],
        max_features=parameters[
            "max_features"
        ],
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


# ============================================================
# NESTED VREMENSKI IZBOR PARAMETARA
# ============================================================

def nested_fast_parameter_selection(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
) -> dict:
    unique_groups = np.unique(groups)

    inner_validation_count = max(
        1,
        int(
            round(
                len(unique_groups) * 0.20
            )
        ),
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

    best_parameters = dict(
        FAST_PARAMETER_SETS[0]
    )

    best_mae = float("inf")

    for parameters in FAST_PARAMETER_SETS:
        model = create_fast_model(
            parameters
        )

        model.fit(
            features[train_mask],
            targets[train_mask],
        )

        predictions = model.predict(
            features[validation_mask]
        )

        mae = mean_absolute_error(
            targets[validation_mask],
            predictions,
        )

        if mae < best_mae - EPSILON:
            best_mae = float(mae)
            best_parameters = dict(
                parameters
            )

    return best_parameters


def nested_precise_parameter_selection(
    fast_features: np.ndarray,
    precise_features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    fast_parameters: dict,
) -> dict:
    """
    Precizni model bira se samo nad kandidatima koje je
    unutrašnji brzi model stvarno zadržao.
    """

    unique_groups = np.unique(groups)

    inner_validation_count = max(
        1,
        int(
            round(
                len(unique_groups) * 0.20
            )
        ),
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

    fast_model = create_fast_model(
        fast_parameters
    )

    fast_model.fit(
        fast_features[train_mask],
        targets[train_mask],
    )

    fast_predictions = fast_model.predict(
        fast_features
    )

    precise_training_indices: list[int] = []
    precise_validation_indices: list[int] = []

    for group_id in unique_groups:
        group_indices = np.flatnonzero(
            groups == group_id
        )

        local_order = np.argsort(
            -fast_predictions[group_indices],
            kind="stable",
        )

        kept = group_indices[
            local_order[
                :min(
                    VALIDATION_FAST_KEEP,
                    len(local_order),
                )
            ]
        ]

        if group_id in inner_train_groups:
            precise_training_indices.extend(
                kept.tolist()
            )
        else:
            precise_validation_indices.extend(
                kept.tolist()
            )

    precise_training_indices = np.asarray(
        precise_training_indices,
        dtype=np.int32,
    )

    precise_validation_indices = np.asarray(
        precise_validation_indices,
        dtype=np.int32,
    )

    best_parameters = dict(
        PRECISE_PARAMETER_SETS[0]
    )

    best_mae = float("inf")

    for parameters in PRECISE_PARAMETER_SETS:
        model = create_precise_model(
            parameters
        )

        model.fit(
            precise_features[
                precise_training_indices
            ],
            targets[
                precise_training_indices
            ],
        )

        predictions = model.predict(
            precise_features[
                precise_validation_indices
            ]
        )

        mae = mean_absolute_error(
            targets[
                precise_validation_indices
            ],
            predictions,
        )

        if mae < best_mae - EPSILON:
            best_mae = float(mae)
            best_parameters = dict(
                parameters
            )

    return best_parameters


# ============================================================
# SPISAK KANDIDATA PRECIZNE OBUKE
# ============================================================

def create_precise_training_indices(
    fast_model,
    fast_features: np.ndarray,
    groups: np.ndarray,
    allowed_groups: np.ndarray,
) -> np.ndarray:
    predictions = fast_model.predict(
        fast_features
    )

    selected: list[int] = []

    for group_id in allowed_groups:
        group_indices = np.flatnonzero(
            groups == group_id
        )

        order = np.argsort(
            -predictions[group_indices],
            kind="stable",
        )

        kept = group_indices[
            order[
                :min(
                    VALIDATION_FAST_KEEP,
                    len(order),
                )
            ]
        ]

        selected.extend(
            kept.tolist()
        )

    return np.asarray(
        selected,
        dtype=np.int32,
    )


# ============================================================
# SPOLJNA HOLDOUT VALIDACIJA
# ============================================================

def evaluate_outer_holdout(
    fast_model,
    precise_model,
    fast_features: np.ndarray,
    precise_features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    candidates: np.ndarray,
    validation_groups: np.ndarray,
    group_time_map: dict[int, int],
    draws: np.ndarray,
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

    fast_target_values: list[np.ndarray] = []
    fast_prediction_values: list[np.ndarray] = []

    precise_target_values: list[np.ndarray] = []
    precise_prediction_values: list[np.ndarray] = []

    fast_recall_successes = 0
    precise_recall_successes = 0

    fast_hits: list[int] = []
    precise_hits: list[int] = []

    for group_id in validation_groups:
        group_indices = np.flatnonzero(
            groups == group_id
        )

        group_candidates = candidates[
            group_indices
        ]

        group_targets = targets[
            group_indices
        ]

        group_fast_scores = fast_predictions[
            group_indices
        ]

        fast_target_values.append(
            group_targets
        )

        fast_prediction_values.append(
            group_fast_scores
        )

        true_order = stable_descending_order(
            group_targets,
            group_candidates,
        )

        best_true_combination = (
            group_candidates[
                true_order[0]
            ]
        )

        fast_order = stable_descending_order(
            group_fast_scores,
            group_candidates,
        )

        fast_recall_candidates = (
            group_candidates[
                fast_order[
                    :min(
                        RECALL_K,
                        len(fast_order),
                    )
                ]
            ]
        )

        if np.any(
            np.all(
                fast_recall_candidates
                == best_true_combination,
                axis=1,
            )
        ):
            fast_recall_successes += 1

        shortlist_local = fast_order[
            :min(
                VALIDATION_FAST_KEEP,
                len(fast_order),
            )
        ]

        shortlist_indices = group_indices[
            shortlist_local
        ]

        shortlist_candidates = candidates[
            shortlist_indices
        ]

        shortlist_targets = targets[
            shortlist_indices
        ]

        shortlist_precise_scores = precise_model.predict(
            precise_features[
                shortlist_indices
            ]
        )

        # Precise MAE samo nad stvarno prosleđenim kandidatima.
        precise_target_values.append(
            shortlist_targets
        )

        precise_prediction_values.append(
            shortlist_precise_scores
        )

        precise_order = stable_descending_order(
            shortlist_precise_scores,
            shortlist_candidates,
        )

        precise_recall_candidates = (
            shortlist_candidates[
                precise_order[
                    :min(
                        RECALL_K,
                        len(precise_order),
                    )
                ]
            ]
        )

        if np.any(
            np.all(
                precise_recall_candidates
                == best_true_combination,
                axis=1,
            )
        ):
            precise_recall_successes += 1

        fast_choice = group_candidates[
            fast_order[0]
        ]

        precise_choice = shortlist_candidates[
            precise_order[0]
        ]

        actual_draw = draws[
            group_time_map[int(group_id)]
        ]

        fast_hits.append(
            int(
                np.isin(
                    fast_choice,
                    actual_draw,
                ).sum()
            )
        )

        precise_hits.append(
            int(
                np.isin(
                    precise_choice,
                    actual_draw,
                ).sum()
            )
        )

    fast_mae = mean_absolute_error(
        np.concatenate(
            fast_target_values
        ),
        np.concatenate(
            fast_prediction_values
        ),
    )

    precise_mae = mean_absolute_error(
        np.concatenate(
            precise_target_values
        ),
        np.concatenate(
            precise_prediction_values
        ),
    )

    denominator = max(
        len(validation_groups),
        1,
    )

    return (
        float(fast_mae),
        float(precise_mae),
        fast_recall_successes / denominator,
        precise_recall_successes / denominator,
        np.asarray(
            fast_hits,
            dtype=float,
        ),
        np.asarray(
            precise_hits,
            dtype=float,
        ),
    )


# ============================================================
# STATISTIČKA PROCENA
# ============================================================

def bootstrap_interval(
    hits: np.ndarray,
) -> tuple[float, float]:
    if len(hits) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    differences = (
        hits - RANDOM_EXPECTED_HITS
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = rng.integers(
        0,
        len(differences),
        size=(
            STATISTICAL_SIMULATIONS,
            len(differences),
        ),
    )

    means = differences[
        indices
    ].mean(axis=1)

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def permutation_test(
    hits: np.ndarray,
) -> float:
    if len(hits) == 0:
        return float("nan")

    differences = (
        hits - RANDOM_EXPECTED_HITS
    )

    observed = abs(
        float(
            differences.mean()
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    signs = rng.choice(
        np.asarray(
            [-1.0, 1.0]
        ),
        size=(
            STATISTICAL_SIMULATIONS,
            len(differences),
        ),
    )

    null_means = np.abs(
        (
            signs * differences
        ).mean(axis=1)
    )

    return float(
        (
            np.sum(
                null_means >= observed
            )
            + 1
        )
        / (
            STATISTICAL_SIMULATIONS
            + 1
        )
    )


# ============================================================
# PAKETI SVIH KOMBINACIJA
# ============================================================

def combination_batches(
    batch_size: int,
) -> Iterator[np.ndarray]:
    iterator = combinations(
        range(
            MIN_NUMBER,
            MAX_NUMBER + 1,
        ),
        TOP_K,
    )

    while True:
        values = list(
            islice(
                iterator,
                batch_size,
            )
        )

        if not values:
            break

        yield np.asarray(
            values,
            dtype=np.int16,
        )


def exhaustive_fast_search(
    fast_model,
    number_scores: np.ndarray,
    pair_scores: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    best_candidates = np.empty(
        (0, TOP_K),
        dtype=np.int16,
    )

    best_scores = np.empty(
        0,
        dtype=float,
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

        scores = fast_model.predict(
            features
        )

        checked += len(batch)

        best_candidates, best_scores = keep_best(
            np.vstack(
                [
                    best_candidates,
                    batch,
                ]
            ),
            np.concatenate(
                [
                    best_scores,
                    scores,
                ]
            ),
            FINAL_FAST_KEEP,
        )

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or checked == TOTAL_COMBINATIONS
        ):
            print(
                f"  Pregledano: {checked:,} / "
                f"{TOTAL_COMBINATIONS:,} "
                f"({checked / TOTAL_COMBINATIONS:7.2%})"
            )

    return (
        best_candidates,
        best_scores,
        checked,
    )


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

    draws = load_lottery_csv(
        csv_path
    )

    print(f"CSV: {csv_path}")
    print(f"Broj redova: {len(draws)}")
    print("Prvi red je najstariji.")
    print("Poslednji red je najnoviji.")

    number_matrix = create_number_matrix(
        draws
    )

    pair_matrix = create_pair_matrix(
        draws
    )

    print(
        "Pravljenje kontinuiranih "
        "time-to-event meta..."
    )

    number_waits = create_next_wait_matrix(
        number_matrix
    )

    pair_waits = create_next_wait_matrix(
        pair_matrix
    )

    group_times = choose_group_times(
        len(draws)
    )

    print(
        "Expanding walk-forward obuka "
        "brojeva i parova..."
    )

    (
        fast_x,
        precise_x,
        combo_y,
        groups,
        candidates,
        group_time_map,
    ) = build_expanding_group_dataset(
        draws,
        number_matrix,
        pair_matrix,
        number_waits,
        pair_waits,
        group_times,
    )

    unique_groups = np.unique(
        groups
    )

    validation_count = max(
        1,
        int(
            round(
                len(unique_groups)
                * VALIDATION_FRACTION
            )
        ),
    )

    training_groups = unique_groups[
        :-validation_count
    ]

    validation_groups = unique_groups[
        -validation_count:
    ]

    training_mask = np.isin(
        groups,
        training_groups,
    )

    print(
        f"Hronoloških grupa za obuku: "
        f"{len(training_groups)}"
    )

    print(
        f"Odvojenih validacionih grupa: "
        f"{len(validation_groups)}"
    )

    print(
        "Nested vremenski izbor "
        "parametara brze faze..."
    )

    fast_parameters = (
        nested_fast_parameter_selection(
            fast_x[training_mask],
            combo_y[training_mask],
            groups[training_mask],
        )
    )

    print(
        "Nested vremenski izbor "
        "parametara precizne faze..."
    )

    precise_parameters = (
        nested_precise_parameter_selection(
            fast_x[training_mask],
            precise_x[training_mask],
            combo_y[training_mask],
            groups[training_mask],
            fast_parameters,
        )
    )

    fast_validation_model = create_fast_model(
        fast_parameters
    )

    fast_validation_model.fit(
        fast_x[training_mask],
        combo_y[training_mask],
    )

    precise_training_indices = (
        create_precise_training_indices(
            fast_validation_model,
            fast_x,
            groups,
            training_groups,
        )
    )

    precise_validation_model = (
        create_precise_model(
            precise_parameters
        )
    )

    precise_validation_model.fit(
        precise_x[
            precise_training_indices
        ],
        combo_y[
            precise_training_indices
        ],
    )

    (
        fast_mae,
        precise_mae,
        fast_recall,
        precise_recall,
        fast_hits,
        precise_hits,
    ) = evaluate_outer_holdout(
        fast_validation_model,
        precise_validation_model,
        fast_x,
        precise_x,
        combo_y,
        groups,
        candidates,
        validation_groups,
        group_time_map,
        draws,
    )

    use_precise = (
        precise_mae < fast_mae
        and precise_recall >= fast_recall
    )

    selected_stage = (
        "precise"
        if use_precise
        else "fast"
    )

    selected_hits = (
        precise_hits
        if use_precise
        else fast_hits
    )

    print(f"Fast MAE: {fast_mae:.6f}")
    print(
        f"Precise MAE: "
        f"{precise_mae:.6f}"
    )

    print(
        f"Fast recall@{RECALL_K}: "
        f"{fast_recall:.1%}"
    )

    print(
        f"Precise recall@{RECALL_K}: "
        f"{precise_recall:.1%}"
    )

    print(
        f"Izabrana završna faza: "
        f"{selected_stage}"
    )

    # Završni kombinacijski modeli koriste sve grupe.
    final_fast_model = create_fast_model(
        fast_parameters
    )

    final_fast_model.fit(
        fast_x,
        combo_y,
    )

    final_precise_training_indices = (
        create_precise_training_indices(
            final_fast_model,
            fast_x,
            groups,
            unique_groups,
        )
    )

    final_precise_model = create_precise_model(
        precise_parameters
    )

    final_precise_model.fit(
        precise_x[
            final_precise_training_indices
        ],
        combo_y[
            final_precise_training_indices
        ],
    )

    print(
        "Obuka završnih distribucijskih "
        "regresora brojeva i parova..."
    )

    final_number_model, final_pair_model = (
        train_expanding_entity_models(
            number_matrix,
            pair_matrix,
            number_waits,
            pair_waits,
            end_limit=len(draws),
            random_state=RANDOM_SEED + 100_000,
        )
    )

    current_number_features = (
        number_features_at(
            number_matrix,
            len(draws),
        )
    )

    current_pair_features = (
        pair_features_at(
            number_matrix,
            pair_matrix,
            len(draws),
        )
    )

    current_number_scores = (
        final_number_model.predict(
            current_number_features
        ).astype(np.float32)
    )

    current_pair_scores = (
        final_pair_model.predict(
            current_pair_features
        ).astype(np.float32)
    )

    print(
        f"Paketni pregled svih "
        f"{TOTAL_COMBINATIONS:,} kombinacija..."
    )

    (
        finalists,
        finalist_fast_scores,
        checked,
    ) = exhaustive_fast_search(
        final_fast_model,
        current_number_scores,
        current_pair_scores,
    )

    if checked != TOTAL_COMBINATIONS:
        raise RuntimeError(
            f"Pregledano je {checked:,}, a očekivano "
            f"{TOTAL_COMBINATIONS:,} kombinacija."
        )

    if use_precise:
        finalist_precise_features = (
            combination_features(
                finalists,
                current_number_scores,
                current_pair_scores,
                precise=True,
            )
        )

        finalist_scores = (
            final_precise_model.predict(
                finalist_precise_features
            )
        )

        final_order = stable_descending_order(
            finalist_scores,
            finalists,
        )
    else:
        final_order = stable_descending_order(
            finalist_fast_scores,
            finalists,
        )

    prediction = finalists[
        final_order[0]
    ]

    average_hits = float(
        selected_hits.mean()
    )

    difference = (
        average_hits
        - RANDOM_EXPECTED_HITS
    )

    bootstrap_low, bootstrap_high = (
        bootstrap_interval(
            selected_hits
        )
    )

    p_value = permutation_test(
        selected_hits
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return GameResult(
        name=name,
        prediction=prediction,
        rows=len(draws),
        combinations_checked=checked,
        training_groups=len(
            training_groups
        ),
        validation_groups=len(
            validation_groups
        ),
        fast_mae=fast_mae,
        precise_mae=precise_mae,
        fast_recall=fast_recall,
        precise_recall=precise_recall,
        selected_stage=selected_stage,
        holdout_average_hits=average_hits,
        holdout_difference=difference,
        bootstrap_low=bootstrap_low,
        bootstrap_high=bootstrap_high,
        permutation_p=p_value,
        elapsed_seconds=elapsed,
    )


# ============================================================
# ISPIS
# ============================================================

def format_combination(
    values: np.ndarray,
) -> str:
    return ", ".join(
        f"{int(value):02d}"
        for value in values
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

    print(
        f"CSV redova: "
        f"{result.rows:,}"
    )

    print(
        f"Pregledano kombinacija: "
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

    print(
        f"Fast MAE: "
        f"{result.fast_mae:.6f}"
    )

    print(
        f"Precise MAE: "
        f"{result.precise_mae:.6f}"
    )

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
    np.random.seed(
        RANDOM_SEED
    )

    print("=" * 72)
    print(
        "LOTO 7/39 — FINALNI "
        "DISTRIBUCIJSKI REGRESIONI SISTEM"
    )
    print("=" * 72)

    print(
        "Teorijska osnovna stopa broja: "
        f"{BASE_RATE:.6f}"
    )

    print(
        "Teorijsko očekivanje pogodaka: "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        "Ukupno mogućih kombinacija: "
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

    print_result(
        loto_result
    )

    print_result(
        loto_plus_result
    )

    print()
    print("=" * 72)
    print("NAPOMENA")
    print("=" * 72)

    print(
        "Isti kompletan postupak pokrenut je "
        "zasebno nad svakim CSV-om."
    )

    print(
        "Pregled svih kombinacija pronalazi najbolje "
        "ocenjenu kombinaciju prema modelu, ali ne "
        "predstavlja garanciju dobitka."
    )


if __name__ == "__main__":
    main()



"""
========================================================================
LOTO 7/39 — FINALNI DISTRIBUCIJSKI REGRESIONI SISTEM
========================================================================
Teorijska osnovna stopa broja: 0.179487
Teorijsko očekivanje pogodaka: 1.256410
Ukupno mogućih kombinacija: 15,380,937

========================================================================
Obrada: Loto
========================================================================
CSV: /data/loto7_4674_k68_loto_2959.csv
Broj redova: 2959
Prvi red je najstariji.
Poslednji red je najnoviji.
Pravljenje kontinuiranih time-to-event meta...
Expanding walk-forward obuka brojeva i parova...
  Expanding grupa 1/98 (istorija do reda 500)
  Expanding grupa 2/98 (istorija do reda 522)
  Expanding grupa 3/98 (istorija do reda 545)
  Expanding grupa 4/98 (istorija do reda 568)
  Expanding grupa 5/98 (istorija do reda 591)
  Expanding grupa 6/98 (istorija do reda 613)
  Expanding grupa 7/98 (istorija do reda 636)
  Expanding grupa 8/98 (istorija do reda 659)
  Expanding grupa 9/98 (istorija do reda 682)
  Expanding grupa 10/98 (istorija do reda 704)
  Expanding grupa 11/98 (istorija do reda 727)
  Expanding grupa 12/98 (istorija do reda 750)
  Expanding grupa 13/98 (istorija do reda 773)
  Expanding grupa 14/98 (istorija do reda 795)
  Expanding grupa 15/98 (istorija do reda 818)
  Expanding grupa 16/98 (istorija do reda 841)
  Expanding grupa 17/98 (istorija do reda 864)
  Expanding grupa 18/98 (istorija do reda 886)
  Expanding grupa 19/98 (istorija do reda 909)
  Expanding grupa 20/98 (istorija do reda 932)
  Expanding grupa 21/98 (istorija do reda 955)
  Expanding grupa 22/98 (istorija do reda 978)
  Expanding grupa 23/98 (istorija do reda 1000)
  Expanding grupa 24/98 (istorija do reda 1023)
  Expanding grupa 25/98 (istorija do reda 1046)
  Expanding grupa 26/98 (istorija do reda 1069)
  Expanding grupa 27/98 (istorija do reda 1091)
  Expanding grupa 28/98 (istorija do reda 1114)
  Expanding grupa 29/98 (istorija do reda 1137)
  Expanding grupa 30/98 (istorija do reda 1160)
  Expanding grupa 31/98 (istorija do reda 1182)
  Expanding grupa 32/98 (istorija do reda 1205)
  Expanding grupa 33/98 (istorija do reda 1228)
  Expanding grupa 34/98 (istorija do reda 1251)
  Expanding grupa 35/98 (istorija do reda 1273)
  Expanding grupa 36/98 (istorija do reda 1296)
  Expanding grupa 37/98 (istorija do reda 1319)
  Expanding grupa 38/98 (istorija do reda 1342)
  Expanding grupa 39/98 (istorija do reda 1364)
  Expanding grupa 40/98 (istorija do reda 1387)
  Expanding grupa 41/98 (istorija do reda 1410)
  Expanding grupa 42/98 (istorija do reda 1433)
  Expanding grupa 43/98 (istorija do reda 1456)
  Expanding grupa 44/98 (istorija do reda 1478)
  Expanding grupa 45/98 (istorija do reda 1501)
  Expanding grupa 46/98 (istorija do reda 1524)
  Expanding grupa 47/98 (istorija do reda 1547)
  Expanding grupa 48/98 (istorija do reda 1569)
  Expanding grupa 49/98 (istorija do reda 1592)
  Expanding grupa 50/98 (istorija do reda 1615)
  Expanding grupa 51/98 (istorija do reda 1638)
  Expanding grupa 52/98 (istorija do reda 1660)
  Expanding grupa 53/98 (istorija do reda 1683)
  Expanding grupa 54/98 (istorija do reda 1706)
  Expanding grupa 55/98 (istorija do reda 1729)
  Expanding grupa 56/98 (istorija do reda 1751)
  Expanding grupa 57/98 (istorija do reda 1774)
  Expanding grupa 58/98 (istorija do reda 1797)
  Expanding grupa 59/98 (istorija do reda 1820)
  Expanding grupa 60/98 (istorija do reda 1843)
  Expanding grupa 61/98 (istorija do reda 1865)
  Expanding grupa 62/98 (istorija do reda 1888)
  Expanding grupa 63/98 (istorija do reda 1911)
  Expanding grupa 64/98 (istorija do reda 1934)
  Expanding grupa 65/98 (istorija do reda 1956)
  Expanding grupa 66/98 (istorija do reda 1979)
  Expanding grupa 67/98 (istorija do reda 2002)
  Expanding grupa 68/98 (istorija do reda 2025)
  Expanding grupa 69/98 (istorija do reda 2047)
  Expanding grupa 70/98 (istorija do reda 2070)
  Expanding grupa 71/98 (istorija do reda 2093)
  Expanding grupa 72/98 (istorija do reda 2116)
  Expanding grupa 73/98 (istorija do reda 2138)
  Expanding grupa 74/98 (istorija do reda 2161)
  Expanding grupa 75/98 (istorija do reda 2184)
  Expanding grupa 76/98 (istorija do reda 2207)
  Expanding grupa 77/98 (istorija do reda 2229)
  Expanding grupa 78/98 (istorija do reda 2252)
  Expanding grupa 79/98 (istorija do reda 2275)
  Expanding grupa 80/98 (istorija do reda 2298)
  Expanding grupa 81/98 (istorija do reda 2321)
  Expanding grupa 82/98 (istorija do reda 2343)
  Expanding grupa 83/98 (istorija do reda 2366)
  Expanding grupa 84/98 (istorija do reda 2389)
  Expanding grupa 85/98 (istorija do reda 2412)
  Expanding grupa 86/98 (istorija do reda 2434)
  Expanding grupa 87/98 (istorija do reda 2457)
  Expanding grupa 88/98 (istorija do reda 2480)
  Expanding grupa 89/98 (istorija do reda 2503)
  Expanding grupa 90/98 (istorija do reda 2525)
  Expanding grupa 91/98 (istorija do reda 2548)
  Expanding grupa 92/98 (istorija do reda 2571)
  Expanding grupa 93/98 (istorija do reda 2594)
  Expanding grupa 94/98 (istorija do reda 2616)
  Expanding grupa 95/98 (istorija do reda 2639)
  Expanding grupa 96/98 (istorija do reda 2662)
  Expanding grupa 97/98 (istorija do reda 2685)
  Expanding grupa 98/98 (istorija do reda 2708)
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Nested vremenski izbor parametara brze faze...
Nested vremenski izbor parametara precizne faze...
Fast MAE: 0.352200
Precise MAE: 0.499639
Fast recall@10: 0.0%
Precise recall@10: 0.0%
Izabrana završna faza: fast
Obuka završnih distribucijskih regresora brojeva i parova...
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
CSV: /data/loto7_4674_k68_loto_plus_1715.csv
Broj redova: 1715
Prvi red je najstariji.
Poslednji red je najnoviji.
Pravljenje kontinuiranih time-to-event meta...
Expanding walk-forward obuka brojeva i parova...
  Expanding grupa 1/98 (istorija do reda 500)
  Expanding grupa 2/98 (istorija do reda 509)
  Expanding grupa 3/98 (istorija do reda 519)
  Expanding grupa 4/98 (istorija do reda 529)
  Expanding grupa 5/98 (istorija do reda 539)
  Expanding grupa 6/98 (istorija do reda 549)
  Expanding grupa 7/98 (istorija do reda 559)
  Expanding grupa 8/98 (istorija do reda 569)
  Expanding grupa 9/98 (istorija do reda 579)
  Expanding grupa 10/98 (istorija do reda 589)
  Expanding grupa 11/98 (istorija do reda 599)
  Expanding grupa 12/98 (istorija do reda 609)
  Expanding grupa 13/98 (istorija do reda 619)
  Expanding grupa 14/98 (istorija do reda 629)
  Expanding grupa 15/98 (istorija do reda 639)
  Expanding grupa 16/98 (istorija do reda 649)
  Expanding grupa 17/98 (istorija do reda 659)
  Expanding grupa 18/98 (istorija do reda 668)
  Expanding grupa 19/98 (istorija do reda 678)
  Expanding grupa 20/98 (istorija do reda 688)
  Expanding grupa 21/98 (istorija do reda 698)
  Expanding grupa 22/98 (istorija do reda 708)
  Expanding grupa 23/98 (istorija do reda 718)
  Expanding grupa 24/98 (istorija do reda 728)
  Expanding grupa 25/98 (istorija do reda 738)
  Expanding grupa 26/98 (istorija do reda 748)
  Expanding grupa 27/98 (istorija do reda 758)
  Expanding grupa 28/98 (istorija do reda 768)
  Expanding grupa 29/98 (istorija do reda 778)
  Expanding grupa 30/98 (istorija do reda 788)
  Expanding grupa 31/98 (istorija do reda 798)
  Expanding grupa 32/98 (istorija do reda 808)
  Expanding grupa 33/98 (istorija do reda 818)
  Expanding grupa 34/98 (istorija do reda 827)
  Expanding grupa 35/98 (istorija do reda 837)
  Expanding grupa 36/98 (istorija do reda 847)
  Expanding grupa 37/98 (istorija do reda 857)
  Expanding grupa 38/98 (istorija do reda 867)
  Expanding grupa 39/98 (istorija do reda 877)
  Expanding grupa 40/98 (istorija do reda 887)
  Expanding grupa 41/98 (istorija do reda 897)
  Expanding grupa 42/98 (istorija do reda 907)
  Expanding grupa 43/98 (istorija do reda 917)
  Expanding grupa 44/98 (istorija do reda 927)
  Expanding grupa 45/98 (istorija do reda 937)
  Expanding grupa 46/98 (istorija do reda 947)
  Expanding grupa 47/98 (istorija do reda 957)
  Expanding grupa 48/98 (istorija do reda 967)
  Expanding grupa 49/98 (istorija do reda 977)
  Expanding grupa 50/98 (istorija do reda 986)
  Expanding grupa 51/98 (istorija do reda 996)
  Expanding grupa 52/98 (istorija do reda 1006)
  Expanding grupa 53/98 (istorija do reda 1016)
  Expanding grupa 54/98 (istorija do reda 1026)
  Expanding grupa 55/98 (istorija do reda 1036)
  Expanding grupa 56/98 (istorija do reda 1046)
  Expanding grupa 57/98 (istorija do reda 1056)
  Expanding grupa 58/98 (istorija do reda 1066)
  Expanding grupa 59/98 (istorija do reda 1076)
  Expanding grupa 60/98 (istorija do reda 1086)
  Expanding grupa 61/98 (istorija do reda 1096)
  Expanding grupa 62/98 (istorija do reda 1106)
  Expanding grupa 63/98 (istorija do reda 1116)
  Expanding grupa 64/98 (istorija do reda 1126)
  Expanding grupa 65/98 (istorija do reda 1136)
  Expanding grupa 66/98 (istorija do reda 1145)
  Expanding grupa 67/98 (istorija do reda 1155)
  Expanding grupa 68/98 (istorija do reda 1165)
  Expanding grupa 69/98 (istorija do reda 1175)
  Expanding grupa 70/98 (istorija do reda 1185)
  Expanding grupa 71/98 (istorija do reda 1195)
  Expanding grupa 72/98 (istorija do reda 1205)
  Expanding grupa 73/98 (istorija do reda 1215)
  Expanding grupa 74/98 (istorija do reda 1225)
  Expanding grupa 75/98 (istorija do reda 1235)
  Expanding grupa 76/98 (istorija do reda 1245)
  Expanding grupa 77/98 (istorija do reda 1255)
  Expanding grupa 78/98 (istorija do reda 1265)
  Expanding grupa 79/98 (istorija do reda 1275)
  Expanding grupa 80/98 (istorija do reda 1285)
  Expanding grupa 81/98 (istorija do reda 1295)
  Expanding grupa 82/98 (istorija do reda 1304)
  Expanding grupa 83/98 (istorija do reda 1314)
  Expanding grupa 84/98 (istorija do reda 1324)
  Expanding grupa 85/98 (istorija do reda 1334)
  Expanding grupa 86/98 (istorija do reda 1344)
  Expanding grupa 87/98 (istorija do reda 1354)
  Expanding grupa 88/98 (istorija do reda 1364)
  Expanding grupa 89/98 (istorija do reda 1374)
  Expanding grupa 90/98 (istorija do reda 1384)
  Expanding grupa 91/98 (istorija do reda 1394)
  Expanding grupa 92/98 (istorija do reda 1404)
  Expanding grupa 93/98 (istorija do reda 1414)
  Expanding grupa 94/98 (istorija do reda 1424)
  Expanding grupa 95/98 (istorija do reda 1434)
  Expanding grupa 96/98 (istorija do reda 1444)
  Expanding grupa 97/98 (istorija do reda 1454)
  Expanding grupa 98/98 (istorija do reda 1464)
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Nested vremenski izbor parametara brze faze...
Nested vremenski izbor parametara precizne faze...
Fast MAE: 0.379962
Precise MAE: 0.499083
Fast recall@10: 0.0%
Precise recall@10: 15.0%
Izabrana završna faza: fast
Obuka završnih distribucijskih regresora brojeva i parova...
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
NEXT: 02, x, 08, y, 19, z, 32
CSV redova: 2,959
Pregledano kombinacija: 15,380,937
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Fast MAE: 0.352200
Precise MAE: 0.499639
Fast recall@10: 0.0%
Precise recall@10: 0.0%
Izabrana završna faza: fast
Holdout prosek pogodaka: 5.800000
Slučajno očekivanje: 1.256410
Razlika prema slučajnoj osnovi: +4.543590
Bootstrap 95% interval razlike: [+4.143590, +4.743590]
Permutaciona p-vrednost: 0.000100
Vreme obrade: 21.84 minuta

========================================================================
Loto Plus
========================================================================
NEXT: 04, x, 18, y, 30, z, 35
CSV redova: 1,715
Pregledano kombinacija: 15,380,937
Hronoloških grupa za obuku: 78
Odvojenih validacionih grupa: 20
Fast MAE: 0.379962
Precise MAE: 0.499083
Fast recall@10: 0.0%
Precise recall@10: 15.0%
Izabrana završna faza: fast
Holdout prosek pogodaka: 2.600000
Slučajno očekivanje: 1.256410
Razlika prema slučajnoj osnovi: +1.343590
Bootstrap 95% interval razlike: [+0.243590, +2.493590]
Permutaciona p-vrednost: 0.035996
Vreme obrade: 14.03 minuta

========================================================================
NAPOMENA
========================================================================
Isti kompletan postupak pokrenut je zasebno nad svakim CSV-om.
Pregled svih kombinacija pronalazi najbolje ocenjenu kombinaciju prema modelu, ali ne predstavlja garanciju dobitka.
"""



"""
v2 

Najvažnije poboljšanje je da početni regresor brojeva ne ostane isti za sve hronološke grupe, 
već da se u svakoj grupi ponovo obuči na proširenoj istoriji dostupnoj do tog trenutka.
Još tri poboljsanja:
- napraviti pravi regresor parova sa kontinuiranom Y_pair metom, umesto ručno ponderisanog parnog skora;
- MAE precizne faze računati samo nad kandidatima koje joj je brza faza zaista prosledila;
- konačnu metodologiju birati nested walk-forward validacijom, pa je zamrznuti za buduća izvlačenja.
Metodološki bolja završna verzija. Ne garantuje bolju loto predikciju, ali uklanja najslabije tačke. 

Te četiri promene zajedno su ugradjene u jednu konačnu verziju _v2:
1. Expanding walk-forward obuka regresora brojeva.
2. Pravi kontinuirani regresor parova.
3. Ispravna procena precizne faze samo nad kandidatima brze faze.
4. Nested hronološka validacija i zatim zamrzavanje izabrane metodologije.
Kompletan predlog najbolje verzije koda.
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
