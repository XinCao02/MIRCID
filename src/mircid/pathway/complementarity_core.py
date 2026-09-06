from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_matrix(path: Path) -> tuple[pd.Index, list[str], np.ndarray]:
    frame = pd.read_csv(path)
    ids = pd.Index(frame.iloc[:, 0].astype(str))
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )
    if ids.duplicated().any():
        raise ValueError(f"Duplicate sample IDs: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values: {path}")
    return ids, [str(column) for column in frame.columns[1:]], values


def load_historical_input(path: Path) -> tuple[pd.Index, list[str], np.ndarray, dict]:
    frame = pd.read_csv(path)
    feature_names = frame.iloc[:, 1].astype(str).tolist()
    numeric = frame.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().sum().sum())
    values = numeric.fillna(0.0).to_numpy(dtype=np.float32).T
    ids = pd.Index(frame.columns[4:].astype(str))
    return ids, feature_names, values, {
        "source_shape": list(frame.shape),
        "matrix_shape": list(values.shape),
        "coerced_non_numeric_cells_to_zero": missing,
        "all_zero_feature_positions_0_based": np.flatnonzero(
            np.all(values == 0.0, axis=0)
        ).tolist(),
    }


def align_to_reference(
    reference_ids: pd.Index, ids: pd.Index, values: np.ndarray, name: str
) -> np.ndarray:
    if set(reference_ids) != set(ids):
        missing = sorted(set(reference_ids) - set(ids))
        extra = sorted(set(ids) - set(reference_ids))
        raise ValueError(f"{name} sample mismatch; missing={missing[:3]}, extra={extra[:3]}")
    positions = ids.get_indexer(reference_ids)
    if (positions < 0).any():
        raise ValueError(f"Failed explicit ID alignment: {name}")
    return values[positions]


def split_indices(
    split_manifest: pd.DataFrame, reference_ids: pd.Index, seed: int
) -> dict[str, np.ndarray]:
    subset = split_manifest.loc[split_manifest["split_seed"] == seed].copy()
    if len(subset) != len(reference_ids) or subset["sample_id"].nunique() != len(
        reference_ids
    ):
        raise ValueError(f"Incomplete split manifest for seed {seed}")
    id_to_position = {sample_id: i for i, sample_id in enumerate(reference_ids)}
    result = {}
    for partition in ["train", "validation", "test"]:
        sample_ids = subset.loc[subset["partition"] == partition, "sample_id"].astype(str)
        result[partition] = np.asarray(
            [id_to_position[sample_id] for sample_id in sample_ids], dtype=np.int64
        )
    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(reference_ids):
        raise ValueError(f"Overlapping or missing split indices for seed {seed}")
    return result


def squared_euclidean_gram(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * values @ values.T
    return np.maximum(distances, 0.0)


def median_squared_distance(values: np.ndarray) -> float:
    distances = squared_euclidean_gram(values)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    if len(positive) == 0:
        raise ValueError("Cannot determine RBF bandwidth from zero distances")
    return float(np.median(positive))


def rbf_gram(values: np.ndarray, median_sq_distance: float) -> np.ndarray:
    gamma = 1.0 / (2.0 * median_sq_distance)
    return np.exp(-gamma * squared_euclidean_gram(values))


def unbiased_hsic(kernel_x: np.ndarray, kernel_y: np.ndarray) -> float:
    if kernel_x.shape != kernel_y.shape or kernel_x.shape[0] < 4:
        raise ValueError("Unbiased HSIC requires equally sized square kernels with n>=4")
    k = np.asarray(kernel_x, dtype=np.float64).copy()
    l = np.asarray(kernel_y, dtype=np.float64).copy()
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(l, 0.0)
    n = k.shape[0]
    trace_term = float(np.sum(k * l))
    sum_term = float(k.sum() * l.sum() / ((n - 1) * (n - 2)))
    row_term = float(2.0 * np.dot(k.sum(axis=1), l.sum(axis=1)) / (n - 2))
    return (trace_term + sum_term - row_term) / (n * (n - 3))


def debiased_cka(kernel_x: np.ndarray, kernel_y: np.ndarray) -> float:
    numerator = unbiased_hsic(kernel_x, kernel_y)
    denominator_x = unbiased_hsic(kernel_x, kernel_x)
    denominator_y = unbiased_hsic(kernel_y, kernel_y)
    if denominator_x <= 0 or denominator_y <= 0:
        return np.nan
    return float(numerator / np.sqrt(denominator_x * denominator_y))


def cka_with_permutation_null(
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    observed = debiased_cka(kernel_x, kernel_y)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        order = rng.permutation(kernel_y.shape[0])
        permuted = kernel_y[np.ix_(order, order)]
        null[index] = debiased_cka(kernel_x, permuted)
    return observed, null


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, (len(ranked) - np.arange(len(ranked))) * ranked)
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def inverse_sqrt(matrix: np.ndarray, ridge: float) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    adjusted = np.maximum(eigenvalues, 0.0) + ridge
    return (eigenvectors * (1.0 / np.sqrt(adjusted))) @ eigenvectors.T


@dataclass
class CCAResult:
    test_correlations: np.ndarray
    train_singular_values: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray


def regularized_cca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridge: float,
    modes: int,
) -> CCAResult:
    n = x_train.shape[0]
    covariance_x = x_train.T @ x_train / (n - 1)
    covariance_y = y_train.T @ y_train / (n - 1)
    cross_covariance = x_train.T @ y_train / (n - 1)
    # Interpret ridge relative to the average retained-PC variance so the same
    # configured value remains meaningful across source spaces and PC counts.
    ridge_x = ridge * float(np.trace(covariance_x) / covariance_x.shape[0])
    ridge_y = ridge * float(np.trace(covariance_y) / covariance_y.shape[0])
    whitening_x = inverse_sqrt(covariance_x, ridge_x)
    whitening_y = inverse_sqrt(covariance_y, ridge_y)
    matrix = whitening_x @ cross_covariance @ whitening_y
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    x_weights = whitening_x @ u[:, :modes]
    y_weights = whitening_y @ vt.T[:, :modes]
    projected_x = x_test @ x_weights
    projected_y = y_test @ y_weights
    correlations = np.empty(modes, dtype=np.float64)
    for mode in range(modes):
        if np.std(projected_x[:, mode]) == 0 or np.std(projected_y[:, mode]) == 0:
            correlations[mode] = np.nan
        else:
            correlations[mode] = np.corrcoef(
                projected_x[:, mode], projected_y[:, mode]
            )[0, 1]
    return CCAResult(
        test_correlations=correlations,
        train_singular_values=singular_values[:modes],
        x_weights=x_weights,
        y_weights=y_weights,
    )


def fit_dual_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> dict:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_train).astype(np.float64)
    y_scaled = y_scaler.fit_transform(y_train).astype(np.float64)
    kernel = x_scaled @ x_scaled.T
    eigenvalues, eigenvectors = eigh(kernel)
    projection = eigenvectors.T @ y_scaled
    dual = eigenvectors @ (projection / (eigenvalues[:, None] + alpha))
    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_train_scaled": x_scaled,
        "dual": dual,
        "alpha": float(alpha),
    }


def predict_dual_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    x_scaled = model["x_scaler"].transform(x).astype(np.float64)
    kernel = x_scaled @ model["x_train_scaled"].T
    predicted_scaled = kernel @ model["dual"]
    return model["y_scaler"].inverse_transform(predicted_scaled)


def cross_validated_dual_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
    folds: int,
    seed: int,
) -> dict:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_by_alpha = {
        float(alpha): np.empty_like(y_train, dtype=np.float64) for alpha in alphas
    }
    fold_assignment = np.full(len(x_train), -1, dtype=np.int64)
    for fold, (inner_train, inner_test) in enumerate(splitter.split(x_train)):
        fold_assignment[inner_test] = fold
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_fit = x_scaler.fit_transform(x_train[inner_train]).astype(np.float64)
        x_holdout = x_scaler.transform(x_train[inner_test]).astype(np.float64)
        y_fit = y_scaler.fit_transform(y_train[inner_train]).astype(np.float64)
        kernel_fit = x_fit @ x_fit.T
        eigenvalues, eigenvectors = eigh(kernel_fit)
        projection = eigenvectors.T @ y_fit
        kernel_holdout = x_holdout @ x_fit.T
        for alpha in alphas:
            dual = eigenvectors @ (
                projection / (eigenvalues[:, None] + float(alpha))
            )
            predicted_scaled = kernel_holdout @ dual
            oof_by_alpha[float(alpha)][inner_test] = y_scaler.inverse_transform(
                predicted_scaled
            )
    if (fold_assignment < 0).any():
        raise RuntimeError("OOF prediction coverage is incomplete")
    scores = {
        alpha: float(
            r2_score(y_train, prediction, multioutput="variance_weighted")
        )
        for alpha, prediction in oof_by_alpha.items()
    }
    best_alpha = max(alphas, key=lambda alpha: (scores[float(alpha)], -float(alpha)))
    final_model = fit_dual_ridge(x_train, y_train, float(best_alpha))
    return {
        "best_alpha": float(best_alpha),
        "alpha_scores": scores,
        "oof_prediction": oof_by_alpha[float(best_alpha)],
        "validation_prediction": predict_dual_ridge(final_model, x_validation),
        "test_prediction": predict_dual_ridge(final_model, x_test),
        "fold_assignment": fold_assignment,
        "final_model": final_model,
    }


def predictability_metrics(
    y_true: np.ndarray, y_predicted: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    per_feature = r2_score(y_true, y_predicted, multioutput="raw_values")
    metrics = {
        "variance_weighted_r2": float(
            r2_score(y_true, y_predicted, multioutput="variance_weighted")
        ),
        "uniform_average_r2": float(
            r2_score(y_true, y_predicted, multioutput="uniform_average")
        ),
        "median_per_feature_r2": float(np.median(per_feature)),
        "fraction_features_r2_gt_0.5": float(np.mean(per_feature > 0.5)),
        "fraction_features_r2_gt_0.8": float(np.mean(per_feature > 0.8)),
        "rmse": float(np.sqrt(np.mean((y_true - y_predicted) ** 2))),
    }
    return metrics, per_feature


def exact_sign_flip_pvalue(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    observed = abs(float(np.mean(delta)))
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(delta)):
        null.append(abs(float(np.mean(delta * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def paired_bootstrap_ci(
    delta: np.ndarray, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(iterations, len(delta)))
    means = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)
