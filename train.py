#!/usr/bin/env python3

"""
Neural-network overtopping predictor based on `database.csv`.

USAGE
-----
This script supports two command-line modes:

1) Train a model
   python train.py train --database database.csv --model model.joblib --diagnostics diagnostics.json

2) Predict with an existing model
   python train.py predict --model model.joblib --output predictions.csv --name 026-004 ^
       --m 14 --beta 0 --h 0.181 --hm0-toe 0.091 --tm-1-0-toe 1.16 ^
       --ht 0.181 --bt 0 --gf 1 --cotad 0.35 --cotau 0.33 ^
       --berm-width 0.792 --hb -0.002 --rc 0.069 --ac 0.069 --gc 0

3) Predict from an input scenario file
   python train.py predict --model model.joblib --from-inp input.txt --output predictions.csv

4) Predict from a CSV / semicolon-separated batch file
   python train.py predict --model model.joblib --from-csv scenarios.csv --output predictions.csv

5) Predict with auto-training if the model file does not yet exist
   python train.py predict --model model.joblib --database database.csv --from-csv scenarios.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# MAIN MODEL / TRAINING PARAMETERS
# =============================================================================
GRAVITY = 9.80665
EPS = 1.0e-12
DEFAULT_SQ_FLOOR = 1.0e-9
DEFAULT_N_MODELS = 50
DEFAULT_HIDDEN_LAYERS: Tuple[int, ...] = (192, 96, 48, 24)
DEFAULT_MAX_ITER = 100000
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_MLP_ALPHA = 0.0003
DEFAULT_MLP_LEARNING_RATE_INIT = 0.00075
DEFAULT_MLP_VALIDATION_FRACTION = 0.12
DEFAULT_MLP_N_ITER_NO_CHANGE = 20

DEFAULT_MODEL_PATH = Path("model.joblib")
DEFAULT_DIAGNOSTICS_PATH = Path("diagnostics.json")
DEFAULT_OUTPUT_PATH = Path("predictions.csv")

DATABASE_FEATURES: List[str] = [
    "m",
    "b",
    "h",
    "Hm0 toe",
    "Tm-1,0 toe",
    "ht",
    "Bt",
    "gf",
    "cotad",
    "cotau",
    "B",
    "hb",
    "Rc",
    "Ac",
    "Gc",
]

NON_NUMERIC_DB_COLUMNS = {"Name", "Remark", "Reference", "blank_1", "blank_2", "blank_3"}

CLI_TO_DB = {
    "m": "m",
    "beta": "b",
    "h": "h",
    "hm0_toe": "Hm0 toe",
    "tm10_toe": "Tm-1,0 toe",
    "ht": "ht",
    "bt": "Bt",
    "gf": "gf",
    "cotad": "cotad",
    "cotau": "cotau",
    "berm_width": "B",
    "hb": "hb",
    "rc": "Rc",
    "ac": "Ac",
    "gc": "Gc",
}


@dataclass
class ModelBundle:
    feature_columns: List[str]
    sq_floor: float
    models: List[Pipeline]
    feature_ranges: Dict[str, Tuple[float, float]]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]

    def _prepare_model_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        model_frame = build_model_feature_frame(frame)
        for col in self.feature_columns:
            if col not in model_frame.columns:
                model_frame[col] = np.nan
        return model_frame[self.feature_columns]

    def _predict_log10_sq_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        model_input = self._prepare_model_input(frame)
        return np.vstack([model.predict(model_input) for model in self.models])

    def predict_sq_distribution(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred_log_sq_matrix = self._predict_log10_sq_matrix(frame)
        pred_sq_matrix = np.power(10.0, pred_log_sq_matrix)
        mean_sq = np.mean(pred_sq_matrix, axis=0)
        p05_sq = np.percentile(pred_sq_matrix, 5.0, axis=0)
        p50_sq = np.percentile(pred_sq_matrix, 50.0, axis=0)
        p95_sq = np.percentile(pred_sq_matrix, 95.0, axis=0)
        return mean_sq, p05_sq, p50_sq, p95_sq

    def predict_q_distribution_lpsm(
        self, frame: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred_log_sq_matrix = self._predict_log10_sq_matrix(frame)
        pred_sq_matrix = np.power(10.0, pred_log_sq_matrix)

        hm0 = pd.to_numeric(frame["Hm0 toe"], errors="coerce").to_numpy(dtype=float)
        scale = overtopping_scale(hm0)

        pred_q_matrix = pred_sq_matrix * scale[None, :]
        pred_lpsm_matrix = 1000.0 * pred_q_matrix

        mean_sq = np.mean(pred_sq_matrix, axis=0)
        p05_sq = np.percentile(pred_sq_matrix, 5.0, axis=0)
        p50_sq = np.percentile(pred_sq_matrix, 50.0, axis=0)
        p95_sq = np.percentile(pred_sq_matrix, 95.0, axis=0)

        mean_lpsm = np.nanmean(pred_lpsm_matrix, axis=0)
        p05_lpsm = np.nanpercentile(pred_lpsm_matrix, 5.0, axis=0)
        p50_lpsm = np.nanpercentile(pred_lpsm_matrix, 50.0, axis=0)
        p95_lpsm = np.nanpercentile(pred_lpsm_matrix, 95.0, axis=0)
        return mean_sq, p05_sq, p50_sq, p95_sq, mean_lpsm, p05_lpsm, p50_lpsm, p95_lpsm


@dataclass
class TrainingArtifacts:
    bundle: ModelBundle
    holdout: Dict[str, Any]
    full_data: Dict[str, Any]


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPS)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def overtopping_scale(hm0: np.ndarray) -> np.ndarray:
    hm0 = np.asarray(hm0, dtype=float)
    scale = np.sqrt(GRAVITY * np.power(hm0, 3))
    invalid = (~np.isfinite(hm0)) | (hm0 <= 0.0)
    if np.any(invalid):
        scale = scale.astype(float)
        scale[invalid] = np.nan
    return scale


def _compute_fit_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = _finite_mask(y_true, y_pred)
    if not np.any(mask):
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "medae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "fit_slope": float("nan"),
            "fit_intercept": float("nan"),
            "n": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    slope = float("nan")
    intercept = float("nan")
    if yt.size >= 2:
        slope, intercept = np.polyfit(yt, yp, 1)
        slope = float(slope)
        intercept = float(intercept)

    return {
        "r2": float(r2_score(yt, yp)) if yt.size >= 2 else float("nan"),
        "mae": float(mean_absolute_error(yt, yp)),
        "medae": float(median_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "bias": float(np.mean(yp - yt)),
        "fit_slope": slope,
        "fit_intercept": intercept,
        "n": int(yt.size),
    }


def _format_box_lines(
    stats: Dict[str, float],
    bundle: ModelBundle,
    sample_label: str,
    n_holdout: Optional[int] = None,
    n_full: Optional[int] = None,
) -> str:
    lines = [
        f"R² = {stats['r2']:.4f}" if math.isfinite(stats["r2"]) else "R² = nan",
        f"MAE = {stats['mae']:.4g}" if math.isfinite(stats["mae"]) else "MAE = nan",
        f"MedAE = {stats['medae']:.4g}" if math.isfinite(stats["medae"]) else "MedAE = nan",
        f"RMSE = {stats['rmse']:.4g}" if math.isfinite(stats["rmse"]) else "RMSE = nan",
        f"bias = {stats['bias']:.4g}" if math.isfinite(stats["bias"]) else "bias = nan",
        f"fit slope = {stats['fit_slope']:.4g}" if math.isfinite(stats["fit_slope"]) else "fit slope = nan",
        f"fit intercept = {stats['fit_intercept']:.4g}" if math.isfinite(stats["fit_intercept"]) else "fit intercept = nan",
        f"{sample_label} = {stats['n']}",
    ]
    if n_holdout is not None and sample_label != "n_holdout":
        lines.append(f"n_holdout = {n_holdout}")
    if n_full is not None and sample_label != "n_full":
        lines.append(f"n_full = {n_full}")
    lines.extend(
        [
            f"n_models = {bundle.metrics.get('n_models')}",
            f"hidden_layers = {bundle.metrics.get('hidden_layers')}",
            f"max_iter = {bundle.metrics.get('max_iter')}",
            f"sq_floor = {bundle.metrics.get('sq_floor'):.3e}",
            f"alpha = {bundle.metrics.get('mlp_alpha'):.3e}",
            f"learning_rate_init = {bundle.metrics.get('mlp_learning_rate_init'):.3e}",
            f"engineered = {bundle.metrics.get('n_engineered_features')}",
        ]
    )
    return "\n".join(lines)


def _annotate_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "black"},
    )


def _safe_axis_limits(
    values: np.ndarray,
    lower_pad_fraction: float = 0.0,
    upper_pad_fraction: float = 0.05,
) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        delta = 1.0 if math.isclose(vmin, 0.0) else 0.05 * abs(vmin)
        return (vmin - lower_pad_fraction * delta, vmax + delta)
    span = vmax - vmin
    return (vmin - lower_pad_fraction * span, vmax + upper_pad_fraction * span)


def _safe_log_axis_limits(
    values: np.ndarray,
    lower_exact: bool = True,
    upper_pad_fraction: float = 0.03,
) -> Tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return (1.0e-12, 1.0)
    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if math.isclose(vmin, vmax):
        return (vmin if lower_exact else vmin / 1.2, vmax * 1.2)
    lower = vmin if lower_exact else vmin / (1.0 + upper_pad_fraction)
    upper = vmax * (1.0 + upper_pad_fraction)
    return (lower, upper)


def _build_prediction_payload(
    true_log10_sq: np.ndarray,
    pred_log10_sq: np.ndarray,
    true_sq: np.ndarray,
    pred_sq: np.ndarray,
    true_q_lpsm: np.ndarray,
    pred_q_lpsm: np.ndarray,
) -> Dict[str, Any]:
    return {
        "true_log10_sq": np.asarray(true_log10_sq, dtype=float).tolist(),
        "pred_log10_sq": np.asarray(pred_log10_sq, dtype=float).tolist(),
        "true_sq": np.asarray(true_sq, dtype=float).tolist(),
        "pred_sq": np.asarray(pred_sq, dtype=float).tolist(),
        "true_q_lpsm": np.asarray(true_q_lpsm, dtype=float).tolist(),
        "pred_q_lpsm": np.asarray(pred_q_lpsm, dtype=float).tolist(),
    }


def save_diagnostic_plots(
    bundle: ModelBundle,
    payload: Dict[str, Any],
    plot_dir: Path,
    key_prefix: str,
    title_suffix: str,
    sample_label: str,
    plot_mean_log10_sq: float,
    plot_std_log10_sq: float,
    n_holdout: Optional[int] = None,
    n_full: Optional[int] = None,
) -> Dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    true_log10_sq = np.asarray(payload["true_log10_sq"], dtype=float)
    pred_log10_sq = np.asarray(payload["pred_log10_sq"], dtype=float)
    true_sq = np.asarray(payload["true_sq"], dtype=float)
    pred_sq = np.asarray(payload["pred_sq"], dtype=float)
    true_q_lpsm = np.asarray(payload["true_q_lpsm"], dtype=float)
    pred_q_lpsm = np.asarray(payload["pred_q_lpsm"], dtype=float)

    true_y = _standardize_for_plot(true_log10_sq, plot_mean_log10_sq, plot_std_log10_sq)
    pred_y = _standardize_for_plot(pred_log10_sq, plot_mean_log10_sq, plot_std_log10_sq)

    stats_y = _compute_fit_stats(true_y, pred_y)
    stats_sq = _compute_fit_stats(true_sq, pred_sq)
    stats_q = _compute_fit_stats(true_q_lpsm, pred_q_lpsm)
    residual_y = pred_y - true_y

    box_y = _format_box_lines(stats_y, bundle, sample_label, n_holdout=n_holdout, n_full=n_full)
    box_sq = _format_box_lines(stats_sq, bundle, sample_label, n_holdout=n_holdout, n_full=n_full)
    box_q = _format_box_lines(stats_q, bundle, sample_label, n_holdout=n_holdout, n_full=n_full)

    out: Dict[str, str] = {}

    # 1) True vs predicted standardized target y
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(true_y, pred_y, s=22, alpha=0.7)
    x_lo, x_hi = _safe_axis_limits(true_y, lower_pad_fraction=0.0, upper_pad_fraction=0.03)
    y_lo, y_hi = _safe_axis_limits(np.concatenate([true_y, pred_y]), 0.03, 0.03)
    line_lo = min(float(np.nanmin(true_y)), float(np.nanmin(pred_y)))
    line_hi = max(float(np.nanmax(true_y)), float(np.nanmax(pred_y)))
    ax.plot([line_lo, line_hi], [line_lo, line_hi], "k--", linewidth=1.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Observed y = (log10(sq) - mu) / sigma")
    ax.set_ylabel("Predicted y = (log10(sq) - mu) / sigma")
    ax.set_title(f"Fit: y [{title_suffix}]")
    ax.grid(True, alpha=0.3)
    _annotate_box(ax, box_y)
    path = plot_dir / f"{key_prefix}fit_y.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out[f"{key_prefix}fit_y"] = str(path)

    # 2) True vs predicted sq
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(true_sq, pred_sq, s=22, alpha=0.7)
    x_vmin, x_vmax = _safe_log_axis_limits(true_sq, lower_exact=True, upper_pad_fraction=0.03)
    y_vmin, y_vmax = _safe_log_axis_limits(np.concatenate([true_sq, pred_sq]), lower_exact=False, upper_pad_fraction=0.03)
    line_lo = min(x_vmin, y_vmin)
    line_hi = max(x_vmax, y_vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([line_lo, line_hi], [line_lo, line_hi], "k--", linewidth=1.0)
    ax.set_xlim(x_vmin, x_vmax)
    ax.set_ylim(y_vmin, y_vmax)
    ax.set_xlabel("Observed sq [-]")
    ax.set_ylabel("Predicted sq [-]")
    ax.set_title(f"Fit: sq [{title_suffix}]")
    ax.grid(True, alpha=0.3)
    _annotate_box(ax, box_sq)
    path = plot_dir / f"{key_prefix}fit_sq.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out[f"{key_prefix}fit_sq"] = str(path)

    # 3) True vs predicted q in l/s/m
    fig, ax = plt.subplots(figsize=(9, 7))
    mask_q = _finite_mask(true_q_lpsm, pred_q_lpsm)
    tq_plot = np.maximum(true_q_lpsm[mask_q], 1.0e-12)
    pq_plot = np.maximum(pred_q_lpsm[mask_q], 1.0e-12)
    ax.scatter(tq_plot, pq_plot, s=22, alpha=0.7)
    x_vmin, x_vmax = _safe_log_axis_limits(tq_plot, lower_exact=True, upper_pad_fraction=0.03)
    y_vmin, y_vmax = _safe_log_axis_limits(np.concatenate([tq_plot, pq_plot]), lower_exact=False, upper_pad_fraction=0.03)
    line_lo = min(x_vmin, y_vmin)
    line_hi = max(x_vmax, y_vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([line_lo, line_hi], [line_lo, line_hi], "k--", linewidth=1.0)
    ax.set_xlim(x_vmin, x_vmax)
    ax.set_ylim(y_vmin, y_vmax)
    ax.set_xlabel("Observed q [l/s/m]")
    ax.set_ylabel("Predicted q [l/s/m]")
    ax.set_title(f"Fit: q [{title_suffix}]")
    ax.grid(True, alpha=0.3)
    _annotate_box(ax, box_q)
    path = plot_dir / f"{key_prefix}fit_q_lpsm.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out[f"{key_prefix}fit_q_lpsm"] = str(path)

    # 4) Residuals in standardized target y
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(true_y, residual_y, s=22, alpha=0.7)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    x_lo, x_hi = _safe_axis_limits(true_y, lower_pad_fraction=0.0, upper_pad_fraction=0.03)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Observed y = (log10(sq) - mu) / sigma")
    ax.set_ylabel("Residual in y = predicted - observed")
    ax.set_title(f"Residuals: y [{title_suffix}]")
    ax.grid(True, alpha=0.3)
    _annotate_box(ax, box_y)
    path = plot_dir / f"{key_prefix}residuals_y.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out[f"{key_prefix}residuals_y"] = str(path)

    # 5) Residual histogram in standardized target y
    fig, ax = plt.subplots(figsize=(9, 7))
    finite_res = residual_y[np.isfinite(residual_y)]
    bins = min(40, max(10, int(np.sqrt(max(1, finite_res.size)) * 2)))
    ax.hist(finite_res, bins=bins)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Residual in y = (log10(sq) - mu) / sigma")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual histogram: y [{title_suffix}]")
    ax.grid(True, alpha=0.3)
    _annotate_box(ax, box_y)
    path = plot_dir / f"{key_prefix}residual_histogram_y.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out[f"{key_prefix}residual_histogram_y"] = str(path)

    return out


def load_database(database_path: Path) -> pd.DataFrame:
    with database_path.open("r", encoding="latin1", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError(f"Database file '{database_path}' does not contain enough rows.")

    header = rows[0]
    data_rows = rows[2:]
    frame = pd.DataFrame(data_rows, columns=header)

    renamed: List[str] = []
    blank_count = 0
    for col in frame.columns:
        if col == "":
            blank_count += 1
            renamed.append(f"blank_{blank_count}")
        else:
            renamed.append(col)
    frame.columns = renamed

    for col in frame.columns:
        if col in NON_NUMERIC_DB_COLUMNS:
            frame[col] = frame[col].astype(str).str.strip()
            frame.loc[frame[col] == "", col] = np.nan
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def build_training_frame(database: pd.DataFrame, sq_floor: float) -> pd.DataFrame:
    required = DATABASE_FEATURES + ["q"]
    missing_cols = [col for col in required if col not in database.columns]
    if missing_cols:
        raise ValueError(f"Database is missing required columns: {missing_cols}")

    train = database[required].copy()
    train = train[train["q"].notna()].copy()
    train = train[train["Hm0 toe"].notna()].copy()
    train = train[np.isfinite(train["Hm0 toe"])].copy()
    train = train[train["Hm0 toe"] > 0.0].copy()
    train["q"] = train["q"].clip(lower=0.0)
    scale = overtopping_scale(train["Hm0 toe"].to_numpy(dtype=float))
    train["sq"] = train["q"].to_numpy(dtype=float) / scale
    train["sq"] = np.where(np.isfinite(train["sq"]), train["sq"], np.nan)
    train = train[train["sq"].notna()].copy()
    train["sq"] = train["sq"].clip(lower=0.0)
    train["sq_train"] = train["sq"].clip(lower=sq_floor)
    return train


def build_model_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame.copy()

    for col in DATABASE_FEATURES:
        if col not in feature_frame.columns:
            feature_frame[col] = np.nan
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")

    tm10 = feature_frame["Tm-1,0 toe"].to_numpy(dtype=float)
    hm0 = feature_frame["Hm0 toe"].to_numpy(dtype=float)
    beta_deg = feature_frame["b"].to_numpy(dtype=float)

    lm10 = np.full(len(feature_frame), np.nan, dtype=float)
    valid_tm10 = np.isfinite(tm10) & (tm10 > 0.0)
    lm10[valid_tm10] = (GRAVITY * np.power(tm10[valid_tm10], 2)) / (2 * np.pi)

    l0 = np.full(len(feature_frame), np.nan, dtype=float)
    l0[valid_tm10] = GRAVITY * np.power(tm10[valid_tm10], 2) / (2.0 * np.pi)
    wave_steepness = _safe_divide(hm0, l0)
    sqrt_wave_steepness = np.sqrt(np.clip(wave_steepness, 0.0, None))

    tanad = _safe_divide(np.ones(len(feature_frame), dtype=float), feature_frame["cotad"].to_numpy(dtype=float))
    tanau = _safe_divide(np.ones(len(feature_frame), dtype=float), feature_frame["cotau"].to_numpy(dtype=float))

    feature_frame["Lm-1,0 toe"] = lm10
    feature_frame["L0 toe"] = l0
    feature_frame["wave_steepness"] = wave_steepness
    feature_frame["sqrt_wave_steepness"] = sqrt_wave_steepness
    feature_frame["xi_m10_lower"] = _safe_divide(tanad, sqrt_wave_steepness)
    feature_frame["xi_m10_upper"] = _safe_divide(tanau, sqrt_wave_steepness)

    feature_frame["h_over_Lm10_toe"] = _safe_divide(feature_frame["h"].to_numpy(dtype=float), lm10)
    feature_frame["Hm0_toe_over_Lm10_toe"] = _safe_divide(hm0, lm10)
    feature_frame["ht_over_Hm0_toe"] = _safe_divide(feature_frame["ht"].to_numpy(dtype=float), hm0)
    feature_frame["Bt_over_Lm10_toe"] = _safe_divide(feature_frame["Bt"].to_numpy(dtype=float), lm10)
    feature_frame["B_over_Lm10_toe"] = _safe_divide(feature_frame["B"].to_numpy(dtype=float), lm10)
    feature_frame["hb_over_Hm0_toe"] = _safe_divide(feature_frame["hb"].to_numpy(dtype=float), hm0)
    feature_frame["Rc_over_Hm0_toe"] = _safe_divide(feature_frame["Rc"].to_numpy(dtype=float), hm0)
    feature_frame["Ac_over_Hm0_toe"] = _safe_divide(feature_frame["Ac"].to_numpy(dtype=float), hm0)
    feature_frame["Gc_over_Lm10_toe"] = _safe_divide(feature_frame["Gc"].to_numpy(dtype=float), lm10)
    feature_frame["h_over_Hm0_toe"] = _safe_divide(feature_frame["h"].to_numpy(dtype=float), hm0)
    feature_frame["Bt_over_Hm0_toe"] = _safe_divide(feature_frame["Bt"].to_numpy(dtype=float), hm0)
    feature_frame["B_over_Hm0_toe"] = _safe_divide(feature_frame["B"].to_numpy(dtype=float), hm0)
    feature_frame["Gc_over_Hm0_toe"] = _safe_divide(feature_frame["Gc"].to_numpy(dtype=float), hm0)
    feature_frame["freeboard_sum_over_Hm0_toe"] = _safe_divide(
        feature_frame["Rc"].to_numpy(dtype=float) + feature_frame["Ac"].to_numpy(dtype=float),
        hm0,
    )
    feature_frame["Rc_over_Ac"] = _safe_divide(
        feature_frame["Rc"].to_numpy(dtype=float),
        feature_frame["Ac"].to_numpy(dtype=float),
    )
    feature_frame["Bt_over_B"] = _safe_divide(
        feature_frame["Bt"].to_numpy(dtype=float),
        feature_frame["B"].to_numpy(dtype=float),
    )

    feature_frame["beta_abs"] = np.abs(beta_deg)
    feature_frame["cos_beta"] = np.cos(np.deg2rad(beta_deg))
    feature_frame["sin_beta_abs"] = np.abs(np.sin(np.deg2rad(beta_deg)))
    feature_frame["gf_cos_beta"] = feature_frame["gf"].to_numpy(dtype=float) * feature_frame["cos_beta"].to_numpy(dtype=float)

    return feature_frame


def _make_stratification_bins(values: np.ndarray, n_bins: int = 12) -> Optional[np.ndarray]:
    try:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
    except ValueError:
        return None
    if hasattr(bins, "codes") and np.unique(bins.codes[bins.codes >= 0]).size >= 2:
        return np.asarray(bins.codes, dtype=int)
    return None


def make_pipeline(random_state: int, hidden_layers: Tuple[int, ...], max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=DEFAULT_MLP_ALPHA,
                    learning_rate_init=DEFAULT_MLP_LEARNING_RATE_INIT,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=DEFAULT_MLP_VALIDATION_FRACTION,
                    n_iter_no_change=DEFAULT_MLP_N_ITER_NO_CHANGE,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_mlp_ensemble(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    n_models: int,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    random_state: int,
) -> List[Pipeline]:
    rng = np.random.default_rng(random_state)
    models: List[Pipeline] = []

    for idx in range(n_models):
        model = make_pipeline(
            random_state=random_state + idx,
            hidden_layers=hidden_layers,
            max_iter=max_iter,
        )

        if n_models > 1:
            sample_idx = rng.integers(low=0, high=len(X_fit), size=len(X_fit))
            X_boot = X_fit.iloc[sample_idx].reset_index(drop=True)
            y_boot = y_fit[sample_idx]
        else:
            X_boot = X_fit
            y_boot = y_fit

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_boot, y_boot)
        models.append(model)

    return models


def _fit_plot_standardization(log10_sq: np.ndarray) -> Tuple[float, float]:
    log10_sq = np.asarray(log10_sq, dtype=float)
    finite = log10_sq[np.isfinite(log10_sq)]
    if finite.size == 0:
        return 0.0, 1.0
    mean_log10_sq = float(np.mean(finite))
    std_log10_sq = float(np.std(finite, ddof=0))
    if (not math.isfinite(std_log10_sq)) or std_log10_sq <= 0.0:
        std_log10_sq = 1.0
    return mean_log10_sq, std_log10_sq


def _standardize_for_plot(log10_sq: np.ndarray, mean_log10_sq: float, std_log10_sq: float) -> np.ndarray:
    log10_sq = np.asarray(log10_sq, dtype=float)
    std = float(std_log10_sq) if math.isfinite(std_log10_sq) and std_log10_sq > 0.0 else 1.0
    return (log10_sq - float(mean_log10_sq)) / std


def _predict_sq_mean(models: List[Pipeline], X: pd.DataFrame, sq_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    pred_log_sq_matrix = np.vstack([model.predict(X) for model in models])
    pred_sq_matrix = np.power(10.0, pred_log_sq_matrix)
    pred_sq_mean = np.mean(pred_sq_matrix, axis=0)
    pred_log_sq_mean = np.log10(np.clip(pred_sq_mean, sq_floor, None))
    return pred_log_sq_mean, pred_sq_mean


def train_model_bundle(
    database_path: Path,
    sq_floor: float = DEFAULT_SQ_FLOOR,
    n_models: int = DEFAULT_N_MODELS,
    hidden_layers: Tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
) -> TrainingArtifacts:
    database = load_database(database_path)
    train_frame = build_training_frame(database, sq_floor=sq_floor)

    model_feature_frame = build_model_feature_frame(train_frame[DATABASE_FEATURES].copy())
    model_feature_columns = list(model_feature_frame.columns)
    X_full = model_feature_frame[model_feature_columns].copy()
    y_log_sq_full = np.log10(train_frame["sq_train"].to_numpy(dtype=float))

    stratify_bins = _make_stratification_bins(y_log_sq_full)
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X_full,
        y_log_sq_full,
        train_frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_bins,
    )

    holdout_target_mean_log10_sq, holdout_target_std_log10_sq = _fit_plot_standardization(y_train)

    eval_models = _fit_mlp_ensemble(
        X_fit=X_train.reset_index(drop=True),
        y_fit=np.asarray(y_train, dtype=float),
        n_models=n_models,
        hidden_layers=hidden_layers,
        max_iter=max_iter,
        random_state=random_state,
    )

    pred_log_sq_mean, pred_sq_mean = _predict_sq_mean(eval_models, X_test, sq_floor=sq_floor)
    true_sq = np.power(10.0, y_test)

    hm0_test = raw_test["Hm0 toe"].to_numpy(dtype=float)
    q_scale_test = overtopping_scale(hm0_test)
    true_q_m3s_per_m = true_sq * q_scale_test
    pred_q_m3s_per_m = pred_sq_mean * q_scale_test
    true_q_lpsm = 1000.0 * true_q_m3s_per_m
    pred_q_lpsm = 1000.0 * pred_q_m3s_per_m

    full_target_mean_log10_sq, full_target_std_log10_sq = _fit_plot_standardization(y_log_sq_full)

    final_models = _fit_mlp_ensemble(
        X_fit=X_full.reset_index(drop=True),
        y_fit=np.asarray(y_log_sq_full, dtype=float),
        n_models=n_models,
        hidden_layers=hidden_layers,
        max_iter=max_iter,
        random_state=random_state,
    )

    pred_log_sq_full_mean, pred_sq_full_mean = _predict_sq_mean(final_models, X_full, sq_floor=sq_floor)
    true_sq_full = np.power(10.0, y_log_sq_full)
    hm0_full = train_frame["Hm0 toe"].to_numpy(dtype=float)
    q_scale_full = overtopping_scale(hm0_full)
    true_q_full_m3s_per_m = true_sq_full * q_scale_full
    pred_q_full_m3s_per_m = pred_sq_full_mean * q_scale_full
    true_q_full_lpsm = 1000.0 * true_q_full_m3s_per_m
    pred_q_full_lpsm = 1000.0 * pred_q_full_m3s_per_m

    feature_ranges: Dict[str, Tuple[float, float]] = {}
    for col in DATABASE_FEATURES:
        finite_values = train_frame[col].to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            feature_ranges[col] = (float("nan"), float("nan"))
        else:
            feature_ranges[col] = (float(np.min(finite_values)), float(np.max(finite_values)))

    stats_log10_sq = _compute_fit_stats(np.asarray(y_test, dtype=float), pred_log_sq_mean)
    stats_sq = _compute_fit_stats(true_sq, pred_sq_mean)
    stats_q = _compute_fit_stats(true_q_lpsm, pred_q_lpsm)

    metrics: Dict[str, Any] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "holdout_rows": int(len(X_test)),
        "full_rows": int(len(X_full)),
        "database_rows_with_valid_q_and_hm0": int(len(train_frame)),
        "r2_log10_sq": float(stats_log10_sq["r2"]),
        "mae_log10_sq": float(stats_log10_sq["mae"]),
        "median_ae_log10_sq": float(stats_log10_sq["medae"]),
        "rmse_log10_sq": float(stats_log10_sq["rmse"]),
        "r2_sq": float(stats_sq["r2"]),
        "mae_sq": float(stats_sq["mae"]),
        "median_ae_sq": float(stats_sq["medae"]),
        "r2_q_lpsm": float(stats_q["r2"]),
        "mae_q_lpsm": float(stats_q["mae"]),
        "median_ae_q_lpsm": float(stats_q["medae"]),
        "rmse_q_lpsm": float(stats_q["rmse"]),
        "bias_q_lpsm": float(stats_q["bias"]),
        "sq_floor": float(sq_floor),
        "n_models": int(n_models),
        "hidden_layers": list(hidden_layers),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "n_engineered_features": int(len(model_feature_columns) - len(DATABASE_FEATURES)),
        "mlp_alpha": float(DEFAULT_MLP_ALPHA),
        "mlp_learning_rate_init": float(DEFAULT_MLP_LEARNING_RATE_INIT),
        "mlp_validation_fraction": float(DEFAULT_MLP_VALIDATION_FRACTION),
        "mlp_n_iter_no_change": int(DEFAULT_MLP_N_ITER_NO_CHANGE),
        "saved_model_refit_on_full_dataset": True,
        "holdout_target_mean_log10_sq": float(holdout_target_mean_log10_sq),
        "holdout_target_std_log10_sq": float(holdout_target_std_log10_sq),
        "full_target_mean_log10_sq": float(full_target_mean_log10_sq),
        "full_target_std_log10_sq": float(full_target_std_log10_sq),
    }

    metadata: Dict[str, Any] = {
        "database_path": str(database_path),
        "model_type": "Bagged MLPRegressor ensemble on log10(sq) with engineered hydraulic features",
        "feature_columns": model_feature_columns,
        "base_feature_columns": DATABASE_FEATURES,
        "target_column": "sq",
        "target_definition": "sq = q / sqrt(g * Hm0_toe^3)",
        "diagnostic_target_definition": "y = (log10(sq) - mu) / sigma",
        "target_units": "-",
        "reported_prediction_units": {
            "sq": "-",
            "q_m3_per_s_per_m": "m^3/s/m",
            "q_l_per_s_per_m": "l/s/m",
        },
    }

    bundle = ModelBundle(
        feature_columns=model_feature_columns,
        sq_floor=sq_floor,
        models=final_models,
        feature_ranges=feature_ranges,
        metrics=metrics,
        metadata=metadata,
    )

    holdout = _build_prediction_payload(
        true_log10_sq=np.asarray(y_test, dtype=float),
        pred_log10_sq=pred_log_sq_mean,
        true_sq=true_sq,
        pred_sq=pred_sq_mean,
        true_q_lpsm=true_q_lpsm,
        pred_q_lpsm=pred_q_lpsm,
    )
    full_data = _build_prediction_payload(
        true_log10_sq=np.asarray(y_log_sq_full, dtype=float),
        pred_log10_sq=pred_log_sq_full_mean,
        true_sq=true_sq_full,
        pred_sq=pred_sq_full_mean,
        true_q_lpsm=true_q_full_lpsm,
        pred_q_lpsm=pred_q_full_lpsm,
    )
    return TrainingArtifacts(bundle=bundle, holdout=holdout, full_data=full_data)


def save_model_bundle(bundle: ModelBundle, output_path: Path) -> None:
    payload = {
        "feature_columns": bundle.feature_columns,
        "sq_floor": bundle.sq_floor,
        "models": bundle.models,
        "feature_ranges": bundle.feature_ranges,
        "metrics": bundle.metrics,
        "metadata": bundle.metadata,
    }
    joblib.dump(payload, output_path)


def load_model_bundle(model_path: Path) -> ModelBundle:
    payload = joblib.load(model_path)
    sq_floor = payload.get("sq_floor", payload.get("q_floor", DEFAULT_SQ_FLOOR))
    return ModelBundle(
        feature_columns=list(payload["feature_columns"]),
        sq_floor=float(sq_floor),
        models=list(payload["models"]),
        feature_ranges=dict(payload["feature_ranges"]),
        metrics=dict(payload["metrics"]),
        metadata=dict(payload["metadata"]),
    )


def write_diagnostics(
    bundle: ModelBundle,
    output_path: Path,
    holdout: Optional[Dict[str, Any]] = None,
    full_data: Optional[Dict[str, Any]] = None,
) -> None:
    plot_paths: Dict[str, str] = {}
    plot_dir = output_path.with_name("plots")
    n_holdout = int(len(holdout["true_log10_sq"])) if holdout is not None else None
    n_full = int(len(full_data["true_log10_sq"])) if full_data is not None else None
    holdout_target_mean_log10_sq = float(bundle.metrics.get("holdout_target_mean_log10_sq", 0.0))
    holdout_target_std_log10_sq = float(bundle.metrics.get("holdout_target_std_log10_sq", 1.0))
    full_target_mean_log10_sq = float(bundle.metrics.get("full_target_mean_log10_sq", 0.0))
    full_target_std_log10_sq = float(bundle.metrics.get("full_target_std_log10_sq", 1.0))

    if holdout is not None:
        plot_paths.update(
            save_diagnostic_plots(
                bundle,
                holdout,
                plot_dir,
                key_prefix="holdout_",
                title_suffix="holdout",
                sample_label="n_holdout",
                plot_mean_log10_sq=holdout_target_mean_log10_sq,
                plot_std_log10_sq=holdout_target_std_log10_sq,
                n_holdout=n_holdout,
                n_full=n_full,
            )
        )
    if full_data is not None:
        plot_paths.update(
            save_diagnostic_plots(
                bundle,
                full_data,
                plot_dir,
                key_prefix="full_",
                title_suffix="full refit",
                sample_label="n_full",
                plot_mean_log10_sq=full_target_mean_log10_sq,
                plot_std_log10_sq=full_target_std_log10_sq,
                n_holdout=n_holdout,
                n_full=n_full,
            )
        )

    diagnostics = {
        "metrics": bundle.metrics,
        "metadata": bundle.metadata,
        "feature_ranges": {
            key: {"min": value[0], "max": value[1]} for key, value in bundle.feature_ranges.items()
        },
        "plot_paths": plot_paths,
    }
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def _fetch_any(row: Dict[str, Any], *candidates: str) -> Any:
    normalized = {_normalize_name(key): value for key, value in row.items()}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in normalized:
            return normalized[key]
    return np.nan


def parse_inp_file(inp_path: Path) -> pd.DataFrame:
    with inp_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input file '{inp_path}' contains no scenarios.")

    records: List[Dict[str, Any]] = []
    for row in rows:
        gamma_d = _safe_float(_fetch_any(row, "gammaf_d", "gammafd", "γfd"))
        gamma_u = _safe_float(_fetch_any(row, "gammaf_u", "gammafu", "γfu"))
        gamma_values = [value for value in (gamma_d, gamma_u) if math.isfinite(value)]
        gf = float(np.mean(gamma_values)) if gamma_values else float("nan")

        records.append(
            {
                "Name": _fetch_any(row, "Name", "Test ID", "TestID", "ID", "Scenario ID", "ScenarioID"),
                "m": _safe_float(_fetch_any(row, "m")),
                "b": _safe_float(_fetch_any(row, "beta", "Β", "b")),
                "h": _safe_float(_fetch_any(row, "h")),
                "Hm0 toe": _safe_float(_fetch_any(row, "Hm0,t", "Hm0_t", "Hm0toe", "Hm0 toe")),
                "Tm-1,0 toe": _safe_float(_fetch_any(row, "Tm-1,0,t", "Tm10t", "Tm-1,0 toe", "Tm-1,0toe")),
                "ht": _safe_float(_fetch_any(row, "ht")),
                "Bt": _safe_float(_fetch_any(row, "Bt")),
                "gf": gf,
                "cotad": _safe_float(_fetch_any(row, "cot(a_d)", "cotad", "cotαd", "cot a d")),
                "cotau": _safe_float(_fetch_any(row, "cot(a_u)", "cotau", "cotαu", "cot a u")),
                "B": _safe_float(_fetch_any(row, "B")),
                "hb": _safe_float(_fetch_any(row, "hb")),
                "Rc": _safe_float(_fetch_any(row, "Rc")),
                "Ac": _safe_float(_fetch_any(row, "Ac")),
                "Gc": _safe_float(_fetch_any(row, "Gc")),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if "Name" not in frame.columns:
        frame["Name"] = [f"scenario_{idx + 1}" for idx in range(len(frame))]
    frame["Name"] = frame["Name"].fillna(pd.Series([f"scenario_{idx + 1}" for idx in range(len(frame))]))
    return frame


def parse_batch_feature_file(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    frame = pd.read_csv(path, sep=delimiter)

    rename_map = {}
    for column in frame.columns:
        norm = _normalize_name(str(column))
        if norm in {"name", "scenario", "scenarioid", "testid", "id"}:
            rename_map[column] = "Name"
        elif norm in {"m"}:
            rename_map[column] = "m"
        elif norm in {"b", "beta"}:
            rename_map[column] = "b"
        elif norm in {"h"}:
            rename_map[column] = "h"
        elif norm in {"hm0toe", "hm0t", "hm0_toe"}:
            rename_map[column] = "Hm0 toe"
        elif norm in {"tm10toe", "tm10t", "tm10_toe"}:
            rename_map[column] = "Tm-1,0 toe"
        elif norm == "ht":
            rename_map[column] = "ht"
        elif norm == "bt":
            rename_map[column] = "Bt"
        elif norm == "gf":
            rename_map[column] = "gf"
        elif norm == "cotad":
            rename_map[column] = "cotad"
        elif norm == "cotau":
            rename_map[column] = "cotau"
        elif norm in {"bermwidth", "bwidth", "bermb"}:
            rename_map[column] = "B"
        elif norm == "hb":
            rename_map[column] = "hb"
        elif norm == "rc":
            rename_map[column] = "Rc"
        elif norm == "ac":
            rename_map[column] = "Ac"
        elif norm == "gc":
            rename_map[column] = "Gc"

    frame = frame.rename(columns=rename_map)
    if "Name" not in frame.columns:
        frame["Name"] = [f"scenario_{idx + 1}" for idx in range(len(frame))]

    for col in DATABASE_FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = frame[col].apply(_safe_float)

    return frame[["Name"] + DATABASE_FEATURES]


def scenario_from_cli(args: argparse.Namespace) -> pd.DataFrame:
    record: Dict[str, Any] = {"Name": args.name or "scenario_1"}
    for cli_key, db_col in CLI_TO_DB.items():
        record[db_col] = _safe_float(getattr(args, cli_key))
    return pd.DataFrame([record])


def build_range_warnings(bundle: ModelBundle, frame: pd.DataFrame) -> List[str]:
    warnings_out: List[str] = []
    for _, row in frame.iterrows():
        out_of_range: List[str] = []
        hm0 = _safe_float(row.get("Hm0 toe"))
        if not math.isfinite(hm0) or hm0 <= 0.0:
            out_of_range.append("Hm0 toe missing or <= 0, so q cannot be reconstructed from sq")
        for col in DATABASE_FEATURES:
            value = _safe_float(row.get(col))
            min_v, max_v = bundle.feature_ranges[col]
            if math.isfinite(value) and math.isfinite(min_v) and math.isfinite(max_v):
                if value < min_v or value > max_v:
                    out_of_range.append(f"{col}={value:g} outside [{min_v:g}, {max_v:g}]")
        warnings_out.append("; ".join(out_of_range))
    return warnings_out


def build_prediction_table(bundle: ModelBundle, scenarios: pd.DataFrame) -> pd.DataFrame:
    frame = scenarios.copy()
    for col in DATABASE_FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = frame[col].apply(_safe_float)

    mean_sq, p05_sq, p50_sq, p95_sq, mean_lpsm, p05_lpsm, p50_lpsm, p95_lpsm = bundle.predict_q_distribution_lpsm(frame)
    q_m3s_per_m = mean_lpsm / 1000.0

    result = frame[["Name"] + DATABASE_FEATURES].copy()
    result["sq_mean"] = mean_sq
    result["sq_p05"] = p05_sq
    result["sq_p50"] = p50_sq
    result["sq_p95"] = p95_sq
    result["q_m3_per_s_per_m"] = q_m3s_per_m
    result["q_l_per_s_per_m"] = mean_lpsm
    result["q_p05_l_per_s_per_m"] = p05_lpsm
    result["q_p50_l_per_s_per_m"] = p50_lpsm
    result["q_p95_l_per_s_per_m"] = p95_lpsm
    result["range_warning"] = build_range_warnings(bundle, frame)
    return result


def print_prediction_table(table: pd.DataFrame) -> None:
    display_cols = [
        "Name",
        "sq_mean",
        "sq_p05",
        "sq_p50",
        "sq_p95",
        "q_l_per_s_per_m",
        "q_p05_l_per_s_per_m",
        "q_p50_l_per_s_per_m",
        "q_p95_l_per_s_per_m",
        "range_warning",
    ]
    compact = table[display_cols].copy()
    for col in [
        "sq_mean",
        "sq_p05",
        "sq_p50",
        "sq_p95",
        "q_l_per_s_per_m",
        "q_p05_l_per_s_per_m",
        "q_p50_l_per_s_per_m",
        "q_p95_l_per_s_per_m",
    ]:
        compact[col] = compact[col].map(lambda x: f"{x:.6g}" if pd.notna(x) else "nan")
    print(compact.to_string(index=False))


def ensure_bundle(args: argparse.Namespace) -> ModelBundle:
    model_path = Path(args.model)
    if model_path.exists():
        return load_model_bundle(model_path)

    if not getattr(args, "database", None):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found and no --database path was supplied for auto-training."
        )

    print(f"Model file '{model_path}' not found. Training a new model from '{args.database}'...")
    artifacts = train_model_bundle(
        database_path=Path(args.database),
        sq_floor=args.sq_floor,
        n_models=args.n_models,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    save_model_bundle(artifacts.bundle, model_path)
    diagnostics_path = Path(args.diagnostics) if getattr(args, "diagnostics", None) else DEFAULT_DIAGNOSTICS_PATH
    write_diagnostics(artifacts.bundle, diagnostics_path, holdout=artifacts.holdout, full_data=artifacts.full_data)
    print(f"Saved model to: {model_path}")
    print(f"Saved diagnostics to: {diagnostics_path}")
    return artifacts.bundle


def command_train(args: argparse.Namespace) -> int:
    artifacts = train_model_bundle(
        database_path=Path(args.database),
        sq_floor=args.sq_floor,
        n_models=args.n_models,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    save_model_bundle(artifacts.bundle, Path(args.model))
    write_diagnostics(artifacts.bundle, Path(args.diagnostics), holdout=artifacts.holdout, full_data=artifacts.full_data)

    print("Training complete.")
    print(f"Model:       {args.model}")
    print(f"Diagnostics: {args.diagnostics}")
    print(json.dumps(artifacts.bundle.metrics, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    bundle = ensure_bundle(args)

    if args.from_inp:
        scenarios = parse_inp_file(Path(args.from_inp))
    elif args.from_csv:
        scenarios = parse_batch_feature_file(Path(args.from_csv))
    else:
        scenarios = scenario_from_cli(args)

    table = build_prediction_table(bundle, scenarios)
    print_prediction_table(table)

    output_path = Path(args.output)
    table.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neural-network predictor for overtopping discharge using adimensional sq = q / sqrt(g * Hm0_toe^3)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the overtopping neural network.")
    train_parser.add_argument("--database", required=True, help="Path to database.csv")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output .joblib model path")
    train_parser.add_argument(
        "--diagnostics",
        default=str(DEFAULT_DIAGNOSTICS_PATH),
        help="Output JSON diagnostics path",
    )
    train_parser.add_argument("--sq-floor", type=float, default=DEFAULT_SQ_FLOOR, help="Lower bound used before log10(sq)")
    train_parser.add_argument("--n-models", type=int, default=DEFAULT_N_MODELS, help="Number of bagged MLP models in the ensemble")
    train_parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum MLP iterations")
    train_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Predict overtopping q and sq.")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .joblib model")
    predict_parser.add_argument("--database", help="Database path used only if auto-training is required")
    predict_parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS_PATH), help="Diagnostics path if auto-training is triggered")
    predict_parser.add_argument("--sq-floor", type=float, default=DEFAULT_SQ_FLOOR, help="Lower bound used before log10(sq) when auto-training")
    predict_parser.add_argument("--n-models", type=int, default=DEFAULT_N_MODELS, help="Number of MLP models if auto-training is triggered")
    predict_parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum MLP iterations if auto-training is triggered")
    predict_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed if auto-training is triggered")
    predict_parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Prediction CSV output path")

    source_group = predict_parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--from-inp", help="Pipe-separated scenario file such as Example_input_file.inp.txt")
    source_group.add_argument("--from-csv", help="CSV/semicolon file with feature columns")

    predict_parser.add_argument("--name", default="scenario_1", help="Scenario name for single-scenario prediction")
    predict_parser.add_argument("--m", type=float, help="Foreshore slope cotangent")
    predict_parser.add_argument("--beta", type=float, help="Wave obliquity in degrees")
    predict_parser.add_argument("--h", type=float, help="Water depth at the toe/front of the structure [m]")
    predict_parser.add_argument("--hm0-toe", dest="hm0_toe", type=float, help="Hm0 at the toe [m]")
    predict_parser.add_argument("--tm-1-0-toe", dest="tm10_toe", type=float, help="Tm-1,0 at the toe [s]")
    predict_parser.add_argument("--ht", type=float, help="Toe water depth [m]")
    predict_parser.add_argument("--bt", type=float, help="Toe width [m]")
    predict_parser.add_argument("--gf", type=float, help="Roughness/permeability factor [-]")
    predict_parser.add_argument("--cotad", type=float, help="Lower slope cotangent [-]")
    predict_parser.add_argument("--cotau", type=float, help="Upper slope cotangent [-]")
    predict_parser.add_argument("--berm-width", dest="berm_width", type=float, help="Berm width B [m]")
    predict_parser.add_argument("--hb", type=float, help="Berm water depth hb [m]")
    predict_parser.add_argument("--rc", type=float, help="Crest freeboard Rc [m]")
    predict_parser.add_argument("--ac", type=float, help="Armour crest freeboard Ac [m]")
    predict_parser.add_argument("--gc", type=float, help="Crest width Gc [m]")
    predict_parser.set_defaults(func=command_predict)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())