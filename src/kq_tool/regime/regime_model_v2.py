"""Regime model v2 with separated labels and model features.

The label path lives in ``regime_labels.py`` and is based on GDP x CPI macro
quadrants. This model path deliberately uses only fast market/price indicators
plus already confirmed previous regime memory. That removes the circular
structure where the same variables define the label and then predict it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime_labels import REGIMES, REGIME_TO_INT, transition_matrix

__all__ = [
    "FEATURES",
    "MEMORY_FEATURES",
    "FORBIDDEN_FEATURE_TERMS",
    "ModelConfig",
    "build_features",
    "calibration_table",
    "log_loss_score",
    "naive_nowcast_baseline",
    "p3_forecast_baseline",
    "walk_forward_predict",
]

FORBIDDEN_FEATURE_TERMS = ("gdp", "cpi")

FEATURES = [
    "spread_level",
    "spread_chg_3m",
    "usd_chg_3m",
    "credit_spread_level",
    "credit_spread_chg_3m",
    "kospi_ret_3m",
    "kospi_vol_6m",
    "base_rate_chg_6m",
]

MEMORY_FEATURES = [f"prev_{regime}" for regime in REGIMES]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for purged walk-forward regime probabilities."""

    task: str = "nowcast"
    horizon_m: int = 3
    label_delay_m: int = 2
    min_train_months: int = 60
    seed: int = 42
    prob_clip: float = 1e-3


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build real-time market-derived features from an observable macro panel."""

    required = ["yield_spread_10y_3y", "usdkrw", "credit_spread", "kospi", "base_rate"]
    missing = [col for col in required if col not in panel]
    if missing:
        raise KeyError(f"Missing regime model feature inputs: {missing}")

    frame = pd.DataFrame(index=panel.index)
    frame["spread_level"] = pd.to_numeric(panel["yield_spread_10y_3y"], errors="coerce")
    frame["spread_chg_3m"] = frame["spread_level"].diff(3)
    frame["usd_chg_3m"] = pd.to_numeric(panel["usdkrw"], errors="coerce").pct_change(3)
    frame["credit_spread_level"] = pd.to_numeric(panel["credit_spread"], errors="coerce")
    frame["credit_spread_chg_3m"] = frame["credit_spread_level"].diff(3)

    kospi = pd.to_numeric(panel["kospi"], errors="coerce")
    kospi_ret = kospi.pct_change()
    frame["kospi_ret_3m"] = kospi.pct_change(3)
    frame["kospi_vol_6m"] = kospi_ret.rolling(6).std()
    frame["base_rate_chg_6m"] = pd.to_numeric(panel["base_rate"], errors="coerce").diff(6)
    return frame[FEATURES]


def _make_classifier(seed: int):
    """Prefer LightGBM and fall back to sklearn with the same predict_proba API."""

    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(REGIMES),
            n_estimators=80,
            num_leaves=7,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=7,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=seed,
        )


def _add_memory(features: pd.DataFrame, labels: pd.Series, cfg: ModelConfig) -> pd.DataFrame:
    out = features.copy()
    previous = labels.shift(cfg.label_delay_m)
    for regime in REGIMES:
        out[f"prev_{regime}"] = (previous == regime).astype(float)
    out.loc[previous.isna(), MEMORY_FEATURES] = np.nan
    return out


def _make_xy(features: pd.DataFrame, labels: pd.Series, cfg: ModelConfig) -> tuple[pd.DataFrame, list[str]]:
    if cfg.task == "nowcast":
        y = labels.copy()
        settle = pd.Series(labels.index + pd.offsets.MonthEnd(cfg.label_delay_m), index=labels.index)
    elif cfg.task == "forecast":
        y = labels.shift(-cfg.horizon_m)
        settle = pd.Series(
            labels.index + pd.offsets.MonthEnd(cfg.horizon_m + cfg.label_delay_m),
            index=labels.index,
        )
    else:
        raise ValueError(f"unknown task: {cfg.task}")

    frame = _add_memory(features, labels, cfg)
    frame["_y"] = y.map(REGIME_TO_INT)
    frame["_settle"] = settle
    x_cols = FEATURES + MEMORY_FEATURES
    frame = frame.dropna(subset=x_cols + ["_y"])
    frame["_y"] = frame["_y"].astype(int)
    return frame, x_cols


def _predict_full(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    full = np.zeros((probs.shape[0], len(REGIMES)))
    for cls_i, cls in enumerate(model.classes_):
        full[:, int(cls)] = probs[:, cls_i]
    row_sums = full.sum(axis=1, keepdims=True)
    full = np.divide(full, row_sums, out=np.full_like(full, 1.0 / len(REGIMES)), where=row_sums > 0)
    return full


def _prior_rows(rows: pd.DataFrame, transition_power: np.ndarray) -> np.ndarray:
    onehot = rows[MEMORY_FEATURES].to_numpy(dtype=float)
    out = np.full((len(rows), len(REGIMES)), 1.0 / len(REGIMES))
    known = onehot.sum(axis=1) > 0
    out[known] = transition_power[onehot[known].argmax(axis=1)]
    return out


def _log_loss_np(probs: np.ndarray, y: np.ndarray, clip: float = 1e-3) -> float:
    p = np.clip(probs, clip, 1 - clip)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _uniform_probs(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(REGIMES), index=index, columns=REGIMES)


def walk_forward_predict(
    features: pd.DataFrame,
    labels: pd.Series,
    cfg: ModelConfig = ModelConfig(),
    retrain_every_m: int = 3,
    blend_with_prior: bool = True,
    val_window_m: int = 24,
    w_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return purged walk-forward OOS probabilities and an audit table."""

    xy, x_cols = _make_xy(features, labels, cfg)
    if xy.empty:
        return pd.DataFrame(columns=REGIMES), pd.DataFrame()

    idx = xy.index
    steps = cfg.label_delay_m if cfg.task == "nowcast" else cfg.horizon_m + cfg.label_delay_m
    probs = pd.DataFrame(np.nan, index=idx, columns=REGIMES)
    audit_rows: list[dict] = []

    model = None
    transition_power = np.full((len(REGIMES), len(REGIMES)), 1.0 / len(REGIMES))
    weight = 0.5
    last_train_i = -10**9
    train_max_settle = pd.NaT
    train_n = 0

    for i, timestamp in enumerate(idx):
        train = xy.loc[xy["_settle"] <= timestamp]
        if len(train) < cfg.min_train_months:
            continue

        hist = pd.Series([REGIMES[value] for value in train["_y"]], index=train.index)
        probs_m, _ = transition_matrix(hist, alpha=1.0)
        transition_power = np.linalg.matrix_power(probs_m.to_numpy(dtype=float), steps)

        can_fit_model = train["_y"].nunique() >= 2
        if not can_fit_model:
            row = xy.loc[[timestamp]]
            probs.loc[timestamp] = _prior_rows(row, transition_power)[0]
            audit_rows.append(
                {
                    "T": timestamp,
                    "train_n": len(train),
                    "train_max_settle": train["_settle"].max(),
                    "w": 1.0,
                    "model_fit": False,
                }
            )
            continue

        if model is None or (i - last_train_i) >= retrain_every_m:
            if blend_with_prior and len(train) > cfg.min_train_months + val_window_m:
                train_part = train.iloc[:-val_window_m]
                val_part = train.iloc[-val_window_m:]
                if train_part["_y"].nunique() >= 2:
                    val_model = _make_classifier(cfg.seed)
                    val_model.fit(train_part[x_cols], train_part["_y"].to_numpy(dtype=int))
                    model_probs = _predict_full(val_model, val_part[x_cols])
                    prior_probs = _prior_rows(val_part, transition_power)
                    y_val = val_part["_y"].to_numpy(dtype=int)
                    weight = min(
                        w_grid,
                        key=lambda w: _log_loss_np((1 - w) * model_probs + w * prior_probs, y_val),
                    )
            elif not blend_with_prior:
                weight = 0.0

            model = _make_classifier(cfg.seed)
            model.fit(train[x_cols], train["_y"].to_numpy(dtype=int))
            last_train_i = i
            train_max_settle = train["_settle"].max()
            train_n = len(train)

        row = xy.loc[[timestamp]]
        model_probs = _predict_full(model, row[x_cols])
        prior_probs = _prior_rows(row, transition_power)
        probs.loc[timestamp] = ((1 - weight) * model_probs + weight * prior_probs)[0]
        audit_rows.append(
            {
                "T": timestamp,
                "train_n": train_n,
                "train_max_settle": train_max_settle,
                "w": weight,
                "model_fit": True,
            }
        )

    audit = pd.DataFrame(audit_rows).set_index("T") if audit_rows else pd.DataFrame()
    return probs.dropna(how="all"), audit


def naive_nowcast_baseline(labels: pd.Series, cfg: ModelConfig = ModelConfig()) -> pd.DataFrame:
    """Baseline: the most recently confirmed regime persists."""

    eps = 0.10
    last_confirmed = labels.shift(cfg.label_delay_m)
    probs = pd.DataFrame(eps / (len(REGIMES) - 1), index=labels.index, columns=REGIMES)
    for timestamp, label in last_confirmed.items():
        if pd.isna(label):
            probs.loc[timestamp] = np.nan
        else:
            probs.loc[timestamp] = eps / (len(REGIMES) - 1)
            probs.loc[timestamp, label] = 1 - eps
    return probs.dropna()


def p3_forecast_baseline(labels: pd.Series, cfg: ModelConfig = ModelConfig(), alpha: float = 1.0) -> pd.DataFrame:
    """Expanding transition-matrix forecast baseline."""

    out = pd.DataFrame(np.nan, index=labels.index, columns=REGIMES)
    confirmed = labels.shift(cfg.label_delay_m)

    for i, timestamp in enumerate(labels.index):
        hist = confirmed.iloc[: i + 1].dropna()
        if len(hist) < 36 or pd.isna(confirmed.loc[timestamp]):
            continue
        probs_m, _ = transition_matrix(hist, alpha=alpha)
        ph = np.linalg.matrix_power(probs_m.to_numpy(dtype=float), cfg.horizon_m)
        current = str(confirmed.loc[timestamp])
        out.loc[timestamp] = ph[REGIMES.index(current)]
    return out.dropna()


def log_loss_score(probs: pd.DataFrame, actual: pd.Series, clip: float = 1e-3) -> float:
    """Multiclass log-loss over the common non-null date range."""

    common = probs.index.intersection(actual.dropna().index)
    if len(common) == 0:
        return float("nan")
    p = probs.loc[common].astype(float).clip(clip, 1 - clip)
    p = p.div(p.sum(axis=1), axis=0)
    y = actual.loc[common]
    losses = [-np.log(float(p.loc[timestamp, y.loc[timestamp]])) for timestamp in common]
    return float(np.mean(losses))


def calibration_table(probs: pd.DataFrame, actual: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Return hit-rate by predicted-confidence bucket."""

    common = probs.index.intersection(actual.dropna().index)
    if len(common) == 0:
        return pd.DataFrame(columns=["n", "avg_conf", "hit_rate"])

    p = probs.loc[common].astype(float)
    predicted = p.idxmax(axis=1)
    confidence = p.max(axis=1)
    hit = (predicted == actual.loc[common]).astype(float)
    bins = pd.cut(confidence, np.linspace(0.25, 1.0, n_bins + 1), include_lowest=True)
    return (
        pd.DataFrame({"conf_bin": bins, "hit": hit, "conf": confidence})
        .groupby("conf_bin", observed=True)
        .agg(n=("hit", "size"), avg_conf=("conf", "mean"), hit_rate=("hit", "mean"))
        .round(3)
    )

