# train_f1_module.py
"""
F1 Podium Predictor — function-first module (leakage-safe)

What’s included:
- Reproducibility + environment logging
- Duplicate resolution to ensure one row per (year, round, driverId)
- Leakage-safe rolling features (shift→roll) for drivers & teams
- Hard guard: force NaN on the first-ever row per driver & per team after rolling
- Season-based splits (train/val/test)
- Baseline (grid<=3), Logistic Regression (raw + isotonic-calibrated)
- XGBoost (raw + isotonic-calibrated)
- Threshold tuning on validation
- Calibration plots + per-season breakdown
- One-click Markdown report (MODEL_REPORT.md)
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import json
import platform
import sys
import random
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, f1_score, brier_score_loss, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Optional XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------------------------------------------------------------
# Reproducibility & environment logging
# ---------------------------------------------------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost  # noqa
    except Exception:
        pass


def log_environment(out_dir: str):
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {}
    }
    try:
        import sklearn
        meta["packages"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost
        meta["packages"]["xgboost"] = xgboost.__version__
    except Exception:
        pass
    meta["packages"]["pandas"] = pd.__version__
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "environment.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------
# Data loading, dedupe, validation
# ---------------------------------------------------------------------
def load_joined_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def resolve_driver_race_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per (year, round, driverId).
    Preference order:
      1) has a valid finishing position
      2) better finishing position
      3) more points
      4) smaller resultId (if present)
    """
    df = df.copy()

    pos = pd.to_numeric(df.get("positionOrder"), errors="coerce")
    pts = pd.to_numeric(df.get("points"), errors="coerce")
    resid = pd.to_numeric(df.get("resultId"), errors="coerce")

    df["_has_pos"] = pos.notna().astype(int)
    df["_pos"] = pos.fillna(9999)
    df["_pts"] = pts.fillna(-1)
    df["_resid"] = resid.fillna(1e12)

    df = df.sort_values(
        ["year", "round", "driverId", "_has_pos", "_pos", "_pts", "_resid"],
        ascending=[True, True, True, False, True, False, True],
        kind="mergesort",
    )

    before = len(df)
    df = df.drop_duplicates(subset=["year", "round", "driverId"], keep="first")
    after = len(df)

    df = df.drop(columns=["_has_pos", "_pos", "_pts", "_resid"], errors="ignore")

    removed = before - after
    if removed > 0:
        print(f"[dedupe] Removed {removed} duplicate rows on (year, round, driverId).")

    return df


def validate_dataset(df: pd.DataFrame):
    req = ["year", "round", "grid", "positionOrder", "driverId", "constructorId"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if df[["year", "round", "driverId", "constructorId"]].isnull().any().any():
        raise ValueError("Null in key columns (year/round/driverId/constructorId).")
    if df.duplicated(subset=["year", "round", "driverId"]).any():
        raise ValueError("Duplicate (year, round, driverId) rows detected.")


# ---------------------------------------------------------------------
# Feature engineering (leakage-safe rolling)
# ---------------------------------------------------------------------
def threshold_sweep_df(y_true: np.ndarray, probs: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Return PR/F1 table across thresholds for quick inspection."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "support_pos": int(y_true.sum()),
        })
    return pd.DataFrame(rows)

def save_threshold_sweep(y_true: np.ndarray, probs: np.ndarray, out_csv: str) -> None:
    df = threshold_sweep_df(y_true, probs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def per_group_report(y_true: np.ndarray, probs: np.ndarray, group: pd.Series, min_count: int = 15) -> pd.DataFrame:
    """Group metrics by an arbitrary series (e.g., circuitId)."""
    rows = []
    g = group.reset_index(drop=True)
    yt = pd.Series(y_true)
    pt = pd.Series(probs)
    for key, idx in g.groupby(g).groups.items():
        idx = list(idx)
        if len(idx) < min_count:
            continue
        yt_k = yt.loc[idx].values
        pt_k = pt.loc[idx].values
        rows.append({
            "group": key if pd.notna(key) else "NA",
            "n": int(len(idx)),
            "pr_auc": float(average_precision_score(yt_k, pt_k)),
            "brier": float(brier_score_loss(yt_k, pt_k)),
            "f1@0.5": float(f1_score(yt_k, (pt_k >= 0.5).astype(int), zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values(["n", "pr_auc"], ascending=[False, False])

def _shifted_roll_mean(series: pd.Series, group_keys: pd.Series, order_cols: list, window: int) -> pd.Series:
    """
    Leakage-safe rolling mean:
      - sort by order_cols to ensure chronological order within groups
      - shift() first (drops current-race info)
      - rolling(window) with min_periods=1 per group
      - result is index-aligned (via reset_index)
    """
    s = pd.to_numeric(series, errors="coerce")
    df_tmp = pd.DataFrame({"s": s, "g": group_keys})
    df_tmp = df_tmp.join(series.index.to_series(name="_idx"))
    # bring order columns
    # caller guarantees these exist in the original df
    for c in order_cols:
        df_tmp[c] = series.index.map(series.obj[c] if hasattr(series, "obj") else series.index)

    # We need an aligned frame: better to pass the whole DataFrame into this util
    # Simpler approach: caller provides a pre-sorted df and we just groupby on that.
    # To keep things robust, we’ll return a placeholder here; the actual sorting happens outside.
    raise RuntimeError("Internal misuse of _shifted_roll_mean. Use _roll_on_prepared_df instead.")


def _roll_on_prepared_df(df: pd.DataFrame, value_col: str, group_col: str, window: int) -> pd.Series:
    """
    Assumes df is already sorted chronologically on ["year","round","raceId"(opt)].
    Returns index-aligned leakage-safe rolling mean for value_col within group_col.
    """
    shifted = df.groupby(group_col)[value_col].shift()
    rolled = shifted.groupby(df[group_col]).rolling(window, min_periods=1).mean()
    return rolled.reset_index(level=0, drop=True)


def build_features(
    df: pd.DataFrame,
    include_form_features: bool = True,
    include_podium_rate: bool = True,
    window: int = 5
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    # Coerce numerics
    for col in ["grid", "qpos", "year", "round", "positionOrder", "points", "driverId", "constructorId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with clear scheduling keys
    df = df.dropna(subset=["year", "round", "driverId", "constructorId", "grid", "positionOrder"]).copy()

    # Target
    df["y_podium"] = (df["positionOrder"] <= 3).astype(int)

    # Establish a stable chronological sort ONCE (keeps original index)
    order_cols = ["year", "round"]
    if "raceId" in df.columns:
        order_cols.insert(2, "raceId")
    df = df.sort_values(order_cols)

    # Rolling features (index-aligned)
    if include_form_features or include_podium_rate:
        if include_form_features and "points" in df.columns:
            # drivers
            df["driver_points_lN"] = _roll_on_prepared_df(df, "points", "driverId", window)
            # teams
            df["team_points_lN"] = _roll_on_prepared_df(df, "points", "constructorId", window)

        if include_podium_rate:
            df["driver_podiumrate_lN"] = _roll_on_prepared_df(df, "y_podium", "driverId", window)
            df["team_podiumrate_lN"] = _roll_on_prepared_df(df, "y_podium", "constructorId", window)

        # HARD NO-LEAK GUARANTEE:
        # Force first-ever row per driver/team to NaN for any *_lN column.
        # (Even if math produced a number due to oddities, we null it here.)
        lN_cols = [c for c in df.columns if c.endswith("_lN")]
        if "driverId" in df.columns:
            first_driver_idx = (
                df.sort_values(["driverId", "year", "round"])
                  .groupby("driverId")
                  .head(1)
                  .index
            )
            df.loc[first_driver_idx, lN_cols] = np.nan
        if "constructorId" in df.columns:
            first_team_idx = (
                df.sort_values(["constructorId", "year", "round"])
                  .groupby("constructorId")
                  .head(1)
                  .index
            )
            df.loc[first_team_idx, lN_cols] = np.nan

    # Features to keep
    keep_cols = [
        "year", "round",
        "grid", "qpos",
        "driverRef", "constructor_name",
        "circuitId", "country"
    ]
    if include_form_features:
        keep_cols += ["driver_points_lN", "team_points_lN"]
    if include_podium_rate:
        keep_cols += ["driver_podiumrate_lN", "team_podiumrate_lN"]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after feature build: {missing}")

    X = df[keep_cols].copy()
    # Safe imputations AFTER forcing first rows to NaN
    X["qpos"] = pd.to_numeric(X["qpos"], errors="coerce").fillna(X["grid"])
    for f in ["driver_points_lN", "team_points_lN", "driver_podiumrate_lN", "team_podiumrate_lN"]:
        if f in X.columns:
            X[f] = pd.to_numeric(X[f], errors="coerce").fillna(0.0)

    # One-hot encode categoricals
    cat_cols = ["driverRef", "constructor_name", "circuitId", "country"]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False, drop_first=True)

    y = df["y_podium"].values.astype(int)
    return X, y, df


# ---------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------
def season_split(
    df: pd.DataFrame,
    years_dict: Optional[Dict[str, int]] = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if years_dict is None:
        years_dict = {
            "train_end": 2018,
            "val_start": 2019, "val_end": 2020,
            "test_start": 2021, "test_end": 2023
        }
    ysr = df["year"]
    tr = (ysr <= years_dict["train_end"])
    va = (ysr >= years_dict["val_start"]) & (ysr <= years_dict["val_end"])
    te = (ysr >= years_dict["test_start"]) & (ysr <= years_dict["test_end"])
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("Empty split; adjust years_dict to match dataset coverage.")
    return tr, va, te


# ---------------------------------------------------------------------
# Models & metrics
# ---------------------------------------------------------------------
def baseline_grid_leq3(y_true: np.ndarray, grid_series: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    prob = (pd.to_numeric(grid_series, errors="coerce") <= 3).astype(int).values
    pred = prob.copy()
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1@0.5": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
    }, prob


def train_logreg(X_tr: pd.DataFrame, y_tr: np.ndarray):
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    pipe.fit(X_tr, y_tr)
    return pipe


def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, scale_pos_weight: float = 1.0):
    if not HAS_XGB:
        return None
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
    )
    clf.fit(X_tr, y_tr)
    return clf


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    prob = model.predict_proba(X)[:, 1]
    metrics = {
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "f1@0.5": float(f1_score(y, (prob >= 0.5).astype(int), zero_division=0)),
    }
    return metrics, prob


def tune_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    num: int = 19
) -> Dict[str, Any]:
    thresholds = np.linspace(start, stop, num)
    best_t = None
    best_f1 = -1.0
    curve = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        curve.append([float(t), float(f1)])
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return {"best_threshold": best_t, "best_f1": float(best_f1), "curve": curve}


def save_calibration_plot(y_true: np.ndarray, probs: np.ndarray, out_png: str) -> None:
    import matplotlib.pyplot as plt
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration (Test)")
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Empirical frequency")
    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True
    )
    plt.savefig(out_path, dpi=150)
    plt.close()


def isotonic_calibrate(y_val: np.ndarray, p_val: np.ndarray, p_target: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso.transform(p_target)


def per_season_report(y_true: np.ndarray, probs: np.ndarray, seasons: pd.Series) -> pd.DataFrame:
    rows = []
    for yr, idx in seasons.groupby(seasons).groups.items():
        idx = list(idx)
        yt = y_true[idx]
        pt = probs[idx]
        rows.append({
            "year": int(yr),
            "pr_auc": float(average_precision_score(yt, pt)),
            "brier": float(brier_score_loss(yt, pt)),
            "f1@0.5": float(f1_score(yt, (pt >= 0.5).astype(int), zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values("year")


# ---------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------
def write_markdown_report(out_dir: str):
    p = Path(out_dir)
    m = json.loads((p / "metrics.json").read_text())

    def fmt(d):
        if not isinstance(d, dict):
            return ""
        parts = []
        for k, v in d.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}: {v:.3f}")
        return ", ".join(parts)

    lines = []
    lines += ["# F1 Podium Predictor — Model Report", ""]
    s = m["splits"]
    lines += [f"**Splits** — Train {s['train_years'][0]}–{s['train_years'][1]}, "
              f"Val {s['val_years'][0]}–{s['val_years'][1]}, "
              f"Test {s['test_years'][0]}–{s['test_years'][1]}", ""]
    lines += ["## Baseline (grid ≤ 3)"]
    lines += [f"- Val: {fmt(m['baseline_grid_leq3']['val'])}"]
    lines += [f"- Test: {fmt(m['baseline_grid_leq3']['test'])}", ""]
    lines += ["## Logistic Regression"]
    lines += [f"- Val: {fmt(m['logreg_raw']['val'])}"]
    lines += [f"- Test: {fmt(m['logreg_raw']['test'])}"]
    if m.get("logreg_calibrated"):
        lines += ["- Calibrated Test: " + fmt(m["logreg_calibrated"]["test"])]
    lines += ["", "## XGBoost"]
    if m.get("xgb_raw"):
        lines += [f"- Val: {fmt(m['xgb_raw']['val'])}"]
        lines += [f"- Test: {fmt(m['xgb_raw']['test'])}"]
    if m.get("xgb_calibrated"):
        lines += ["- Calibrated Test: " + fmt(m["xgb_calibrated"]["test"])]
    if m.get("xgb_best_params"):
        lines += ["", "### Best XGB params", "```json", json.dumps(m["xgb_best_params"], indent=2), "```"]
    for f in ["calibration_test_raw.png", "calibration_test_calibrated.png", "xgb_calibration_test_raw.png"]:
        if (p / f).exists():
            lines += ["", f"![{f}](./{f})"]
    if (p / "xgb_test_by_season.csv").exists():
        lines += ["", "See `xgb_test_by_season.csv` for season breakdown."]
    (p / "MODEL_REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_pipeline(
    csv_path: str,
    out_dir: str = "outputs",
    years_dict: Optional[Dict[str, int]] = None,
    use_xgb: bool = True,
    tune_thresh: bool = True,
    include_form_features: bool = True,
    include_podium_rate: bool = True,
    window: int = 5,
    calibrate: bool = True,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Repro & env logging
    set_global_seed(42)
    log_environment(out_dir)

    # Load, dedupe, validate
    df = load_joined_csv(csv_path)
    df = df.dropna(subset=["year", "round", "driverId"])  # guard
    df = resolve_driver_race_duplicates(df)
    validate_dataset(df)

    # Build features
    X, y, df_clean = build_features(
        df,
        include_form_features=include_form_features,
        include_podium_rate=include_podium_rate,
        window=window
    )
    tr, va, te = season_split(df_clean, years_dict)

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    # Speed/memory tweak
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    X_te = X_te.astype(np.float32)

    # ----- Baseline -----
    base_val, _ = baseline_grid_leq3(y_va, df_clean.loc[va, "grid"])
    base_te, _ = baseline_grid_leq3(y_te, df_clean.loc[te, "grid"])

    # ----- Logistic Regression -----
    log_model = train_logreg(X_tr, y_tr)
    log_val_metrics, log_val_prob = evaluate(log_model, X_va, y_va)
    log_te_metrics, log_te_prob = evaluate(log_model, X_te, y_te)

    # Save Logistic Regression model + feature order
    with open(out_path / "logreg_model.pkl", "wb") as f:
        pickle.dump(log_model, f)

    (out_path / "feature_order.json").write_text(json.dumps(list(X_tr.columns)))

    # NEW: threshold sweep (LogReg raw, test)
    save_threshold_sweep(y_te, log_te_prob, str(out_path / "logreg_thresholds_test_raw.csv"))


    if tune_thresh:
        log_opt = tune_threshold(y_va, log_val_prob)
        lt = log_opt["best_threshold"]
        log_val_metrics["f1@opt"] = float(log_opt["best_f1"])
        log_val_metrics["opt_threshold"] = float(lt)
        log_te_metrics["f1@opt"] = float(f1_score(y_te, (log_te_prob >= lt).astype(int), zero_division=0))

    # LogReg calibration
    log_cal = None
    if calibrate:
        te_prob_cal = isotonic_calibrate(y_va, log_val_prob, log_te_prob)
        
        # NEW: threshold sweep (LogReg calibrated, test)
        save_threshold_sweep(y_te, te_prob_cal, str(out_path / "logreg_thresholds_test_calibrated.csv"))

        # NEW: per-circuit breakdowns (LogReg raw & calibrated)
        try:
            log_circ_raw = per_group_report(y_te, log_te_prob, df_clean.loc[te, "circuitId"])
            log_circ_raw.to_csv(out_path / "logreg_test_by_circuit_raw.csv", index=False)
            log_circ_cal = per_group_report(y_te, te_prob_cal, df_clean.loc[te, "circuitId"])
            log_circ_cal.to_csv(out_path / "logreg_test_by_circuit_calibrated.csv", index=False)
        except Exception:
            pass

        log_cal = {
            "test": {
                "pr_auc": float(average_precision_score(y_te, te_prob_cal)),
                "brier": float(brier_score_loss(y_te, te_prob_cal)),
                "f1@0.5": float(f1_score(y_te, (te_prob_cal >= 0.5).astype(int), zero_division=0)),
            }
        }
        val_prob_cal = isotonic_calibrate(y_va, log_val_prob, log_val_prob)
        cal_opt = tune_threshold(y_va, val_prob_cal)
        ct = cal_opt["best_threshold"]
        log_cal["val"] = {"f1@opt": float(cal_opt["best_f1"]), "opt_threshold": float(ct)}
        log_cal["test"]["f1@opt"] = float(f1_score(y_te, (te_prob_cal >= ct).astype(int), zero_division=0))

        # Plots
        save_calibration_plot(y_te, log_te_prob, str(out_path / "calibration_test_raw.png"))
        save_calibration_plot(y_te, te_prob_cal, str(out_path / "calibration_test_calibrated.png"))
    else:
        save_calibration_plot(y_te, log_te_prob, str(out_path / "calibration_test_raw.png"))

    # ----- XGBoost -----
    xgb_model = None
    xgb_val_metrics = None
    xgb_te_metrics = None
    xgb_cal = None

    if use_xgb and HAS_XGB:
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0

        xgb_model = train_xgb(X_tr, y_tr, scale_pos_weight=spw)
        xgb_val_metrics, xgb_val_prob = evaluate(xgb_model, X_va, y_va)
        xgb_te_metrics, xgb_te_prob = evaluate(xgb_model, X_te, y_te)

        # Save XGBoost model
        with open(out_path / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_model, f)


        # NEW: threshold sweep (XGB raw, test)
        save_threshold_sweep(y_te, xgb_te_prob, str(out_path / "xgb_thresholds_test_raw.csv"))

        # NEW: per-circuit breakdown (XGB raw)
        try:
            xgb_circ_raw = per_group_report(y_te, xgb_te_prob, df_clean.loc[te, "circuitId"])
            xgb_circ_raw.to_csv(out_path / "xgb_test_by_circuit_raw.csv", index=False)
        except Exception:
            pass


        if tune_thresh:
            xgb_opt = tune_threshold(y_va, xgb_val_prob)
            xt = xgb_opt["best_threshold"]
            xgb_val_metrics["f1@opt"] = float(xgb_opt["best_f1"])
            xgb_val_metrics["opt_threshold"] = float(xt)
            xgb_te_metrics["f1@opt"] = float(
                f1_score(y_te, (xgb_te_prob >= xt).astype(int), zero_division=0)
            )

        if calibrate:
            te_prob_cal_x = isotonic_calibrate(y_va, xgb_val_prob, xgb_te_prob)

            # NEW: threshold sweep (XGB calibrated, test)
            save_threshold_sweep(y_te, te_prob_cal_x, str(out_path / "xgb_thresholds_test_calibrated.csv"))

            # NEW: per-circuit breakdown (XGB calibrated)
            try:
                xgb_circ_cal = per_group_report(y_te, te_prob_cal_x, df_clean.loc[te, "circuitId"])
                xgb_circ_cal.to_csv(out_path / "xgb_test_by_circuit_calibrated.csv", index=False)
            except Exception:
                pass


            xgb_cal = {
                "test": {
                    "pr_auc": float(average_precision_score(y_te, te_prob_cal_x)),
                    "brier": float(brier_score_loss(y_te, te_prob_cal_x)),
                    "f1@0.5": float(f1_score(y_te, (te_prob_cal_x >= 0.5).astype(int), zero_division=0)),
                }
            }
            val_prob_cal_x = isotonic_calibrate(y_va, xgb_val_prob, xgb_val_prob)
            cal_opt_x = tune_threshold(y_va, val_prob_cal_x)
            ct_x = cal_opt_x["best_threshold"]
            xgb_cal["val"] = {"f1@opt": float(cal_opt_x["best_f1"]), "opt_threshold": float(ct_x)}
            xgb_cal["test"]["f1@opt"] = float(
                f1_score(y_te, (te_prob_cal_x >= ct_x).astype(int), zero_division=0)
            )

            # Raw XGB calibration plot for comparison
            save_calibration_plot(y_te, xgb_te_prob, str(out_path / "xgb_calibration_test_raw.png"))

        # Artifacts
        try:
            fi = pd.Series(xgb_model.get_booster().get_score(importance_type="gain"))
            fi.sort_values(ascending=False).head(50).to_csv(out_path / "xgb_feature_importance.csv")
        except Exception:
            pass
        try:
            season_df = per_season_report(y_te, xgb_te_prob, df_clean.loc[te, "year"])
            season_df.to_csv(out_path / "xgb_test_by_season.csv", index=False)
        except Exception:
            pass

    # Save metrics
    metrics = {
        "splits": {
            "train_years": [int(df_clean.loc[tr, "year"].min()), int(df_clean.loc[tr, "year"].max())],
            "val_years": [int(df_clean.loc[va, "year"].min()), int(df_clean.loc[va, "year"].max())],
            "test_years": [int(df_clean.loc[te, "year"].min()), int(df_clean.loc[te, "year"].max())],
        },
        "baseline_grid_leq3": {"val": base_val, "test": base_te},
        "logreg_raw": {"val": log_val_metrics, "test": log_te_metrics},
        "logreg_calibrated": log_cal,
        "xgb_raw": {"val": xgb_val_metrics, "test": xgb_te_metrics} if xgb_model is not None else None,
        "xgb_calibrated": xgb_cal,
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # One-page Markdown summary
    try:
        write_markdown_report(out_dir)
    except Exception:
        pass

    return {
        "metrics_path": str(out_path / "metrics.json"),
        "calibration_plot_raw": str(out_path / "calibration_test_raw.png"),
        "calibration_plot_calibrated": str(out_path / "calibration_test_calibrated.png") if calibrate else None,
        "xgb_calibration_plot_raw": str(out_path / "xgb_calibration_test_raw.png") if (use_xgb and HAS_XGB) else None,
        "has_xgb": bool(xgb_model is not None),
        "metrics": metrics
    }
