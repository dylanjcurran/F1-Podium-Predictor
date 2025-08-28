import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from train_f1_module import build_features, season_split

def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def align_features(X_row: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """Ensure the single-row DataFrame matches training column order."""
    
    for col in feature_order:
        if col not in X_row.columns:
            X_row[col] = 0.0
    
    extras = [c for c in X_row.columns if c not in feature_order]
    if extras:
        X_row = X_row.drop(columns=extras)
    
    return X_row[feature_order]

def pick_random_test_row(df_clean: pd.DataFrame, test_mask: pd.Series, seed: int | None = None) -> int:
    """Return the index (position) of a random row from the test set."""
    rng = random.Random(seed)
    test_indices = np.flatnonzero(test_mask.values)
    if len(test_indices) == 0:
        raise ValueError("Test split is empty. Check your years_dict or dataset coverage.")
    return int(rng.choice(test_indices))

if __name__ == "__main__":

    # --- Paths ---
    
    # Adjust if your repo layout differs
    data_csv = "../data/processed/final_dataset.csv"
    out_dir  = Path("../outputs")

    # --- Load artifacts ---
    feature_order = json.loads((out_dir / "feature_order.json").read_text())
    
    # Choose which model to demo:
    
    # model_path = out_dir / "logreg_model.pkl"
    model_path = out_dir / "xgb_model.pkl"
    
    model = load_model(model_path)

    # --- Load raw data & build features exactly like training ---
    df_raw = pd.read_csv(data_csv)
    
    X_all, y_all, df_clean = build_features(
        df_raw,
        include_form_features=True,
        include_podium_rate=True,
        window=5,
    )

    # --- Season split (same default years) and pick a random TEST row ---
    tr, va, te = season_split(df_clean, years_dict=None)
    idx = pick_random_test_row(df_clean, te)  # random each run

    # --- Pull a single feature row and align to training order ---
    X_row = X_all.iloc[[idx]].copy()
    X_row = align_features(X_row, feature_order)

    # --- Predict probability ---
    proba = float(model.predict_proba(X_row.astype(np.float32))[0, 1])

    # --- Human-friendly context from df_clean ---
    row = df_clean.iloc[idx]
    driver = str(row.get("driverRef", "NA"))
    team = str(row.get("constructor_name", "NA"))
    gp_name = str(row.get("name", row.get("circuit_name", "Grand Prix")))
    country = str(row.get("country", ""))
    year = int(row.get("year", -1)) if not pd.isna(row.get("year")) else -1
    rnd = int(row.get("round", -1)) if not pd.isna(row.get("round")) else -1
    grid = int(row.get("grid", -1)) if not pd.isna(row.get("grid")) else -1
    qpos = row.get("qpos", None)
    qpos_str = "NA" if pd.isna(qpos) else str(int(qpos))

    # --- Target (for reference) ---
    actual_podium = int(row.get("y_podium", 0))

    # --- Print summary ---
    print("\n=== F1 Podium Predictor — Random Test Row ===")
    print(f"Model: {model_path.name}")
    print(f"Event: {gp_name} ({country}) — Year {year}, Round {rnd}")
    print(f"Driver: {driver} | Team: {team}")
    print(f"Grid: {grid} | Q pos: {qpos_str}")
    print(f"Predicted P(podium): {proba:.3f}")
    print(f"Actual podium (for reference): {actual_podium}")
    print("=============================================\n")
