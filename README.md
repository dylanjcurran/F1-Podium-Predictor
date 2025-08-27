# F1 Podium Predictor 🏎️

Predicting Formula 1 podium finishes (top-3 results) using historical race data (1950–2023).  
This project explores leakage-safe feature engineering, calibration of probabilities, and model robustness across seasons and circuits.

## 🚀 Project Overview
- **Goal:** Predict whether a driver will finish on the podium (top-3).  
- **Dataset:** Kaggle Formula 1 World Championship (1950–2020), extended to 2023.  
- **Approach:**  
  - Strict season-based splits (train ≤ 2018, validate 2019–2020, test 2021–2023).  
  - Leakage-safe rolling features: driver/team form (last-N points & podium rates).  
  - Models: Logistic Regression and XGBoost, with isotonic calibration for probability quality.  
  - Baselines: Simple grid-position rule (driver starting ≤3rd).

## 📊 Key Results
| Model                  | PR-AUC | Brier Score | F1@opt |
|-------------------------|--------|-------------|--------|
| Baseline (grid ≤3)     | 0.41   | 0.133       | 0.60   |
| Logistic Regression     | 0.63   | 0.108       | 0.60   |
| Logistic Reg (Calibr.)  | 0.61   | **0.085**   | 0.60   |
| XGBoost                 | **0.74** | 0.105     | **0.69** |
| XGBoost (Calibrated)    | 0.70   | **0.069**   | **0.69** |

**Takeaway:**  
- XGBoost nearly doubled PR-AUC vs baseline (0.41 → 0.74).  
- Calibration cut Brier loss in half (0.133 → 0.069), producing well-calibrated probabilities.  
- F1 on unseen test seasons reaches 0.69.  

## 📂 Repo Structure
f1-podium-predictor/  
├─ scripts/  
│ ├─ run_model.py
│ ├─ train_f1_module.py  
│ └─ data_manipulation.py
├─ data/
├─ outputs/  
│ ├─ metrics.json  
│ ├─ MODEL_REPORT.md  
│ ├─ calibration_test_raw.png  
│ ├─ calibration_test_calibrated.png  
│ ├─ xgb_calibration_test_raw.png  
│ ├─ xgb_feature_importance.csv  
│ ├─ xgb_test_by_season.csv  
│ ├─ xgb_test_by_circuit_{raw,calibrated}.csv  
│ ├─ logreg_test_by_circuit_{raw,calibrated}.csv  
│ └─ *_thresholds_test_{raw,calibrated}.csv  
├─ tests/  
│ └─ test_pipeline.py (smoke test) 
├─ requirements.txt  
└─ README.md  

## 📸 Example Outputs
Calibration (raw vs calibrated):  

![Calibration Raw](outputs/calibration_test_raw.png)  
![Calibration Calibrated](outputs/calibration_test_calibrated.png)  

XGBoost Calibration:  

![XGB Calibration](outputs/xgb_calibration_test_raw.png)  

## 🔧 How to Run
1. Clone the repo and install requirements:  
   git clone https://github.com/dylanjcurran/F1-Podium-Predictor.git  
   cd F1-Podium-Predictor  
   pip install -r requirements.txt  

2. Place dataset CSVs in `data/raw/`.  
   - Dataset: Kaggle Formula 1 World Championship (1950–2020).  
   - Extend to 2023 if available (or use provided joined CSV).  

3. Run the pipeline:  
   python scripts/run_model.py  

4. Outputs will be written to `outputs/`:  
   - metrics.json, MODEL_REPORT.md  
   - calibration plots  
   - threshold sweeps  
   - per-season & per-circuit breakdowns  
   - feature importance CSV  

## ✅ Next Steps
- Add SHAP feature explanations for XGBoost.  
- Extend to driver-constructor matching for future prediction tasks.  
- Package into a web demo (Streamlit).  

## 📄 License
This project is licensed under the MIT License — see LICENSE for details.
