from train_f1_module import run_pipeline

out = run_pipeline(
    csv_path="../data/derived_data/final_dataset.csv",
    out_dir="../outputs",
    years_dict={"train_end": 2018, "val_start": 2019, "val_end": 2020, "test_start": 2021, "test_end": 2023},
    use_xgb=True,
    tune_thresh=True,
    include_form_features=True,
    include_podium_rate=True,
    window=5,
    calibrate=True
)
print("Wrote:", out["metrics_path"])
