import os
import pytest
from scripts.train_f1_module import run_pipeline

@pytest.mark.skipif("DATA_CSV" not in os.environ, reason="Set DATA_CSV=/path/to/your/rows.csv to run")
def test_pipeline_smoke(tmp_path):
    outdir = tmp_path / "out"
    result = run_pipeline(
        csv_path=os.environ["DATA_CSV"],
        out_dir=str(outdir),
        use_xgb=True,
        calibrate=True,
        tune_thresh=True,
    )
    assert (outdir / "metrics.json").exists()
    assert "metrics" in result and isinstance(result["metrics"], dict)
