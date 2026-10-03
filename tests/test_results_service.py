import pytest
from pathlib import Path
from src.services.results import ResultsService
import json

def test_results_service_load_summary(tmp_path):
    # Mock a summary file
    d = tmp_path / "results"
    d.mkdir()
    f = d / "summary.jsonl"
    f.write_text('{"exp_name": "test_exp", "global_metrics": {"pr_auc": 0.5}}\n')
    
    svc = ResultsService(base_dir=str(d))
    summary = svc.load_summary()
    
    assert len(summary) == 1
    assert summary[0]["exp_name"] == "test_exp"
    assert summary[0]["global_metrics"]["pr_auc"] == 0.5

def test_results_service_malformed_record(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    f = d / "summary.jsonl"
    f.write_text('{"exp_name": "test_exp", "global_metrics": {"pr_auc": 0.5}}\nBAD JSON\n')
    
    svc = ResultsService(base_dir=str(d))
    summary = svc.load_summary()
    
    assert len(summary) == 1
    assert summary[0]["exp_name"] == "test_exp"

def test_results_service_missing_experiment(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    f = d / "summary.jsonl"
    f.write_text('{"exp_name": "test_exp", "global_metrics": {"pr_auc": 0.5}}\n')
    
    svc = ResultsService(base_dir=str(d))
    with pytest.raises(ValueError, match="not found in summary"):
        svc.get_experiment("missing_exp")

def test_results_service_numeric_metric_extraction(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    f = d / "summary.jsonl"
    f.write_text('{"exp_name": "test_exp", "global_metrics": {"pr_auc": 0.5, "f1": NaN, "missing": null}}\n')
    
    svc = ResultsService(base_dir=str(d))
    metrics = svc.get_experiment_metrics("test_exp")
    
    assert metrics["pr_auc"] == 0.5
    assert metrics["f1"] is None
    assert metrics["missing"] is None
