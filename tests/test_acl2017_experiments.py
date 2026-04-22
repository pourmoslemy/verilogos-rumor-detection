"""Tests for running experiment models on ACL2017 rumor detection data.

The ACL2017 dataset is read from:
historical_data/rumor_detection_acl2017/{twitter15,twitter16}

Workflow:
1) Parse label/source_tweets/tree files.
2) Build a temporary FakeNewsNet-compatible CSV (text,label).
3) Run a minimal experiment suite using the existing experiment manager.
"""

import csv
import importlib.util
import tempfile
from collections import Counter
from pathlib import Path

import pytest
import yaml

from verilogos.experiments.manager.experiment_manager import ExperimentManager


ACL2017_ROOT = Path("historical_data/rumor_detection_acl2017")

LABEL_MAP_BINARY = {
    "true": 0,
    "false": 1,
    "unverified": None,
    "non-rumor": 1,
}


def _sklearn_available() -> bool:
    """Return True when scikit-learn is importable."""
    return importlib.util.find_spec("sklearn") is not None


def _xgboost_available() -> bool:
    """Return True when XGBoost is importable."""
    return importlib.util.find_spec("xgboost") is not None


def _parse_acl2017_pairs(root: Path, version: str, include_uncertain: bool = False):
    """Parse ACL2017 labels + source tweets for one version."""
    version_dir = root / version
    label_file = version_dir / "label.txt"
    source_file = version_dir / "source_tweets.txt"

    if not label_file.exists() or not source_file.exists():
        raise FileNotFoundError(f"Missing ACL2017 files in {version_dir}")

    # source_tweets format: <tweet_id>\t<text>
    source_texts = {}
    with source_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            tweet_id, text = line.split("\t", 1)
            source_texts[tweet_id.strip()] = text.strip()

    # labels format: <label>:<tweet_id>
    data = []
    with label_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label_str, tweet_id = [x.strip() for x in line.split(":", 1)]
            label = LABEL_MAP_BINARY.get(label_str.lower())
            if label is None:
                if not include_uncertain:
                    continue
                # keep uncertain classes mapped to fake
                label = 1

            text = source_texts.get(tweet_id, "")
            if not text:
                continue

            data.append((text, label))

    return data


def build_acl2017_csv(root: Path, output_csv: Path, include_uncertain: bool = False):
    """Create FakeNewsNet-compatible CSV from ACL2017 data."""
    all_records = []
    all_records.extend(
        _parse_acl2017_pairs(root, "twitter15", include_uncertain=include_uncertain)
    )
    all_records.extend(
        _parse_acl2017_pairs(root, "twitter16", include_uncertain=include_uncertain)
    )

    if not all_records:
        raise RuntimeError("No ACL2017 records parsed")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in all_records:
            # FakeNewsNet expects labels as strings 'fake'/'real' for this project
            writer.writerow([text, "fake" if label == 1 else "real"])

    return all_records


def _acl2017_stats(records):
    dist = Counter(label for _, label in records)
    return {
        "total": len(records),
        "fake": dist[1],
        "real": dist[0],
    }


class TestACL2017DataLoadAndSplit:
    """Validate the ACL2017 dataset conversion to FakeNewsNet format."""

    @pytest.mark.parametrize("include_uncertain", [False])
    def test_acl2017_dataset_parse_and_convert(self, include_uncertain):
        assert ACL2017_ROOT.exists(), f"Missing ACL2017 folder: {ACL2017_ROOT}"
        with tempfile.TemporaryDirectory() as td:
            temp_csv = Path(td) / "acl2017_converted.csv"
            records = build_acl2017_csv(
                ACL2017_ROOT,
                temp_csv,
                include_uncertain=include_uncertain,
            )
            assert temp_csv.exists()
            stats = _acl2017_stats(records)
            assert stats["total"] > 300
            assert stats["fake"] > 0 and stats["real"] > 0


class TestACL2017ExperimentRun:
    """Run a constrained experiment set on ACL2017 data."""

    def test_run_experiments_on_acl2017(self):
        if not _sklearn_available():
            pytest.skip(
                "scikit-learn is not installed in this environment. "
                "Install sklearn to run this experiment test."
            )

        assert ACL2017_ROOT.exists(), f"Missing ACL2017 folder: {ACL2017_ROOT}"

        models = ["LogisticRegression", "RandomForest"]
        if _xgboost_available():
            models.append("XGBoost")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "acl2017_fakenewsnet.csv"
            build_acl2017_csv(ACL2017_ROOT, csv_path, include_uncertain=False)

            config = {
                "datasets": ["FakeNewsNet"],
                "models": models,
                "ablations": {
                    "topology": [False],
                    "text": [True],
                },
                "data_paths": {
                    "fakenewsnet": str(csv_path),
                    "liar": None,
                    "pheme": None,
                },
                "parallel": {"max_workers": 1},
                "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "random_seed": 42,
                "output": {
                    "results_dir": "results_acl2017",
                    "save_plots": True,
                    "save_models": False,
                    "export_csv": True,
                },
            }

            cfg_path = td_path / "experiments_acl2017.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f)

            manager = ExperimentManager(
                config_path=str(cfg_path),
                results_dir=str(td_path / "results"),
                max_workers=1,
            )

            results = manager.run_single_dataset("FakeNewsNet", parallel=False)

            successful = [
                r for r in results
                if r.get("status") == "success"
                and not r.get("error")
                and not r.get("traceback")
            ]
            failed = [
                r for r in results
                if r.get("status") != "success" or r.get("error") or r.get("traceback")
            ]

            assert len(successful) >= 1, (
                f"No successful experiments; failures={[(r.get('task_id'), r.get('error')) for r in failed]}"
            )
            assert all("f1" in r and "accuracy" in r for r in successful)
