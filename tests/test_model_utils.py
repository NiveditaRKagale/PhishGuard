import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import src.model_utils as model_utils


def test_split_data_is_stratified_and_disjoint():
    X = pd.DataFrame({"row": range(120)})
    y = np.repeat([0, 1], 60)

    X_train, X_validation, X_test, y_train, y_validation, y_test = model_utils.split_data(X, y)

    assert [len(part) for part in (X_train, X_validation, X_test)] == [72, 24, 24]
    assert not (set(X_train.index) & set(X_validation.index))
    assert not (set(X_train.index) & set(X_test.index))
    assert not (set(X_validation.index) & set(X_test.index))
    assert all(set(labels) == {0, 1} for labels in (y_train, y_validation, y_test))


def test_train_and_save_selects_on_validation_and_reports_test(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, len(model_utils.FEATURE_NAMES))), columns=model_utils.FEATURE_NAMES)
    y = np.repeat([0, 1], 60)
    monkeypatch.setattr(model_utils, "load_dataset", lambda _: object())
    monkeypatch.setattr(model_utils, "build_matrix", lambda _: (X, y))
    monkeypatch.setattr(model_utils, "get_models", lambda: {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=5, random_state=42),
    })
    monkeypatch.setattr(model_utils.config, "MODEL", tmp_path / "model.joblib")
    monkeypatch.setattr(model_utils.config, "METRICS", tmp_path / "metrics.json")

    results, best, _ = model_utils.train_and_save("unused.csv")

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    expected = max(metrics["validation_results"], key=lambda name: (
        metrics["validation_results"][name]["f1"],
        metrics["validation_results"][name]["recall"],
        name == "Random Forest",
    ))
    assert best == expected
    assert set(results) == set(metrics["results"])
    assert (metrics["n_train"], metrics["n_validation"], metrics["n_test"]) == (72, 24, 24)
    assert metrics["selection_metric"] == "validation_f1"
    assert model_utils.load_bundle()["model_name"] == best