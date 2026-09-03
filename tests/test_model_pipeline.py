import numpy as np
import pandas as pd
import pytest

from olist_ecommerce_client_clustering.model import RFMSClusteringModel
from olist_ecommerce_client_clustering.monitoring import (
    drift_report,
    population_stability_index,
)


def rfms(rows=60):
    rng = np.random.default_rng(42)
    first = pd.DataFrame({"recency": rng.normal(15, 2, rows // 2), "frequency": rng.integers(2, 6, rows // 2), "monetary": rng.normal(150, 10, rows // 2), "review_score": rng.normal(4.5, .1, rows // 2)})
    second = pd.DataFrame({"recency": rng.normal(90, 3, rows // 2), "frequency": rng.integers(8, 12, rows // 2), "monetary": rng.normal(700, 20, rows // 2), "review_score": rng.normal(2., .1, rows // 2)})
    return pd.concat([first, second], ignore_index=True)


def test_model_predict_does_not_refit_reference_model(tmp_path):
    data = rfms()
    model = RFMSClusteringModel(umap_n_neighbors=10, dbscan_eps=1.0).fit(data)
    baseline_labels = model.labels_.copy()
    predictions = model.predict(data.iloc[:8])
    assert len(predictions) == 8
    np.testing.assert_array_equal(model.labels_, baseline_labels)
    path = tmp_path / "model.pkl"
    model.save(path)
    np.testing.assert_array_equal(RFMSClusteringModel.load(path).predict(data.iloc[:8]), predictions)


def test_model_rejects_missing_or_null_features():
    with pytest.raises(ValueError, match="manquantes"):
        RFMSClusteringModel().fit(pd.DataFrame({"recency": [1]}))
    data = rfms(10)
    data.loc[0, "monetary"] = np.nan
    with pytest.raises(ValueError, match="nulles"):
        RFMSClusteringModel().fit(data)


def test_drift_report_detects_shift_and_keeps_stable_population_clean():
    reference = rfms(100)
    stable = reference.copy()
    shifted = reference.assign(monetary=reference["monetary"] + 10_000)
    assert population_stability_index(reference["monetary"], stable["monetary"]) == 0
    assert drift_report(reference, stable)["drift_detected"] is False
    report = drift_report(reference, shifted)
    assert report["drift_detected"] is True
    assert report["feature_psi"]["monetary"] > .2
