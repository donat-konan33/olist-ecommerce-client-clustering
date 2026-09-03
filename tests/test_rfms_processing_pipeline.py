import importlib.util
from pathlib import Path

import pandas as pd


def load_pipeline():
    path = Path(__file__).parents[1] / "scripts/01_rfms_processing_pipeline.py"
    spec = importlib.util.spec_from_file_location("rfms_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RFMSProcessingPipeline


def test_calculate_monetary_and_review_split(tmp_path):
    pipeline = load_pipeline()(tmp_path)
    pipeline.items_df = pd.DataFrame({"order_id": ["a", "a", "b"], "order_item_id": [1, 2, 1], "price": [10., 5., 20.], "freight_value": [2., 3., 4.]})
    monetary = pipeline.calculate_monetary().set_index("order_id")
    assert monetary.loc["a", "number_of_items"] == 2
    assert monetary.loc["a", "total_amount"] == 20
    active, silent = pipeline.split_by_review_status(pd.DataFrame({"review_score": [4., None]}))
    assert len(active) == 1
    assert len(silent) == 1


def test_preprocess_reviews_aggregates_multiple_reviews(tmp_path):
    pipeline = load_pipeline()(tmp_path)
    pipeline.reviews_df = pd.DataFrame({"order_id": ["a", "a", "b"], "review_score": [4, 5, 1], "comment": ["x", "y", "z"]})
    pipeline.preprocess_reviews()
    reviews = pipeline.reviews_df.set_index("order_id")
    assert reviews.loc["a", "review_score"] == 4.5
    assert set(reviews.index.astype(str)) == {"a", "b"}
