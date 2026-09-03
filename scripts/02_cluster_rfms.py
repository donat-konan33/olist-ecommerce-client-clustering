"""Entraîne, évalue et persiste la segmentation RFMS."""
import argparse
import json
from pathlib import Path

import pandas as pd

from olist_ecommerce_client_clustering.model import FEATURES, RFMSClusteringModel


def log_mlflow(params, metrics, artifact, tracking_uri):
    """Journalise dans MLflow seulement si l'option est demandée."""
    if not tracking_uri:
        return
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("olist-rfms-clustering")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(artifact))


def main():
    parser = argparse.ArgumentParser(description="Entraînement de la segmentation RFMS")
    parser.add_argument("--split-date", default="2017-12-31")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--mlflow-tracking-uri")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.input or root / "data/processed/rfms_active_reviewers.parquet"
    rfms = pd.read_parquet(source)
    baseline = rfms.loc[rfms["order_purchase_timestamp"] <= pd.Timestamp(args.split_date)]
    model = RFMSClusteringModel().fit(baseline)
    metrics = model.metrics()
    model_path = root / "artifacts/models/rfms_clustering.pkl"
    model.save(model_path)
    labels_path = root / f"data/clustered/clusters_labels_until_{args.split_date}.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.loc[:, ["customer_unique_id", *FEATURES]].assign(cluster=model.labels_).to_parquet(labels_path, index=False)
    metadata = {"split_date": args.split_date, "n_observations": len(baseline), **metrics}
    metadata_path = root / f"artifacts/cluster_performance/clustering_performance_until_{args.split_date}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_mlflow({"split_date": args.split_date, "n_neighbors": model.n_neighbors, "eps": model.eps}, metrics, model_path, args.mlflow_tracking_uri)
    print(json.dumps({**metadata, "model": str(model_path)}, indent=2))


if __name__ == "__main__":
    main()
