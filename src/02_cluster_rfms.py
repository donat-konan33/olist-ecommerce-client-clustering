import argparse
import pandas as pd
from pathlib import Path
from olist_ecommerce_client_clustering.utils import RFMSClusteringModel


if __name__ == "__main__":

    # --- argparse pour rendre la date flexible ---
    parser = argparse.ArgumentParser(description="RFMS clustering with custom split date")
    parser.add_argument(
        "--split-date",
        type=str,
        default="2017-12", # par défaut on segmente sur la baseline
        help="Date de coupure des données au format YYYY-MM"
    )
    args = parser.parse_args()
    split_date = args.split_date

    # obtenir les données nettoyées et prédire les labels
    clustering_pipeline = RFMSClusteringModel()

    here = Path(__file__).resolve()
    project_root = here.parent.parent

    rfms_path = project_root / "data/processed/rfms_active_reviewers.parquet"
    rfms = pd.read_parquet(rfms_path)[[
        "customer_unique_id", "recency", "frequency",
        "monetary", "review_score", "order_purchase_timestamp"
    ]]

    # données baseline avant split-date
    df1 = rfms[rfms['order_purchase_timestamp'] <= split_date][[
        "customer_unique_id", "recency", "frequency", "monetary", "review_score"
    ]]

    clustering_pipeline.fit(df1)
    labels_, nb_cluster, noise_ratio, clustered_ratio, dbcv_score = \
        clustering_pipeline.dbcv()

    print("=== Clustering effectué avec succès ===")
    print(f"Données concernées sont celles recoltées fin : {split_date}")
    print(f"Clusters: {nb_cluster} | Taux de bruit: {noise_ratio:.3f} | DBCV: {dbcv_score:.3f}")

    # -------- Sauvegarde des labels --------
    data_dir = project_root / "data/clustered"
    data_dir.mkdir(parents=True, exist_ok=True)

    labels = df1.copy()
    labels["cluster"] = labels_

    output_path = data_dir / f"clusters_labels_until_{split_date}.parquet"
    labels.to_parquet(output_path, index=False, engine="fastparquet")

    print(f"Les données étiquettées stockées ici : {output_path}")

    # ------ Stocker les méta données de la performance de clustering -------
    artifacts_dir = project_root / "artifacts/cluster_performance"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    meta_data = {
        "split_date": split_date,
        "nb_clusters": nb_cluster,
        "noise_ratio": noise_ratio,
        "clustered_ratio": clustered_ratio,
        "dbcv_score": dbcv_score
    }
    meta_output_path = artifacts_dir / f"clustering_performance_until_{split_date}.json"
    with open(meta_output_path, "w") as f:
        import json
        json.dump(meta_data, f, indent=4)
    print(f"Méta données de performance stockées ici : {meta_output_path}")
