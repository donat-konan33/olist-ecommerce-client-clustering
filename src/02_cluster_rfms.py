
import pandas as pd
from pathlib import Path
from olist_ecommerce_client_clustering.utils import RFMSClusteringModel



if __name__ == "__main__":
    # get processed data and predict labels
    pipeline = RFMSClusteringModel()

    here = Path().resolve()
    repo_wd = here.parent

    rfms_path = repo_wd / "data/processed/rfms_active_reviewers.parquet"
    rfms = pd.read_parquet(rfms_path)[["customer_unique_id", "recency", "frequency",
                                   "monetary", "review_score",
                                   "order_purchase_timestamp"]] # uniquement pour le client qui notent après commandes:

    # données clients de la période avant "2018-01-01" ayant servi au choix du pipeline du modèle
    df1 = rfms[rfms['order_purchase_timestamp'] < "2018-01-01"]
    df1 = df1[["customer_unique_id", 'recency', 'frequency', 'monetary', 'review_score']]

    # données clients de la période après "2018-01-01"
    df2 =  rfms[rfms['order_purchase_timestamp'] >= "2018-01-01"]
    df2 = df2[["customer_unique_id", 'recency', 'frequency', 'monetary',
               'review_score', "order_purchase_timestamp"]]

    pipeline.fit(df1)
    X_processed = pipeline.get_umap_baseline_embedding()
    predicted_labels = pipeline.get_baseline_labels()
    labels_, nb_cluster, noise_ratio, clustered_ratio, dbcv_score = pipeline.dbcv()
