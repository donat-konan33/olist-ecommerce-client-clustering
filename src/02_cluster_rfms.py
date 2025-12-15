from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import umap
import pandas as pd
import numpy as np
import os
from pathlib import Path


# TODO: Pipeline complet de clustering
class RFMSClusteringModel:
    """Entire Clustering model class"""

    def __init__(self,
                 umap_n_neighbors=50,
                 umap_random_state=12,
                 dbscan_eps=0.6
                 ):

        self.n_neighbors = umap_n_neighbors
        self.random_state = umap_random_state
        self.eps = dbscan_eps

        self.pipeline = None
        self.embedding_ = None
        self.labels_ = None

    # création de notre transformer spécifique à nos données
    def fit(self, X):
        """
        Docstring for model_pipeline
        X: Data to input which labels must be predicted ;
        whatever features it contains, "frequency", "monetary", "recency"
        and "review_score" are mandatory
        """

        numerical_skewed = ["frequency", "monetary"]
        numerical_normal = ["recency", "review_score"]

        # umap transformer
        umap_transformer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.1,
            n_components=3,
            random_state=self.random_state
        )

        # dbscan model
        model = DBSCAN(
            eps=self.eps,
            min_samples=5,
            n_jobs=-1
        )

        preprocessor = ColumnTransformer(                         # liste les différentes transformations à appliquer
            transformers=[
                ('skewed', Pipeline(steps=[
                    ('yeo', PowerTransformer(method="yeo-johnson")),
                    ('scaler', RobustScaler())
                ]), numerical_skewed),
                ('normal', RobustScaler(), numerical_normal)
            ]
        )

        pipeline  = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("umap", umap_transformer)
                ("model", model)
            ]
        )

        self.pipeline.fit(X)  # recursive

        self.embedding_ = self.pipeline.named_steps["umap"].embedding_
        self.labels_ = self.pipeline.named_steps["model"].labels_

        return self


    def get_embedding(self):
        return self.embedding_

    def dbcv(self):
        """
        Get density validity of dbscan clustering model
        Output : a tuple of :
                    nb_cluster : Number of clusters
                    nb_noise : Number of noises
                    classified_percentage : Percentage of objects classified not as a noise : classified_percentage
        """
        from hdbscan.validity import validity_index

        labels = self.labels_
        X = self.embedding_

        nb_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        nb_noise = np.sum(labels == -1)
        classified_percentage = (len(labels) - nb_noise) / len(labels)
        score = validity_index(X.astype(np.float64), labels)

        return nb_cluster, nb_noise, classified_percentage, score



if __name__ == "__main__":
    # get processed data and predict labels
    pipeline = RFMSClusteringModel()

    here = Path().resolve()
    repo_wd = here.parent

    rfms_path = repo_wd / "data/processed/rfms_active_reviewers.parquet"
    rfms = pd.read_parquet(rfms_path)[["customer_unique_id", "recency", "frequency",
                                   "monetary", "review_score",
                                   "order_purchase_timestamp"]] # uniquement pour le client qui notent après commandes:

    df1 = rfms[rfms['order_purchase_timestamp'] < "2018-01-01"]
    df1 = df1[["customer_unique_id", 'recency', 'frequency', 'monetary', 'review_score']]

    X_processed, predicted_labels = pipeline.fit(df1)
    nb_cluster, nb_noise, classified_percentage, score = pipeline.dbcv()
