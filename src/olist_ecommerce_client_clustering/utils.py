import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler


# TODO: Pipeline complet de clustering
class RFMSClusteringModel:
    """Entire Clustering model class"""

    def __init__(self, umap_n_neighbors=50, umap_random_state=12, dbscan_eps=0.6):

        self.n_neighbors = umap_n_neighbors
        self.random_state = umap_random_state
        self.eps = dbscan_eps

        self.pipeline = None
        self.embedding_ = None
        self.labels_ = None
        self.X_preprocessed = None

        self.preprocessor = None
        self.umap_transformer = None
        self.model = None

        self.nb_cluster = None
        self.nb_noise = None
        self.clustered_ratio = None
        self.dbcv_score = None

    # création de notre transformer spécifique à nos données
    def fit(self, X_ref):
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
            random_state=self.random_state,
        )

        # dbscan model
        model = DBSCAN(eps=self.eps, min_samples=5, n_jobs=-1)

        preprocessor = (
            ColumnTransformer(  # liste les différentes transformations à appliquer
                transformers=[
                    (
                        "skewed",
                        Pipeline(
                            steps=[
                                ("yeo", PowerTransformer(method="yeo-johnson")),
                                ("scaler", RobustScaler()),
                            ]
                        ),
                        numerical_skewed,
                    ),
                    ("normal", RobustScaler(), numerical_normal),
                ]
            )
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("umap", umap_transformer),
                ("model", model),
            ]
        )

        self.pipeline.fit(X_ref)  # recursive

        # baseline objects for next drifts predictions
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.umap_transformer = self.pipeline.named_steps["umap"]
        self.model = self.pipeline.named_steps["model"]

        # baseline data
        self.X_ref_preprocessed = self.pipeline.named_steps["preprocessor"].transform(
            X_ref
        )
        self.embedding_ = self.pipeline.named_steps["umap"].embedding_
        self.labels_ = self.pipeline.named_steps["model"].labels_

        return self  # mise à jour des attributs

    def transform(self, X_cur):
        """
        Transform new data using the fitted pipeline (no refit) : neede for drift detection
        """
        # Same scaling as T0
        X_cur_preprocessed = self.preprocessor.transform(X_cur)  # data drift

        # Project into T0 UMAP space
        X_cur_embedding = self.umap_transformer.transform(
            X_cur_preprocessed
        )  # representative drift

        # Reclustering in the same space
        labels_cur = self.model.fit_predict(X_cur_embedding)  # cluster drift

        return X_cur_preprocessed, X_cur_embedding, labels_cur

    def get_xref_preprocessed(self, is_contained_feature: bool = True):
        """
        Need to fit data after calling this method

        :param self: Description
        :param is_contained_feature: Description
        :type is_contained_feature: bool
        """
        try:
            if is_contained_feature:
                columns = ["recency", "frequency", "monetary", "review_score"]
                return pd.DataFrame(data=self.X_preprocessed, columns=columns)
        except Exception as e:
            print(
                f"Likely data is not still fitted So it is {self.X_preprocessed} value: {e}"
            )

    def get_umap_baseline_embedding(self):
        return self.embedding_

    def get_baseline_labels(self):
        return self.labels_

    def dbcv(self):
        """
        Get density validity of dbscan clustering model
        Output : a tuple of :
                    nb_cluster : Number of clusters
                    noise_ratio : Number of noises
                    clustered_ratio : ratio of objects classfied not as a noise
        """
        from hdbscan.validity import validity_index

        labels = self.labels_
        X = self.embedding_

        self.nb_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        self.noise_ratio = np.sum(labels == -1)
        self.clustered_ratio = 1 - self.noise_ratio
        self.dbcv_score = validity_index(X.astype(np.float64), labels)

        return (
            self.labels_,
            self.nb_cluster,
            self.noise_ratio,
            self.clustered_ratio,
            self.dbcv_score,
        )
