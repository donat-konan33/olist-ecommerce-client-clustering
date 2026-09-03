"""Model de segmentation RFMS et persistence de ses artefacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler

FEATURES = ("recency", "frequency", "monetary", "review_score")


class RFMSClusteringModel:
    """Pipeline RFMS reproductible avec projection et affectation DBSCAN."""

    def __init__(self, umap_n_neighbors: int = 50, umap_random_state: int = 12,
                 dbscan_eps: float = 0.6, dbscan_min_samples: int = 5) -> None:
        self.n_neighbors = umap_n_neighbors
        self.random_state = umap_random_state
        self.eps = dbscan_eps
        self.min_samples = dbscan_min_samples
        self.pipeline: Pipeline | None = None
        self.X_ref_preprocessed: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(FEATURES).difference(X.columns)
        if missing:
            raise ValueError(f"Colonnes RFMS manquantes: {sorted(missing)}")
        if X.loc[:, FEATURES].isna().any().any():
            raise ValueError("Les variables RFMS ne doivent pas contenir de valeurs nulles")
        return X.loc[:, FEATURES]

    def fit(self, X_ref: pd.DataFrame) -> "RFMSClusteringModel":
        X = self._validate_features(X_ref)
        preprocessor = ColumnTransformer([
            ("skewed", Pipeline([("yeo", PowerTransformer(method="yeo-johnson")),
                                  ("scaler", RobustScaler())]), ["frequency", "monetary"]),
            ("normal", RobustScaler(), ["recency", "review_score"]),
        ])
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("umap", umap.UMAP(n_neighbors=self.n_neighbors, min_dist=0.1,
                                n_components=3, random_state=self.random_state)),
            ("model", DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)),
        ])
        self.pipeline.fit(X)
        self.X_ref_preprocessed = self.pipeline.named_steps["preprocessor"].transform(X)
        self.embedding_ = self.pipeline.named_steps["umap"].embedding_
        self.labels_ = self.pipeline.named_steps["model"].labels_.copy()
        return self

    def _require_fitted(self) -> Pipeline:
        if self.pipeline is None or self.embedding_ is None or self.labels_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant utilisation")
        return self.pipeline

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Projette des observations sans modifier le modèle de référence."""
        pipeline = self._require_fitted()
        values = self._validate_features(X)
        preprocessed = pipeline.named_steps["preprocessor"].transform(values)
        embedding = pipeline.named_steps["umap"].transform(preprocessed)
        return preprocessed, embedding

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Affecte au cluster du coeur DBSCAN le plus proche, sinon au bruit (-1)."""
        pipeline = self._require_fitted()
        _, embedding = self.transform(X)
        dbscan = pipeline.named_steps["model"]
        core_indices = dbscan.core_sample_indices_
        if len(core_indices) == 0:
            return np.full(len(embedding), -1, dtype=int)
        core_points = self.embedding_[core_indices]
        core_labels = self.labels_[core_indices]
        # Recherche indexée: une matrice dense n_current x n_core sature la mémoire dès ~10k points.
        distances, nearest = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(core_points).kneighbors(embedding)
        return np.where(distances[:, 0] <= self.eps, core_labels[nearest[:, 0]], -1).astype(int)

    def metrics(self) -> dict[str, float | int]:
        self._require_fitted()
        labels = self.labels_
        noise_ratio = float(np.mean(labels == -1))
        return {"nb_clusters": int(len(set(labels)) - int(-1 in labels)),
                "noise_ratio": noise_ratio, "clustered_ratio": 1 - noise_ratio}

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "RFMSClusteringModel":
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("Artefact de modèle RFMS invalide")
        return model
