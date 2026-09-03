"""Calcul de dérive de données et de dérive de segmentation RFMS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .model import FEATURES


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI par quantiles de référence; supérieur à .2 signale une dérive notable."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref, _ = np.histogram(reference.dropna(), bins=edges)
    cur, _ = np.histogram(current.dropna(), bins=edges)
    ref_share = np.clip(ref / max(ref.sum(), 1), 1e-6, None)
    cur_share = np.clip(cur / max(cur.sum(), 1), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 reference_labels: np.ndarray | None = None,
                 current_labels: np.ndarray | None = None,
                 psi_threshold: float = 0.2) -> dict:
    missing = set(FEATURES).difference(reference.columns) | set(FEATURES).difference(current.columns)
    if missing:
        raise ValueError(f"Colonnes RFMS manquantes: {sorted(missing)}")
    feature_psi = {feature: population_stability_index(reference[feature], current[feature])
                   for feature in FEATURES}
    report: dict = {"feature_psi": feature_psi,
                    "drift_detected": any(value >= psi_threshold for value in feature_psi.values()),
                    "psi_threshold": psi_threshold}
    if reference_labels is not None and current_labels is not None:
        labels = np.union1d(reference_labels, current_labels)
        ref_dist = np.array([(reference_labels == label).mean() for label in labels])
        cur_dist = np.array([(current_labels == label).mean() for label in labels])
        report["cluster_js_distance"] = float(jensenshannon(ref_dist, cur_dist))
        report["reference_noise_ratio"] = float(np.mean(reference_labels == -1))
        report["current_noise_ratio"] = float(np.mean(current_labels == -1))
    return report


def save_report(report: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
