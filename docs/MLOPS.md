# Exploitation ML

## Entraîner et tracer

```bash
poetry run python scripts/02_cluster_rfms.py --split-date 2017-12-31 \
  --mlflow-tracking-uri file:./mlruns
```

Le script crée le modèle, les labels et les métriques. Avec l'URI MLflow, il journalise paramètres, métriques et artefact. Interface: `poetry run mlflow ui --backend-store-uri ./mlruns`.

## Prédire

Le parquet doit contenir `recency`, `frequency`, `monetary`, `review_score`.

```bash
poetry run python scripts/04_predict_clusters.py --input data/processed/rfms_active_reviewers.parquet --output data/predictions/rfms_scored.parquet
```

DBSCAN n'a pas de prédiction native: un client est affecté au point de coeur historique le plus proche dans UMAP, sinon au bruit (`-1`).

## Monitoring et drift

```bash
poetry run python scripts/03_cluster_monotoring.py --reference data/clustered/clusters_labels_until_2017-12-31.parquet --current data/processed/rfms_active_reviewers.parquet
```

Le rapport JSON contient le PSI par variable et la distance de Jensen-Shannon des clusters. PSI >= 0,20 déclenche `drift_detected`; inspecter puis réentraîner après validation métier.
