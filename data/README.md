# Données du projet

Ce dossier regroupe les données utilisées par le pipeline de segmentation des
clients Olist. Les sous-dossiers indiquent l'étape du traitement et permettent
de distinguer les données sources des résultats calculés.

## Organisation

### `raw/`

Données originales du dataset public Olist téléchargées depuis Kaggle. Elles ne
sont pas modifiées par les scripts.

Le script `scripts/download-data.sh` crée ce dossier et télécharge les fichiers
CSV avec la commande suivante :

```bash
./scripts/download-data.sh
```

Les fichiers sont notamment :

- `olist_orders_dataset.csv` : commandes, clients rattachés, statut et dates ;
- `olist_customers_dataset.csv` : identifiants clients et localisation ;
- `olist_order_items_dataset.csv` : articles, prix et frais de livraison ;
- `olist_order_payments_dataset.csv` : moyens de paiement et montants réglés ;
- `olist_order_reviews_dataset.csv` : notes et commentaires des avis ;
- `olist_products_dataset.csv` : informations sur les produits ;
- `olist_sellers_dataset.csv` : informations sur les vendeurs ;
- `olist_geolocation_dataset.csv` : correspondance des codes postaux et des
  coordonnées géographiques ;
- `product_category_name_translation.csv` : traduction des catégories de
  produits.

Le pipeline RFMS charge directement les cinq premiers fichiers listés ci-dessus.
Les autres fichiers restent disponibles pour l'EDA et d'éventuelles analyses
portant sur les produits, les vendeurs ou la géographie.

### `processed/`

Fichiers Parquet créés par
`scripts/01_rfms_processing_pipeline.py` à partir des données de `raw/` :

- `rfms_data.parquet` : table RFMS complète, agrégée par `customer_unique_id` ;
- `rfms_active_reviewers.parquet` : clients dont la note moyenne est connue ;
- `rfms_silent_customers.parquet` : clients sans avis, donc sans `review_score`.

Chaque ligne représente un client. Les variables principales sont `recency`,
`frequency`, `monetary` et `review_score`, accompagnées de la date du dernier
achat et des informations de localisation majoritaires.

Génération :

```bash
poetry run python scripts/01_rfms_processing_pipeline.py
```

### `clustered/`

Fichiers Parquet produits par `scripts/02_cluster_rfms.py` après entraînement
du modèle UMAP + DBSCAN :

- `clusters_labels_until_2017-12-31.parquet` : clients de la période de
  référence, leurs variables RFMS et leur étiquette `cluster`.

La date intégrée au nom dépend de l'option `--split-date`. Le fichier est créé
avec la commande :

```bash
poetry run python scripts/02_cluster_rfms.py --split-date 2017-12-31
```

### `predictions/`

Fichiers Parquet contenant les résultats du scoring avec le modèle sauvegardé
dans `artifacts/models/`. Le script conserve les colonnes d'entrée et ajoute la
colonne `cluster`.

Exemple :

```bash
poetry run python scripts/04_predict_clusters.py \
  --input data/processed/rfms_active_reviewers.parquet \
  --output data/predictions/rfms_scored.parquet
```

## Reproductibilité

Les fichiers de `processed/`, `clustered/` et `predictions/` sont des résultats
générés : ils peuvent être recréés à partir de `raw/`, des scripts et du modèle
entraîné. Les données brutes peuvent être volumineuses ; leur présence ou leur
stockage externe doit respecter les conditions de distribution du dataset
Olist. Pour reproduire une exécution, téléchargez les données, installez les
dépendances avec `poetry install`, puis exécutez les commandes ci-dessus dans
l'ordre.
