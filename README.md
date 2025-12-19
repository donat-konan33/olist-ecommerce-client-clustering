# OList Ecommerce Client Clustering

Segmentation marketing basée sur RFM enrichi pour identifier et analyser les profils clients du dataset Brazilian E-Commerce d'Olist.

## 📊 Objectif

Créer une **segmentation marketing exploitable** différenciant les bons et moins bons clients en termes de comportement d'achat et de satisfaction, avec recommandations de maintenance du modèle.

## 📁 Structure du projet

```
olist-ecommerce-client-clustering/
.
├── artifacts/              # Configs + résultats de monitoring
│   ├── config/
│   └── metrics_embeddings/
│
├── assets/                 # Images pour documentation
│   └── images/
│
├── data/                   # Données du projet
│   ├── raw/                # données brutes Olist
│   └── processed/          # données transformées (RFMS)
│
├── notebooks/              # Notebooks d'analyse & expérimentation
│
├── outputs/                # Résultats du projet
│   ├── figures/
│   └── reports/
│
├── scripts/                # Scripts utilitaires
│
├── src/                    # Code source & pipelines
│   ├── 01_rfms_processing_pipeline.py
│   ├── 02_cluster_rfms.py
│   ├── 03_cluster_monotoring.py
│   └── olist_ecommerce_client_clustering/
│
├── tests/                  # Tests unitaires
│
├── pyproject.toml
└── README.md

```

## 🚀 Installation

### 1. Télécharger les données

```bash
# Configurer votre clé Kaggle, puis:
./scripts/download-data.sh
```

Les 5 fichiers CSV seront extraits dans `data/raw/`.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécuter la pipeline RFM

```bash
python src/01_rfms_processing_pipeline.py
```

Génère deux fichiers parquet dans `data/processed/`:
- `rfms_active_reviewers.parquet` (98k+ clients avec avis)
- `rfms_silent_customers.parquet` (~1k clients sans avis)

```bash
python src/02_cluster_rfms.py --split-date '2017-12'
```

Génère deux fichiers :
- **parquet** dans ``data/clustered``:
  - ``clusters_labels_until_2017-12.parquet``

- **json** dans ``artifacts/cluster_performance``
  - ``clustering_performance_until_2017-12.json``


## 📈 Livrables

| Phase | Fichier | Contenu |
|-------|---------|---------|
| **Exploration** | `notebooks/01_eda.ipynb` | Analyse détaillée des sources, justification des choix RFM |
| **Feature Engineering** | `src/01_rfms_processing_pipeline.py` | Code production (PEP8) : chargement → transformation → export parquet |
| **Modélisation** | `notebooks/02_clustering.ipynb` | Comparaison K-Means, GMM, Agglomerative, DBSCAN/HDBSCAN ; sélection meilleur modèle |
| **Maintenance** | `notebooks/03_cluster_monotoring.ipynb` | Recommandation fréquence mise à jour ; Data Drift et Clustering drift  |
| **Personas** | `notebooks/04_cluster_profiling.ipynb` | Définir les segments marketing exploitables |

## 📊 Métriques RFM

Par client (customer_unique_id):

- **Recency (R)**: Jours depuis dernier achat
- **Frequency (F)**: Nombre total de commandes
- **Monetary (M)**: Montant total dépensé (€)
- **Satisfaction (S)**: Score moyen des avis (1-5)


## 💡 Points clés

✅ **Segmentation stable et maintenable** : Algorithme reproductible, fréquence de mise à jour définie

✅ **Exploitable par Marketing** : Personas clairs avec actions recommandées

✅ **Scalable** : Intégration de nouveaux clients définie

✅ **Code production** : Respecte PEP8 et conventions Olist

## 📚 Données source

Dataset Olist pour la RFM (~100k commandes, 2016-2018):
- olist_orders_dataset
- olist_customers_dataset
- olist_order_items_dataset
- olist_order_payments_dataset
- olist_order_reviews_dataset

## 📖 Schéma de la base

![olist-database-modeling](assets/images/olist-database-modeling.png)

## 🙏 Remerciements

```
@misc{olist_andr__sionek_2018,
	title={Brazilian E-Commerce Public Dataset by Olist},
	url={https://www.kaggle.com/dsv/195341},
	DOI={10.34740/KAGGLE/DSV/195341},
	publisher={Kaggle},
	author={Olist and André Sionek},
	year={2018}
}
```


---

**Status**: En cours de développement

**Dernière mise à jour**: Décembre 2025
