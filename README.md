# OList Ecommerce Client Clustering

Segmentation marketing basée sur RFM enrichi pour identifier et analyser les profils clients du dataset Brazilian E-Commerce d'Olist.

![olist-database-modeling](assets/images/olist-database-modeling.png)


# OList Ecommerce Client Clustering

Segmentation marketing basée sur RFM enrichi pour identifier et analyser les profils clients du dataset Brazilian E-Commerce d'Olist.

## 📊 Objectif

Créer une **segmentation marketing exploitable** différenciant les bons et moins bons clients en termes de comportement d'achat et de satisfaction, avec recommandations de maintenance du modèle.

## 📁 Structure du projet

```
olist-ecommerce-client-clustering/
├── notebooks/
│   ├── 01_eda.ipynb                      # Analyse exploratoire des données
│   ├── 02_modeling.ipynb                 # Tests des algorithmes de clustering
│   └── 03_simulation.ipynb               # Fréquence de mise à jour du modèle
├── src/
│   └── 01_rfms_processing_pipeline.py    # Pipeline de traitement RFM
├── data/
│   ├── raw/                              # Données brutes Olist (9 CSV)
│   └── processed/                        # Données transformées (Parquet)
├── assets/images/
│   └── olist-database-modeling.png       # Schéma de la base de données
├── scripts/
│   └── download-data.sh                  # Téléchargement Kaggle
└── requirements.txt
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

## 📈 Livrables

| Phase | Fichier | Contenu |
|-------|---------|---------|
| **Exploration** | `notebooks/01_eda.ipynb` | Analyse détaillée des 5 sources, justification des choix RFM |
| **Feature Engineering** | `src/01_rfms_processing_pipeline.py` | Code production (PEP8) : chargement → transformation → export parquet |
| **Modélisation** | `notebooks/02_modeling.ipynb` | Comparaison K-Means, GMM, Agglomerative ; sélection meilleur modèle |
| **Personas** | `notebooks/02_modeling.ipynb` | 5 segments marketing exploitables |
| **Maintenance** | `notebooks/03_simulation.ipynb` | Recommandation fréquence mise à jour ; drift detection |

## 📊 Métriques RFM

Par client (customer_unique_id):

- **Recency (R)**: Jours depuis dernier achat
- **Frequency (F)**: Nombre total de commandes
- **Monetary (M)**: Montant total dépensé (€)
- **Satisfaction (S)**: Score moyen des avis (1-5)

## 🎯 Personas Marketing attendus

1. **Premium Loyalists** → Haute fréquence + dépense élevée + satisfaction
2. **Bargain Hunters** → Fréquence moyenne + basse dépense
3. **High-Value Buyers** → Basse fréquence + dépense très élevée
4. **At-Risk Customers** → Recency très élevée (inactifs)
5. **Early-Churners** → Très peu de commandes

## 💡 Points clés

✅ **Segmentation stable et maintenable** : Algorithme reproductible, fréquence de mise à jour définie
✅ **Exploitable par Marketing** : Personas clairs avec actions recommandées
✅ **Scalable** : Intégration de nouveaux clients définie
✅ **Code production** : Respecte PEP8 et conventions Olist

## 📚 Données source

Dataset Olist (~100k commandes, 2016-2018):
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
**Dernière mise à jour**: Novembre 2025
