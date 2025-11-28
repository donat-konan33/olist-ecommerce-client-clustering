# OList Ecommerce Clustering issue solved

Here are all data available about its database:

Data Schema

![olist-database-modeling](assets/images/olist-database-modeling.png)

This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces.

Attention You must focus on :

1. An order might have multiple items.
2. Each item might be fulfilled by a distinct seller.
3. All text identifying stores and partners where replaced by the names of Game of Thrones great houses.

Download data:

Put to scripts/download-data.sh

```

#!/bin/bash
kaggle datasets download olistbr/brazilian-ecommerce

```

and then hit this command in your terminal:

```
./scripts/download-data.sh
```


1. **Clustering RFM + Feature Engineering pour Segmentation marketing**

**🎯 Objectif**

Créer une segmentation marketing avancée basée sur RFM enrichi avec les données du dataset

A. **Feature Engineering Marketing : Préparation et intégration RFM**

- Créer les métriques classiques :

  - Récence : date d’achat la plus récente
  - Fréquence : nombre de commandes
  - Monétaire : montant total dépensé


**C. Clustering**

- Standardisation
- Dimension reduction (PCA)
- Tester K-Means, GMM, Agglomerative
- Valider les clusters (Silhouette, Davies-Bouldin)


**La segmentation proposée doit être exploitable et facile d’utilisation par notre équipe Marketing**. Elle doit au minimum **pouvoir différencier les bons et moins bons clients** en termes de commandes et de satisfaction. Nous attendons bien sûr une segmentation sur l’ensemble des clients.

Dans un deuxième temps, une fois le modèle de segmentation choisi, nous souhaiterions  que vous nous fassiez **une recommandation de fréquence à laquelle la segmentation doit être mise à jour pour rester pertinente**, afin de pouvoir effectuer **un devis de contrat de maintenance**.


**D. Interprétation marketing**

Créer des personas :

- Premium Loyalists
- Bargain Hunters
- Low-Frequency High-Value buyers
- At-risk customers
- Early-churners

**E. Livrables**

Strategy book de **traitement des données (data pipeline), de segmentations (simple et interprétable par l'équipe Marketing) et de maintenance du modèle de segmentation régulière prenant en compte le nouveaux clients et les nouveaux comportement (but: segmentation cohérente)**

- Un notebook de l'analyse exploratoire (non cleané, pour comprendre la démarche d'acquisition de données RFM).
- Un notebook (ou code commenté au choix) d’essais des différentes approches de modélisation (non cleané, pour comprendre la démarche de modélisation).
- Un notebook de simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation (à une éventuelle dérive du modèle).
**NB** : Le code fourni doit respecter la **convention PEP8**, pour être utilisable par Olist.

---
## Suggestion de **Feature Engineering Marketing**

Ajouter des variables complémentaires :

- catégorie préférée
- panier moyen
- fidélité (répétition de vendeurs)
- délai moyen de livraison
- taux de retour ou remboursement
- sentiment moyen des reviews
  - Données externes possibles :
  - socio-démographie par code postal
  - revenus moyens par région
  - IPCA / inflation (corrélation prix vs satisfaction)

---

## Reconnaissances

Tous mes remerciements à Olist pour leurs données open source :

```
@misc{olist_andr__sionek_2018,
	title={Brazilian E-Commerce Public Dataset by Olist},
	url={https://www.kaggle.com/dsv/195341},
	DOI={10.34740/KAGGLE/DSV/195341},
	publisher={Kaggle},
	author={Olist and André Sionek},
	year={2018}
}```
