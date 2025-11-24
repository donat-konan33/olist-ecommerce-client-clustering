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


1. Clustering RFM + Feature Engineering pour Segmentation marketing

🎯 Objectif

Créer une segmentation marketing avancée basée sur RFM enrichi avec les données du dataset

A. **Préparation et intégration RFM**

- Créer les métriques classiques :

  - Récence : date d’achat la plus récente

  - Fréquence : nombre de commandes

  - Monétaire : montant total dépensé

B. **Feature Engineering Marketing**

Ajouter des variables complémentaires :

- catégorie préférée
- panier moyen
- fidélité (répétition de vendeurs)
- délai moyen de livraison
- taux de retour ou remboursement
- sentiment moyen des reviews (si existe)
- Données externes possibles :
- socio-démographie par code postal
- revenus moyens par région
- IPCA / inflation (corrélation prix vs satisfaction)

C. Clustering

Standardisation

Dimension reduction (PCA)

Tester K-Means, GMM, Agglomerative

Valider les clusters (Silhouette, Davies-Bouldin)

D. Interprétation marketing

Créer des personas :

Premium Loyalists

Bargain Hunters

Low-Frequency High-Value buyers

At-risk customers

Early-churners

E. Livrables

Strategy book de segmentations

Recommandations marketing :

email ciblé selon cluster

promos pour réactiver certains segments

programmes de fidélité adaptés à la valeur client



---

Remerciement à Olist pour leurs données open source :
```
@misc{olist_andr__sionek_2018,
	title={Brazilian E-Commerce Public Dataset by Olist},
	url={https://www.kaggle.com/dsv/195341},
	DOI={10.34740/KAGGLE/DSV/195341},
	publisher={Kaggle},
	author={Olist and André Sionek},
	year={2018}
}```
