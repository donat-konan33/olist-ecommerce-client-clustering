"""Profile les segments entraînés: table de personas + projection UMAP colorée."""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from olist_ecommerce_client_clustering.model import RFMSClusteringModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#4C6EF5", "#12B886", "#F59F00", "#E8590C", "#AE3EC9", "#1098AD", "#F03E3E", "#495057"]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Profilage des segments RFMS")
    parser.add_argument("--labels", type=Path,
                        default=root / "data/clustered/clusters_labels_until_2017-12-31.parquet")
    parser.add_argument("--model", type=Path, default=root / "artifacts/models/rfms_clustering.pkl")
    parser.add_argument("--figure", type=Path, default=root / "outputs/figures/clusters_umap.png")
    parser.add_argument("--report", type=Path, default=root / "outputs/reports/cluster_personas.csv")
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    profile = (labels.groupby("cluster")
               .agg(clients=("cluster", "size"),
                    recency=("recency", "median"),
                    frequency=("frequency", "mean"),
                    monetary=("monetary", "median"),
                    review_score=("review_score", "mean"))
               .round(2))
    profile["part_%"] = (100 * profile["clients"] / len(labels)).round(1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.report)
    print(profile.to_string())

    embedding = RFMSClusteringModel.load(args.model).embedding_
    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")
    for index, cluster in enumerate(sorted(labels["cluster"].unique())):
        mask = (labels["cluster"] == cluster).to_numpy()
        color = "#CED4DA" if cluster == -1 else PALETTE[index % len(PALETTE)]
        name = "Bruit (-1)" if cluster == -1 else f"Segment {cluster}"
        axes.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                     s=2, alpha=0.5, c=color, label=f"{name} — {mask.sum():,} clients".replace(",", " "))
    axes.set_xlabel("UMAP-1"), axes.set_ylabel("UMAP-2"), axes.set_zlabel("UMAP-3")
    axes.set_title("Segments clients Olist — projection UMAP 3D (DBSCAN)")
    axes.legend(loc="upper left", markerscale=6, fontsize=8, frameon=False)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=150, bbox_inches="tight")
    print(f"Figure écrite dans {args.figure}")


if __name__ == "__main__":
    main()
