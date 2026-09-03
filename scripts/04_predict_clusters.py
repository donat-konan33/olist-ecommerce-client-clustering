"""Attribue un segment à de nouveaux clients RFMS avec le modèle gelé."""
import argparse
from pathlib import Path

import pandas as pd

from olist_ecommerce_client_clustering.model import FEATURES, RFMSClusteringModel


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prédiction de segments RFMS")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "artifacts/models/rfms_clustering.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.input)
    result = data.copy()
    result["cluster"] = RFMSClusteringModel.load(args.model).predict(data.loc[:, FEATURES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"{len(result)} prédictions écrites dans {args.output}")


if __name__ == "__main__":
    main()
