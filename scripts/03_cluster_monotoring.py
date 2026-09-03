"""Compare un jeu RFMS courant à la référence et écrit un rapport de dérive."""
import argparse
from pathlib import Path

import pandas as pd

from olist_ecommerce_client_clustering.model import FEATURES, RFMSClusteringModel
from olist_ecommerce_client_clustering.monitoring import drift_report, save_report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Monitoring RFMS et détection de dérive")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "artifacts/models/rfms_clustering.pkl")
    parser.add_argument("--output", type=Path, default=root / "artifacts/monitoring/drift_report.json")
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    args = parser.parse_args()
    reference, current = pd.read_parquet(args.reference), pd.read_parquet(args.current)
    model = RFMSClusteringModel.load(args.model)
    report = drift_report(reference, current, model.predict(reference.loc[:, FEATURES]), model.predict(current.loc[:, FEATURES]), args.psi_threshold)
    save_report(report, args.output)
    print(f"Rapport écrit dans {args.output}; dérive détectée: {report['drift_detected']}")


if __name__ == "__main__":
    main()
