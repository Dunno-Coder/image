import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import BASELINE_MODELS, IMPROVEMENT_REPORT_CSV, RESULTS_CSV


def _best_rows_by_f1(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    numeric["f1"] = pd.to_numeric(numeric["f1"], errors="coerce")
    numeric = numeric.dropna(subset=["f1"])
    idx = numeric.groupby(["dataset_name", "model_name"])["f1"].idxmax()
    return numeric.loc[idx].reset_index(drop=True)


def build_improvement_report(results_csv: Path = RESULTS_CSV) -> Tuple[pd.DataFrame, float]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    required = {"dataset_name", "model_name", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required column(s): {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"].fillna("test") == "test"].copy()

    best = _best_rows_by_f1(df)
    rows = []

    for dataset_name in sorted(best["dataset_name"].dropna().unique()):
        dataset_rows = best[best["dataset_name"] == dataset_name]
        baseline_rows = dataset_rows[dataset_rows["model_name"].isin(BASELINE_MODELS)]
        proposed_rows = dataset_rows[dataset_rows["model_name"] == "proposed_mnff_edl"]

        if baseline_rows.empty or proposed_rows.empty:
            continue

        best_baseline = baseline_rows.sort_values("f1", ascending=False).iloc[0]
        proposed = proposed_rows.sort_values("f1", ascending=False).iloc[0]
        baseline_f1 = float(best_baseline["f1"])
        proposed_f1 = float(proposed["f1"])
        improvement_percent = (
            ((proposed_f1 - baseline_f1) / baseline_f1) * 100.0 if baseline_f1 > 0 else np.nan
        )

        rows.append(
            {
                "dataset_name": dataset_name,
                "best_baseline_model": best_baseline["model_name"],
                "best_baseline_f1": baseline_f1,
                "proposed_f1": proposed_f1,
                "improvement_percent": improvement_percent,
                "meets_5_percent": bool(improvement_percent >= 5.0),
                "meets_10_percent": bool(improvement_percent >= 10.0),
                "meets_20_percent": bool(improvement_percent >= 20.0),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        average_improvement = float("nan")
    else:
        average_improvement = float(report["improvement_percent"].mean())
        report = pd.concat(
            [
                report,
                pd.DataFrame(
                    [
                        {
                            "dataset_name": "AVERAGE",
                            "best_baseline_model": "",
                            "best_baseline_f1": "",
                            "proposed_f1": "",
                            "improvement_percent": average_improvement,
                            "meets_5_percent": bool(average_improvement >= 5.0),
                            "meets_10_percent": bool(average_improvement >= 10.0),
                            "meets_20_percent": bool(average_improvement >= 20.0),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return report, average_improvement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare proposed model with the best foundation baseline.")
    parser.add_argument("--results_csv", default=str(RESULTS_CSV))
    parser.add_argument("--output_csv", default=str(IMPROVEMENT_REPORT_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, average = build_improvement_report(Path(args.results_csv))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    print(report)
    print(f"Average improvement percent: {average:.4f}")
    print(f"Saved report to {output_csv}")


if __name__ == "__main__":
    main()

