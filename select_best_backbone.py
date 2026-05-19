import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import BASELINE_MODELS, BEST_BACKBONE_CONFIG_JSON, RESULTS_CSV


SUPPORTED_SELECTION_METRICS = ("f1", "auroc", "balanced_accuracy", "accuracy")
MEAN_METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auroc",
    "uncertainty_mean",
    "inference_time_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best Vision Foundation backbone from results.csv.")
    parser.add_argument(
        "--metric",
        default="f1",
        choices=SUPPORTED_SELECTION_METRICS,
        help="Metric used to select the best candidate backbone.",
    )
    parser.add_argument("--results_csv", default=str(RESULTS_CSV), help="Path to results/results.csv.")
    parser.add_argument(
        "--output_json",
        default=str(BEST_BACKBONE_CONFIG_JSON),
        help="Path to save the selected best backbone JSON config.",
    )
    return parser.parse_args()


def _empty_mean_metrics() -> Dict[str, float | None]:
    return {f"mean_{metric}": None for metric in MEAN_METRIC_COLUMNS}


def _to_json_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _prepare_results(df: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    df = df.copy()
    if "split" in df.columns:
        df = df[df["split"].fillna("test") == "test"].copy()

    for metric in MEAN_METRIC_COLUMNS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
        else:
            df[metric] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    required_columns = {"dataset_name", "model_name", selection_metric}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"results.csv is missing required column(s): {sorted(missing_columns)}")

    df = df[df["model_name"].isin(BASELINE_MODELS)].copy()
    df = df.dropna(subset=[selection_metric])
    if df.empty:
        return df

    # If multiple runs exist for the same dataset/model, keep the best run for
    # the selected metric so repeated experiments do not overweight one dataset.
    idx = df.groupby(["dataset_name", "model_name"])[selection_metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def select_best_backbone(
    results_csv: Path = RESULTS_CSV,
    output_json: Path = BEST_BACKBONE_CONFIG_JSON,
    selection_metric: str = "f1",
) -> Tuple[Dict[str, object], List[str]]:
    warning_messages: List[str] = []

    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported metric '{selection_metric}'. Use one of: {SUPPORTED_SELECTION_METRICS}"
        )

    if not results_csv.exists():
        message = "No results/results.csv found. Please train and evaluate candidate backbone models first."
        print(message)
        return {}, [message]

    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        message = f"Could not read {results_csv}: {exc}"
        print(message)
        return {}, [message]

    try:
        candidate_df = _prepare_results(df, selection_metric)
    except Exception as exc:
        message = f"Could not prepare backbone results: {exc}"
        print(message)
        return {}, [message]

    if candidate_df.empty:
        message = (
            "No usable candidate backbone results found. "
            "Please train/evaluate dinov3_linear, siglip2_linear, or aimv2_linear first."
        )
        print(message)
        return {}, [message]

    present_models = sorted(candidate_df["model_name"].dropna().unique().tolist())
    missing_models = [model for model in BASELINE_MODELS if model not in present_models]
    if missing_models:
        warning = f"Missing candidate model(s) in results.csv: {', '.join(missing_models)}"
        warning_messages.append(warning)
        print(f"Warning: {warning}")

    available_datasets = sorted(candidate_df["dataset_name"].dropna().unique().tolist())
    candidate_summaries: List[Dict[str, object]] = []

    for model_name in BASELINE_MODELS:
        model_rows = candidate_df[candidate_df["model_name"] == model_name]
        if model_rows.empty:
            continue

        summary: Dict[str, object] = {
            "model_name": model_name,
            "available_datasets": sorted(model_rows["dataset_name"].dropna().unique().tolist()),
        }
        for metric in MEAN_METRIC_COLUMNS:
            summary[f"mean_{metric}"] = _to_json_float(model_rows[metric].mean())
        summary["selected_metric_score"] = summary[f"mean_{selection_metric}"]
        candidate_summaries.append(summary)

    valid_summaries = [
        summary
        for summary in candidate_summaries
        if summary.get("selected_metric_score") is not None
    ]
    if not valid_summaries:
        message = f"No candidate model has a valid '{selection_metric}' score."
        print(message)
        return {}, [message]

    best_summary = max(valid_summaries, key=lambda item: float(item["selected_metric_score"]))
    output: Dict[str, object] = {
        "selection_metric": selection_metric,
        "best_model_name": best_summary["model_name"],
        "selected_score": best_summary["selected_metric_score"],
        **_empty_mean_metrics(),
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "available_datasets": available_datasets,
        "candidate_models": list(BASELINE_MODELS),
        "warning_messages": warning_messages,
    }

    for metric in MEAN_METRIC_COLUMNS:
        output[f"mean_{metric}"] = best_summary.get(f"mean_{metric}")

    output["candidate_model_summaries"] = candidate_summaries

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Selected best backbone: {output['best_model_name']}")
    print(f"Selection metric: {selection_metric}")
    print(f"Selected score: {output['selected_score']}")
    print(f"Saved best backbone config to {output_json}")
    return output, warning_messages


def main() -> None:
    args = parse_args()
    select_best_backbone(
        results_csv=Path(args.results_csv),
        output_json=Path(args.output_json),
        selection_metric=args.metric,
    )


if __name__ == "__main__":
    main()
