#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_row(row: dict[str, str]) -> float:
    success_rate = parse_float(row.get("synthesis_success_rate"), 0.0)
    clipping_rate = parse_float(row.get("clipping_rate"), 1.0)
    mos_lite = parse_float(row.get("mos_lite"), 0.0)
    wer_proxy = parse_float(row.get("wer_proxy"), 1.0)
    return (
        (success_rate * 0.35)
        + ((1.0 - clipping_rate) * 0.15)
        + ((mos_lite / 5.0) * 0.35)
        + ((1.0 - wer_proxy) * 0.15)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best checkpoint using objective + listening criteria."
    )
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="CSV with columns: checkpoint,synthesis_success_rate,clipping_rate,mos_lite,wer_proxy",
    )
    parser.add_argument("--out-json", default="artifacts/best_checkpoint_selection.json")
    parser.add_argument("--out-csv", default="artifacts/tts_eval_summary.csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("No rows in metrics CSV.")

    for row in rows:
        row["combined_score"] = round(score_row(row), 6)

    rows_sorted = sorted(rows, key=lambda r: float(r["combined_score"]), reverse=True)
    best = rows_sorted[0]

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_checkpoint": best.get("checkpoint", ""),
                "combined_score": best["combined_score"],
                "ranking": rows_sorted,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Best checkpoint: {best.get('checkpoint', '')} score={best['combined_score']}")


if __name__ == "__main__":
    main()
