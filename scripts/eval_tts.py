#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Template evaluator for Kikuyu TTS checkpoints.")
    parser.add_argument("--predictions-csv", required=True, help="CSV: id,reference_text,hyp_text,mos(optional)")
    parser.add_argument("--out-csv", default="artifacts/tts_eval_summary.csv")
    args = parser.parse_args()

    rows = []
    with Path(args.predictions_csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    non_empty = sum(1 for r in rows if (r.get("hyp_text", "") or "").strip())
    mos_vals = [float(r["mos"]) for r in rows if (r.get("mos", "") or "").strip()]
    avg_mos = sum(mos_vals) / len(mos_vals) if mos_vals else 0.0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["samples", "non_empty_hyp", "avg_mos"])
        writer.writeheader()
        writer.writerow({"samples": total, "non_empty_hyp": non_empty, "avg_mos": round(avg_mos, 4)})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
