#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DATASET_ORDER = ["twitter15", "twitter16", "pheme"]
MODE_ORDER = ["tda_only", "text_only", "hybrid"]
RUN_TS = "20260429T142746Z"


@dataclass
class ProgressEntry:
    dataset: str
    mode: str
    start: str | None = None
    end: str | None = None
    log_path: str | None = None

    @property
    def duration_seconds(self) -> int | None:
        if not self.start or not self.end:
            return None
        start = datetime.fromisoformat(self.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.end.replace("Z", "+00:00"))
        return int((end - start).total_seconds())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final VeriLogos metrics into structured summaries.")
    parser.add_argument("--export-root", required=True, help="Path to the local VeriLogos export directory")
    return parser.parse_args()


def parse_results_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_name, metrics = next(iter(payload.items()))
    return {
        "model_name": model_name,
        "accuracy": float(metrics["accuracy"]),
        "weighted_precision": float(metrics["precision"]),
        "weighted_recall": float(metrics["recall"]),
        "weighted_f1": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
    }


def parse_classification_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip("\n") for line in text.splitlines()]
    metrics: dict[str, Any] = {}
    confusion_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] == "Fake" and len(parts) >= 5:
            metrics.update({
                "fake_precision": float(parts[1]),
                "fake_recall": float(parts[2]),
                "fake_f1": float(parts[3]),
                "fake_support": int(parts[4]),
            })
        elif parts[0] == "Real" and len(parts) >= 5 and not confusion_started:
            metrics.update({
                "real_precision": float(parts[1]),
                "real_recall": float(parts[2]),
                "real_f1": float(parts[3]),
                "real_support": int(parts[4]),
            })
        elif parts[0] == "accuracy" and len(parts) >= 3:
            metrics["report_accuracy"] = float(parts[1])
            metrics["total_support"] = int(parts[2])
        elif parts[0] == "macro" and len(parts) >= 6:
            metrics.update({
                "macro_precision": float(parts[2]),
                "macro_recall": float(parts[3]),
                "macro_f1": float(parts[4]),
            })
        elif parts[0] == "weighted" and len(parts) >= 6:
            metrics.update({
                "report_weighted_precision": float(parts[2]),
                "report_weighted_recall": float(parts[3]),
                "report_weighted_f1": float(parts[4]),
            })
        elif stripped.startswith("Confusion Matrix"):
            confusion_started = True
        elif confusion_started and stripped.startswith("Actual Fake"):
            nums = re.findall(r"\d+", stripped)
            if len(nums) >= 2:
                metrics["cm_actual_fake_pred_fake"] = int(nums[0])
                metrics["cm_actual_fake_pred_real"] = int(nums[1])
        elif confusion_started and stripped.startswith("Real"):
            nums = re.findall(r"\d+", stripped)
            if len(nums) >= 2:
                metrics["cm_actual_real_pred_fake"] = int(nums[0])
                metrics["cm_actual_real_pred_real"] = int(nums[1])
    return metrics


def parse_progress(path: Path) -> tuple[dict[tuple[str, str], ProgressEntry], dict[str, Any]]:
    runs: dict[tuple[str, str], ProgressEntry] = {}
    overall: dict[str, Any] = {"started": None, "ended": None, "status": "UNKNOWN"}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("STARTED "):
            overall["started"] = line.split()[1]
            overall["status"] = "RUNNING"
            continue
        if line.startswith("ALL_COMPLETED "):
            overall["ended"] = line.split()[1]
            overall["status"] = "COMPLETED"
            continue
        m = re.match(r"^(RUNNING|COMPLETED|FAILED)\s+(\d+)\s+(\S+)\s+(\S+)\s+([0-9T:\-]+Z)(?:\s+log=(\S+))?", line)
        if not m:
            continue
        state, _idx, dataset, mode, ts, log_path = m.groups()
        entry = runs.setdefault((dataset, mode), ProgressEntry(dataset=dataset, mode=mode))
        if state == "RUNNING":
            entry.start = ts
        elif state in {"COMPLETED", "FAILED"}:
            entry.end = ts
            entry.log_path = log_path
    return runs, overall


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for key, _label in columns:
            value = row.get(key, "")
            vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def main() -> int:
    args = parse_args()
    export_root = Path(args.export_root)
    artifacts_root = export_root / "remote_artifacts"
    runs_root = artifacts_root / "final_runs" / RUN_TS
    progress_path = artifacts_root / "status" / "progress.txt"

    progress_map, overall_progress = parse_progress(progress_path)
    rows: list[dict[str, Any]] = []

    for dataset in DATASET_ORDER:
        for mode in MODE_ORDER:
            run_dir = runs_root / dataset / mode
            result_path = run_dir / "results.json"
            report_path = run_dir / "classification_report.txt"
            result_metrics = parse_results_json(result_path)
            report_metrics = parse_classification_report(report_path)
            progress = progress_map.get((dataset, mode), ProgressEntry(dataset=dataset, mode=mode))
            cm_fake_total = report_metrics["cm_actual_fake_pred_fake"] + report_metrics["cm_actual_fake_pred_real"]
            cm_real_total = report_metrics["cm_actual_real_pred_fake"] + report_metrics["cm_actual_real_pred_real"]
            predicted_fake_total = report_metrics["cm_actual_fake_pred_fake"] + report_metrics["cm_actual_real_pred_fake"]
            predicted_real_total = report_metrics["cm_actual_fake_pred_real"] + report_metrics["cm_actual_real_pred_real"]
            one_class_predicted = predicted_fake_total == 0 or predicted_real_total == 0
            severe_collapse = one_class_predicted or report_metrics["fake_recall"] == 0.0 or report_metrics["real_recall"] == 0.0
            recall_balance_gap = abs(report_metrics["fake_recall"] - report_metrics["real_recall"])
            row = {
                "dataset": dataset,
                "mode": mode,
                **result_metrics,
                **report_metrics,
                "cm_fake_total": cm_fake_total,
                "cm_real_total": cm_real_total,
                "predicted_fake_total": predicted_fake_total,
                "predicted_real_total": predicted_real_total,
                "one_class_predicted": one_class_predicted,
                "severe_collapse": severe_collapse,
                "recall_balance_gap": round(recall_balance_gap, 4),
                "scientific_reliability": "NO" if severe_collapse else ("CAUTION" if recall_balance_gap >= 0.5 else "YES"),
                "run_start_utc": progress.start,
                "run_end_utc": progress.end,
                "duration_seconds": progress.duration_seconds,
                "results_json": str(result_path.relative_to(export_root)),
                "classification_report": str(report_path.relative_to(export_root)),
            }
            rows.append(row)

    csv_path = export_root / "FINAL_METRICS_SUMMARY.csv"
    json_path = export_root / "FINAL_METRICS_SUMMARY.json"
    md_path = export_root / "FINAL_METRICS_SUMMARY.md"

    csv_columns = [
        "dataset", "mode", "accuracy", "weighted_f1", "macro_f1", "fake_recall", "real_recall",
        "roc_auc", "average_precision", "one_class_predicted", "severe_collapse", "scientific_reliability",
        "duration_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "run_timestamp": RUN_TS,
        "overall_progress": overall_progress,
        "rows": rows,
        "best_overall_by_weighted_f1": max(rows, key=lambda r: r["weighted_f1"]),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    md_rows = []
    for row in rows:
        md_rows.append({
            "dataset": row["dataset"],
            "mode": row["mode"],
            "accuracy": f"{row['accuracy']:.4f}",
            "weighted_f1": f"{row['weighted_f1']:.4f}",
            "macro_f1": f"{row['macro_f1']:.4f}",
            "fake_recall": f"{row['fake_recall']:.4f}",
            "real_recall": f"{row['real_recall']:.4f}",
            "collapse": "YES" if row["severe_collapse"] else "NO",
        })
    md_columns = [
        ("dataset", "Dataset"), ("mode", "Mode"), ("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1"),
        ("macro_f1", "Macro F1"), ("fake_recall", "Fake Recall"), ("real_recall", "Real Recall"), ("collapse", "Collapse"),
    ]
    best = summary_payload["best_overall_by_weighted_f1"]
    md_text = "\n".join([
        "# Final Metrics Summary",
        "",
        f"- Export root: `{export_root}`",
        f"- Completed run status: `{overall_progress['status']}`",
        f"- Best overall by weighted F1: `{best['dataset']} / {best['mode']}` (`{best['weighted_f1']:.4f}`)",
        f"- Severe collapse runs detected: `{sum(1 for row in rows if row['severe_collapse'])}`",
        "",
        markdown_table(md_rows, md_columns),
        "",
    ])
    md_path.write_text(md_text, encoding="utf-8")

    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "rows": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
