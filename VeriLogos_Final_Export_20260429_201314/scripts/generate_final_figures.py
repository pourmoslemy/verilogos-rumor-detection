#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASET_ORDER = ["twitter15", "twitter16", "pheme"]
MODE_ORDER = ["tda_only", "text_only", "hybrid"]
MODE_COLORS = {"tda_only": "#1f77b4", "text_only": "#2ca02c", "hybrid": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final VeriLogos presentation charts")
    parser.add_argument("--export-root", required=True)
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def reorder(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    key = {(d, m): i for i, (d, m) in enumerate((d, m) for d in DATASET_ORDER for m in MODE_ORDER)}
    return sorted(rows, key=lambda r: key[(r['dataset'], r['mode'])])


def labels(rows: list[dict[str, str]]) -> list[str]:
    return [f"{r['dataset']}\n{r['mode']}" for r in rows]


def simple_bar(rows: list[dict[str, str]], metric: str, title: str, ylabel: str, output: Path) -> None:
    rows = reorder(rows)
    x = np.arange(len(rows))
    y = [to_float(r, metric) for r in rows]
    colors = [MODE_COLORS[r['mode']] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, y, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels(rows), rotation=0)
    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def class_recall_balance(rows: list[dict[str, str]], output: Path) -> None:
    rows = reorder(rows)
    fig, ax = plt.subplots(figsize=(8, 8))
    markers = {"twitter15": "o", "twitter16": "s", "pheme": "^"}
    for row in rows:
        x = to_float(row, 'fake_recall')
        y = to_float(row, 'real_recall')
        ax.scatter(x, y, s=120, marker=markers[row['dataset']], color=MODE_COLORS[row['mode']], edgecolor='black')
        ax.text(x + 0.01, y + 0.01, f"{row['dataset']}/{row['mode']}", fontsize=8)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Fake recall')
    ax.set_ylabel('Real recall')
    ax.set_title('Class Recall Balance')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def metric_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    rows = reorder(rows)
    metrics = ['accuracy', 'weighted_f1', 'macro_f1', 'fake_recall', 'real_recall', 'roc_auc', 'average_precision']
    data = np.array([[to_float(row, metric) for metric in metrics] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels(rows))
    ax.set_title('Metric Heatmap by Dataset/Mode')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def collapse_warning(rows: list[dict[str, str]], output: Path) -> None:
    rows = reorder(rows)
    x = np.arange(len(rows))
    fake = np.array([to_float(r, 'fake_recall') for r in rows])
    real = np.array([to_float(r, 'real_recall') for r in rows])
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, fake, width, label='Fake recall', color='#ff7f0e')
    ax.bar(x + width / 2, real, width, label='Real recall', color='#17becf')
    for i, row in enumerate(rows):
        if row['severe_collapse'] == 'True':
            ax.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.08)
            ax.text(i, 1.02, 'collapse', ha='center', va='bottom', color='red', fontsize=8, rotation=90)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels(rows))
    ax.set_ylabel('Recall')
    ax.set_title('Collapse Warning Chart')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def best_model_comparison(rows: list[dict[str, str]], output: Path) -> None:
    top = sorted(rows, key=lambda r: float(r['weighted_f1']), reverse=True)[:5]
    labels_local = [f"{r['dataset']} / {r['mode']}" for r in top]
    vals = [float(r['weighted_f1']) for r in top]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels_local[::-1], vals[::-1], color='#4c78a8')
    ax.set_xlabel('Weighted F1')
    ax.set_xlim(0, 1.0)
    ax.set_title('Top 5 Models by Weighted F1')
    for bar, value in zip(bars, vals[::-1]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va='center')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def per_dataset_mode_comparison(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    metric = 'weighted_f1'
    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = [r for r in rows if r['dataset'] == dataset]
        subset = sorted(subset, key=lambda r: MODE_ORDER.index(r['mode']))
        vals = [float(r[metric]) for r in subset]
        colors = [MODE_COLORS[r['mode']] for r in subset]
        ax.bar([r['mode'] for r in subset], vals, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_title(dataset)
        ax.set_ylabel('Weighted F1')
        for idx, value in enumerate(vals):
            ax.text(idx, value + 0.02, f"{value:.3f}", ha='center', fontsize=8)
    fig.suptitle('Per-Dataset Mode Comparison')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    export_root = Path(args.export_root)
    charts_dir = export_root / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(export_root / 'FINAL_METRICS_SUMMARY.csv')

    simple_bar(rows, 'accuracy', 'Accuracy by Dataset/Mode', 'Accuracy', charts_dir / 'accuracy_by_dataset_mode.png')
    simple_bar(rows, 'weighted_f1', 'Weighted F1 by Dataset/Mode', 'Weighted F1', charts_dir / 'weighted_f1_by_dataset_mode.png')
    simple_bar(rows, 'macro_f1', 'Macro F1 by Dataset/Mode', 'Macro F1', charts_dir / 'macro_f1_by_dataset_mode.png')
    simple_bar(rows, 'fake_recall', 'Fake Recall by Dataset/Mode', 'Fake Recall', charts_dir / 'fake_recall_by_dataset_mode.png')
    simple_bar(rows, 'real_recall', 'Real Recall by Dataset/Mode', 'Real Recall', charts_dir / 'real_recall_by_dataset_mode.png')
    class_recall_balance(rows, charts_dir / 'class_recall_balance.png')
    metric_heatmap(rows, charts_dir / 'metric_heatmap.png')
    collapse_warning(rows, charts_dir / 'collapse_warning_chart.png')
    best_model_comparison(rows, charts_dir / 'best_model_comparison.png')
    per_dataset_mode_comparison(rows, charts_dir / 'per_dataset_mode_comparison.png')

    print('\n'.join(str(p) for p in sorted(charts_dir.glob('*.png'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
