#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

DATASET_ORDER = ["twitter15", "twitter16", "pheme"]
MODE_ORDER = ["tda_only", "text_only", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build final VeriLogos technical reports')
    parser.add_argument('--export-root', required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def fmt(x: float) -> str:
    return f"{x:.4f}"


def make_table(rows: list[dict[str, str]], keys: list[str], headers: list[str]) -> str:
    header = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(str(row[k]) for k in keys) + ' |')
    return '\n'.join([header, sep] + body)


def main() -> int:
    args = parse_args()
    export_root = Path(args.export_root)
    rows = read_csv(export_root / 'FINAL_METRICS_SUMMARY.csv')
    summary = json.loads((export_root / 'FINAL_METRICS_SUMMARY.json').read_text(encoding='utf-8'))
    progress_lines = (export_root / 'remote_artifacts/status/progress.txt').read_text(encoding='utf-8').strip().splitlines()

    keyed = {(r['dataset'], r['mode']): r for r in rows}
    best_overall = max(rows, key=lambda r: float(r['weighted_f1']))
    best_per_dataset = {}
    for dataset in DATASET_ORDER:
        subset = [r for r in rows if r['dataset'] == dataset]
        best_per_dataset[dataset] = max(subset, key=lambda r: float(r['weighted_f1']))

    hybrid_analysis = []
    for dataset in DATASET_ORDER:
        hybrid = float(keyed[(dataset, 'hybrid')]['weighted_f1'])
        singles = [float(keyed[(dataset, mode)]['weighted_f1']) for mode in ['tda_only', 'text_only']]
        best_single = max(singles)
        hybrid_analysis.append({
            'dataset': dataset,
            'hybrid_weighted_f1': hybrid,
            'best_single_weighted_f1': best_single,
            'hybrid_delta_vs_best_single': hybrid - best_single,
            'hybrid_improves': hybrid > best_single,
        })

    collapse_rows = [r for r in rows if r['severe_collapse'] == 'True']
    warning_lines = [
        '- `pheme / hybrid` is not scientifically reliable for final claims due to severe one-class collapse.',
    ]
    if any(r['dataset'] == 'pheme' and r['mode'] == 'text_only' for r in collapse_rows):
        warning_lines.append('- `pheme / text_only` also collapses to one predicted class and should not be treated as reliable.')
    if any(r['dataset'] == 'twitter15' and r['mode'] == 'tda_only' for r in collapse_rows):
        warning_lines.append('- `twitter15 / tda_only` collapses to one predicted class and needs rebalancing or threshold investigation.')

    exec_status = '\n'.join([
        '# Execution Status',
        '',
        '- Operational readiness: all 9 training runs completed successfully.',
        '- Detached remote session finished cleanly and wrote all expected logs.',
        '- Progress trace:',
        '',
        '```text',
        *progress_lines,
        '```',
        '',
        '- Copied artifacts include 9 `results.json`, 9 `classification_report.txt`, 9 run logs, confusion matrices, ROC/PR curves, training history images, and metrics comparison images.',
        '- Runtime note: training executed in the audited server environment where CUDA was unavailable during readiness validation, so observed runtimes reflect CPU execution.',
        '',
    ])
    (export_root / 'EXECUTION_STATUS.md').write_text(exec_status, encoding='utf-8')

    summary_rows = []
    for r in rows:
        summary_rows.append({
            'dataset': r['dataset'],
            'mode': r['mode'],
            'accuracy': fmt(float(r['accuracy'])),
            'weighted_f1': fmt(float(r['weighted_f1'])),
            'macro_f1': fmt(float(r['macro_f1'])),
            'fake_recall': fmt(float(r['fake_recall'])),
            'real_recall': fmt(float(r['real_recall'])),
            'collapse': 'YES' if r['severe_collapse'] == 'True' else 'NO',
        })

    technical = '\n'.join([
        '# VeriLogos Final Technical Report',
        '',
        '## 1. Scope',
        '- This report consolidates the nine completed final training runs across `twitter15`, `twitter16`, and `pheme` for `tda_only`, `text_only`, and `hybrid` modes.',
        '- Source artifacts were copied from the remote Linux server without modifying any original training outputs.',
        '',
        '## 2. Operational Outcome',
        '- All 9 planned runs completed successfully.',
        '- Required plots, `results.json`, classification reports, logs, and progress traces were collected locally.',
        '',
        '## 3. Consolidated Metric Table',
        make_table(summary_rows, ['dataset', 'mode', 'accuracy', 'weighted_f1', 'macro_f1', 'fake_recall', 'real_recall', 'collapse'], ['Dataset', 'Mode', 'Accuracy', 'Weighted F1', 'Macro F1', 'Fake Recall', 'Real Recall', 'Collapse']),
        '',
        '## 4. Best Models',
        f"- Best overall by weighted F1: `{best_overall['dataset']} / {best_overall['mode']}` with weighted F1 `{fmt(float(best_overall['weighted_f1']))}` and accuracy `{fmt(float(best_overall['accuracy']))}`.",
        f"- Best per dataset: twitter15=`{best_per_dataset['twitter15']['mode']}`, twitter16=`{best_per_dataset['twitter16']['mode']}`, pheme=`{best_per_dataset['pheme']['mode']}`.",
        '',
        '## 5. Hybrid vs Single-Mode Analysis',
        *[
            f"- `{item['dataset']}`: hybrid weighted F1 `{fmt(item['hybrid_weighted_f1'])}` vs best single-mode `{fmt(item['best_single_weighted_f1'])}` (delta `{item['hybrid_delta_vs_best_single']:+.4f}`) -> {'improves' if item['hybrid_improves'] else 'does not improve'}."
            for item in hybrid_analysis
        ],
        '',
        '## 6. Collapse and Reliability Analysis',
        f"- Severe collapse detected in `{len(collapse_rows)}` run(s): {', '.join(f'{r['dataset']}/{r['mode']}' for r in collapse_rows)}.",
        *warning_lines,
        '- `pheme / hybrid` is not scientifically reliable for model-selection claims until the collapse mechanism is fixed and rerun.',
        '',
        '## 7. Reported Figures',
        '- `charts/accuracy_by_dataset_mode.png`',
        '- `charts/weighted_f1_by_dataset_mode.png`',
        '- `charts/macro_f1_by_dataset_mode.png`',
        '- `charts/fake_recall_by_dataset_mode.png`',
        '- `charts/real_recall_by_dataset_mode.png`',
        '- `charts/class_recall_balance.png`',
        '- `charts/metric_heatmap.png`',
        '- `charts/collapse_warning_chart.png`',
        '- `charts/best_model_comparison.png`',
        '- `charts/per_dataset_mode_comparison.png`',
        '',
        '## 8. Final Conclusion',
        '- Operational readiness: complete. All scheduled runs finished and produced artifacts.',
        '- Scientific readiness: conditional. `pheme / hybrid` exhibits severe one-class collapse, and other collapse cases exist, so final scientific claims must be limited to the stable runs until remediation experiments are completed.',
        '',
    ])
    (export_root / 'FINAL_TECHNICAL_REPORT.md').write_text(technical, encoding='utf-8')

    exec_summary = '\n'.join([
        '# Final Findings Executive Summary',
        '',
        '- Operational readiness: all 9 final runs completed successfully and all required artifacts were collected.',
        f"- Best overall model: `{best_overall['dataset']} / {best_overall['mode']}` with weighted F1 `{fmt(float(best_overall['weighted_f1']))}`.",
        '- Best per dataset:',
        f"  - twitter15: `{best_per_dataset['twitter15']['mode']}`",
        f"  - twitter16: `{best_per_dataset['twitter16']['mode']}`",
        f"  - pheme: `{best_per_dataset['pheme']['mode']}`",
        '- Hybrid effectiveness:',
        *[
            f"  - {item['dataset']}: {'improves' if item['hybrid_improves'] else 'does not improve'} over the best single-mode by `{item['hybrid_delta_vs_best_single']:+.4f}` weighted F1."
            for item in hybrid_analysis
        ],
        '- Critical scientific warning:',
        '  - `pheme / hybrid` collapses to one predicted class and is not reliable for final scientific conclusions.',
        '  - Additional collapse cases exist in `pheme / text_only` and `twitter15 / tda_only`.',
        '- Recommendation: present operational completion as a success, but present scientific readiness as conditional pending collapse remediation and reruns.',
        '',
    ])
    (export_root / 'FINAL_FINDINGS_EXECUTIVE_SUMMARY.md').write_text(exec_summary, encoding='utf-8')

    presentation = '\n'.join([
        '# Final Presentation Notes',
        '',
        '## Slide 1 — Objective',
        '- Final VeriLogos training campaign across three datasets and three model modes.',
        '- Goal: compare TDA-only, text-only, and hybrid variants under the corrected REAL=0 / FAKE=1 pipeline.',
        '',
        '## Slide 2 — Execution Outcome',
        '- All 9 runs completed successfully.',
        '- Full artifact set collected: logs, metrics JSON, classification reports, confusion matrices, ROC/PR curves, training histories.',
        '',
        '## Slide 3 — Top-Line Performance',
        f"- Best overall weighted F1: `{fmt(float(best_overall['weighted_f1']))}` from `{best_overall['dataset']} / {best_overall['mode']}`.",
        '- Show `best_model_comparison.png` and `weighted_f1_by_dataset_mode.png`.',
        '',
        '## Slide 4 — Per-Dataset Winners',
        f"- twitter15 winner: `{best_per_dataset['twitter15']['mode']}`",
        f"- twitter16 winner: `{best_per_dataset['twitter16']['mode']}`",
        f"- pheme winner: `{best_per_dataset['pheme']['mode']}`",
        '- Show `per_dataset_mode_comparison.png`.',
        '',
        '## Slide 5 — Hybrid Benefit Check',
        *[
            f"- {item['dataset']}: hybrid delta vs best single = `{item['hybrid_delta_vs_best_single']:+.4f}` weighted F1."
            for item in hybrid_analysis
        ],
        '',
        '## Slide 6 — Scientific Risk',
        '- Show `collapse_warning_chart.png` and `class_recall_balance.png`.',
        '- Explain that `pheme / hybrid` predicts only one class on the final test set, making its apparent completion operationally valid but scientifically unsafe.',
        '',
        '## Slide 7 — Final Message',
        '- Operational readiness: yes.',
        '- Scientific readiness: conditional.',
        '- Next experiments should focus on class-collapse remediation, calibration, reweighting, and reruns on the unstable configurations.',
        '',
    ])
    (export_root / 'FINAL_PRESENTATION_NOTES.md').write_text(presentation, encoding='utf-8')

    print(json.dumps({
        'execution_status': str(export_root / 'EXECUTION_STATUS.md'),
        'technical_report': str(export_root / 'FINAL_TECHNICAL_REPORT.md'),
        'executive_summary': str(export_root / 'FINAL_FINDINGS_EXECUTIVE_SUMMARY.md'),
        'presentation_notes': str(export_root / 'FINAL_PRESENTATION_NOTES.md'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
