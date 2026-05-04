# Final Presentation Notes

## Slide 1 — Objective
- Final VeriLogos training campaign across three datasets and three model modes.
- Goal: compare TDA-only, text-only, and hybrid variants under the corrected REAL=0 / FAKE=1 pipeline.

## Slide 2 — Execution Outcome
- All 9 runs completed successfully.
- Full artifact set collected: logs, metrics JSON, classification reports, confusion matrices, ROC/PR curves, training histories.

## Slide 3 — Top-Line Performance
- Best overall weighted F1: `0.8239` from `twitter16 / hybrid`.
- Show `best_model_comparison.png` and `weighted_f1_by_dataset_mode.png`.

## Slide 4 — Per-Dataset Winners
- twitter15 winner: `text_only`
- twitter16 winner: `hybrid`
- pheme winner: `tda_only`
- Show `per_dataset_mode_comparison.png`.

## Slide 5 — Hybrid Benefit Check
- twitter15: hybrid delta vs best single = `-0.0136` weighted F1.
- twitter16: hybrid delta vs best single = `+0.0348` weighted F1.
- pheme: hybrid delta vs best single = `-0.1259` weighted F1.

## Slide 6 — Scientific Risk
- Show `collapse_warning_chart.png` and `class_recall_balance.png`.
- Explain that `pheme / hybrid` predicts only one class on the final test set, making its apparent completion operationally valid but scientifically unsafe.

## Slide 7 — Final Message
- Operational readiness: yes.
- Scientific readiness: conditional.
- Next experiments should focus on class-collapse remediation, calibration, reweighting, and reruns on the unstable configurations.
