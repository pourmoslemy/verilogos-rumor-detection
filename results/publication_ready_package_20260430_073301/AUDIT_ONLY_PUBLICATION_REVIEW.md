# Audit-Only Publication Review

- Timestamp (UTC): 2026-04-30T07:14:32.798676+00:00
- Status: READY_FOR_MANUSCRIPT_ASSEMBLY
- Training/reruns/tuning performed: NO
- Scope: reporting-only audit against frozen artifacts

## Executive Summary
- Frozen metrics and winner claims match the audited final-run artifacts and the documented `pheme/text_only` closure artifact.
- Failed baselines `twitter15/tda_only`, `pheme/text_only`, and `pheme/hybrid` are explicitly excluded from scientific ranking claims.
- `pheme/tda_only` remains clearly marked weak/diagnostic and is not presented as a strong PHEME success.
- Reporting-only fixes were applied to strengthen label-semantics documentation and align LaTeX/figure caveats with Markdown outputs.
- The initial audit parser discrepancy on `pheme/hybrid` macro-F1 was resolved by deferring to the stored frozen `macro avg` line (`0.3274`) in the original classification report.

## Files Reviewed
- `/home/ubuntu/colab_deployment/PUBLICATION_READY_EXECUTIVE_SUMMARY.md`
- `/home/ubuntu/colab_deployment/publication_tables.md`
- `/home/ubuntu/colab_deployment/publication_tables.csv`
- `/home/ubuntu/colab_deployment/publication_tables.tex`
- `/home/ubuntu/colab_deployment/FIGURE_PLAN.md`
- `/home/ubuntu/colab_deployment/figures_manifest.json`
- `/home/ubuntu/colab_deployment/FINAL_RECOMMENDED_MODELS.md`
- `/home/ubuntu/colab_deployment/FINAL_RECOMMENDED_MODELS.json`
- `/home/ubuntu/colab_deployment/FINAL_RECOMMENDED_CONFIGS.md`
- `/home/ubuntu/colab_deployment/FINAL_RECOMMENDED_CONFIGS.json`
- `/home/ubuntu/colab_deployment/FINAL_TUNING_RECOMMENDATION.md`
- `/home/ubuntu/colab_deployment/PAPER_READY_CLAIMS.md`
- `/home/ubuntu/colab_deployment/SUBMISSION_CHECKLIST.md`
- `/home/ubuntu/colab_deployment/publication_package_audit.json`
- `/home/ubuntu/colab_deployment/publication_figures/collapse_warning_chart.png`
- `/home/ubuntu/colab_deployment/publication_figures/metric_heatmap.png`
- `/home/ubuntu/colab_deployment/publication_figures/overall_model_comparison_bar_chart.png`
- `/home/ubuntu/colab_deployment/publication_figures/per_dataset_weighted_f1_comparison.png`
- `/home/ubuntu/colab_deployment/publication_figures/prediction_distribution_visualization.png`
- `/home/ubuntu/colab_deployment/publication_figures/reliability_matrix.png`
- `/home/ubuntu/colab_deployment/publication_figures/reliable_confusion_matrix_summary.png`

## Verification Results
- Metrics cross-check source: `results/final_runs/20260429T142746Z/*/*/results.json` plus `results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json` for `pheme/text_only`.
- Binary label semantics now documented: `REAL = 0`, `FAKE = 1`; unsupported `unverified`/`unconfirmed` labels are outside the reported binary claim set and are not silently remapped.
- Standalone checkpoint absence remains documented in model/config/package claims.
- CSV intentionally preserves raw precision; Markdown and LaTeX tables intentionally round to four decimals.

## Fixes Made
- `/home/ubuntu/colab_deployment/PUBLICATION_READY_EXECUTIVE_SUMMARY.md`: Added explicit binary label semantics and unsupported-label caveat.
- `/home/ubuntu/colab_deployment/publication_tables.md`: Added table-level label semantics and precision policy note.
- `/home/ubuntu/colab_deployment/publication_tables.tex`: Added LaTeX label-semantics note and explicit exclusion flags for failed runs.
- `/home/ubuntu/colab_deployment/FIGURE_PLAN.md`: Added figure-interpretation caveat for failed and weak runs.
- `/home/ubuntu/colab_deployment/PAPER_READY_CLAIMS.md`: Added explicit label-semantics wording to paper-ready claims.
- `/home/ubuntu/colab_deployment/SUBMISSION_CHECKLIST.md`: Added checklist item for binary label-semantics documentation.
- `/home/ubuntu/colab_deployment/figures_manifest.json`: Added global figure-manifest caveats for label semantics and failed-run rendering.

## Backups Created
- `/home/ubuntu/colab_deployment/PUBLICATION_READY_EXECUTIVE_SUMMARY.md.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/publication_tables.md.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/publication_tables.tex.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/FIGURE_PLAN.md.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/PAPER_READY_CLAIMS.md.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/SUBMISSION_CHECKLIST.md.auditbak_20260430T071432Z`
- `/home/ubuntu/colab_deployment/figures_manifest.json.auditbak_20260430T071432Z`

## Open Issues
- None

## Warnings
- PHEME conclusions remain weak because only `pheme/tda_only` is non-collapsed and it shows severe recall imbalance.
- Stored final-run classification reports predate later label-order cleanup, so publication claims should continue to rely on audited aggregate metrics, confusion matrices, and collapse diagnostics.

## Final Decision
- READY_FOR_MANUSCRIPT_ASSEMBLY
