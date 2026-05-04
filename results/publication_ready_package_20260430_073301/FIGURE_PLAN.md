# Figure Plan

All figures are regenerated locally from frozen final-run metrics and decision artifacts only. No training or reruns were performed.

Failed runs remain visible in overview figures only as explicitly color-coded failed baselines; they are not eligible for scientific ranking claims. PHEME `tda_only` is shown as weak/diagnostic, not as a strong success.

## Overall Model Comparison
- File: `/home/ubuntu/colab_deployment/publication_figures/overall_model_comparison_bar_chart.png`
- Caption: Weighted F1 across all nine runs, colored by reliability class.
- Purpose: Show the global performance spread while keeping failed baselines visible.
- Placement: Main paper
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z

## Per-Dataset Weighted F1 Comparison
- File: `/home/ubuntu/colab_deployment/publication_figures/per_dataset_weighted_f1_comparison.png`
- Caption: Mode-wise weighted F1 within each dataset.
- Purpose: Support dataset-level winner claims.
- Placement: Main paper
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z

## Collapse Warning Chart
- File: `/home/ubuntu/colab_deployment/publication_figures/collapse_warning_chart.png`
- Caption: Per-class recall for each run, with collapsed runs explicitly marked.
- Purpose: Prevent aggregate metrics from hiding one-class failure.
- Placement: Main paper
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z, /home/ubuntu/colab_deployment/results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json

## Metric Heatmap
- File: `/home/ubuntu/colab_deployment/publication_figures/metric_heatmap.png`
- Caption: Heatmap over accuracy, F1, and per-class recall for all runs.
- Purpose: Compact appendix summary of performance and imbalance.
- Placement: Appendix
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z, /home/ubuntu/colab_deployment/results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json

## Reliability Matrix
- File: `/home/ubuntu/colab_deployment/publication_figures/reliability_matrix.png`
- Caption: Dataset-by-mode matrix labeling each run as valid, weak, or failed.
- Purpose: Fast audit view of what is scientifically claimable.
- Placement: Main paper
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z, /home/ubuntu/colab_deployment/results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json

## Reliable Confusion-Matrix Summary
- File: `/home/ubuntu/colab_deployment/publication_figures/reliable_confusion_matrix_summary.png`
- Caption: Confusion matrices for non-collapsed runs only.
- Purpose: Appendix evidence that recommended models predict both classes.
- Placement: Appendix
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z

## Prediction Distribution Visualization
- File: `/home/ubuntu/colab_deployment/publication_figures/prediction_distribution_visualization.png`
- Caption: Stacked predicted-class counts highlighting one-class collapse.
- Purpose: Appendix support for collapse diagnostics.
- Placement: Appendix
- Sources: /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z, /home/ubuntu/colab_deployment/results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json
