# VeriLogos Final Technical Report

## 1. Scope
- This report consolidates the nine completed final training runs across `twitter15`, `twitter16`, and `pheme` for `tda_only`, `text_only`, and `hybrid` modes.
- Source artifacts were copied from the remote Linux server without modifying any original training outputs.

## 2. Operational Outcome
- All 9 planned runs completed successfully.
- Required plots, `results.json`, classification reports, logs, and progress traces were collected locally.

## 3. Consolidated Metric Table
| Dataset | Mode | Accuracy | Weighted F1 | Macro F1 | Fake Recall | Real Recall | Collapse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| twitter15 | tda_only | 0.5000 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | YES |
| twitter15 | text_only | 0.8230 | 0.8229 | 0.8229 | 0.7965 | 0.8496 | NO |
| twitter15 | hybrid | 0.8097 | 0.8093 | 0.8093 | 0.7611 | 0.8584 | NO |
| twitter16 | tda_only | 0.4960 | 0.4153 | 0.4171 | 0.1270 | 0.8710 | NO |
| twitter16 | text_only | 0.7920 | 0.7892 | 0.7890 | 0.9048 | 0.6774 | NO |
| twitter16 | hybrid | 0.8240 | 0.8239 | 0.8239 | 0.8413 | 0.8065 | NO |
| pheme | tda_only | 0.5263 | 0.4447 | 0.4389 | 0.8974 | 0.1351 | NO |
| pheme | text_only | 0.4868 | 0.3188 | 0.3274 | 0.0000 | 1.0000 | YES |
| pheme | hybrid | 0.4868 | 0.3188 | 0.3274 | 0.0000 | 1.0000 | YES |

## 4. Best Models
- Best overall by weighted F1: `twitter16 / hybrid` with weighted F1 `0.8239` and accuracy `0.8240`.
- Best per dataset: twitter15=`text_only`, twitter16=`hybrid`, pheme=`tda_only`.

## 5. Hybrid vs Single-Mode Analysis
- `twitter15`: hybrid weighted F1 `0.8093` vs best single-mode `0.8229` (delta `-0.0136`) -> does not improve.
- `twitter16`: hybrid weighted F1 `0.8239` vs best single-mode `0.7892` (delta `+0.0348`) -> improves.
- `pheme`: hybrid weighted F1 `0.3188` vs best single-mode `0.4447` (delta `-0.1259`) -> does not improve.

## 6. Collapse and Reliability Analysis
- Severe collapse detected in `3` run(s): twitter15/tda_only, pheme/text_only, pheme/hybrid.
- `pheme / hybrid` is not scientifically reliable for final claims due to severe one-class collapse.
- `pheme / text_only` also collapses to one predicted class and should not be treated as reliable.
- `twitter15 / tda_only` collapses to one predicted class and needs rebalancing or threshold investigation.
- `pheme / hybrid` is not scientifically reliable for model-selection claims until the collapse mechanism is fixed and rerun.

## 7. Reported Figures
- `charts/accuracy_by_dataset_mode.png`
- `charts/weighted_f1_by_dataset_mode.png`
- `charts/macro_f1_by_dataset_mode.png`
- `charts/fake_recall_by_dataset_mode.png`
- `charts/real_recall_by_dataset_mode.png`
- `charts/class_recall_balance.png`
- `charts/metric_heatmap.png`
- `charts/collapse_warning_chart.png`
- `charts/best_model_comparison.png`
- `charts/per_dataset_mode_comparison.png`

## 8. Final Conclusion
- Operational readiness: complete. All scheduled runs finished and produced artifacts.
- Scientific readiness: conditional. `pheme / hybrid` exhibits severe one-class collapse, and other collapse cases exist, so final scientific claims must be limited to the stable runs until remediation experiments are completed.
