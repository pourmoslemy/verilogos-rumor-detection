# Final Recommended Models

Collapsed runs are ineligible for final recommended weights. Recommendations follow scientific reliability first, then weighted F1, then macro F1, then recall balance.

## Twitter15
- Primary model: `twitter15/text_only`
- Weighted F1: 0.8229
- Macro F1: 0.8229
- Reliability: VALID
- Artifact path: `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter15/text_only`
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter15_text_only.yaml`
- Checkpoint/weights path: not preserved as a standalone weight artifact in the final run directory
- Publication-safe: YES
- Deployment-safe: YES
- Backup model: `twitter15/hybrid`

## Twitter16
- Primary model: `twitter16/hybrid`
- Weighted F1: 0.8239
- Macro F1: 0.8239
- Reliability: VALID
- Artifact path: `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter16/hybrid`
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_hybrid.yaml`
- Checkpoint/weights path: not preserved as a standalone weight artifact in the final run directory
- Publication-safe: YES
- Deployment-safe: YES
- Backup model: `twitter16/text_only`

## PHEME
- Primary model: `pheme/tda_only`
- Weighted F1: 0.4447
- Macro F1: 0.4389
- Reliability: WEAK BUT USABLE
- Artifact path: `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/pheme/tda_only`
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_pheme_tda_only.yaml`
- Checkpoint/weights path: not preserved as a standalone weight artifact in the final run directory
- Publication-safe: YES
- Deployment-safe: NO
- Backup model: none recommended

## Single Best Overall Recommendation
- Run: `twitter16/hybrid`
- Weighted F1: 0.8239
- Macro F1: 0.8239
- Artifact path: `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter16/hybrid`
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_hybrid.yaml`
- Checkpoint/weights path: not preserved as a standalone weight artifact in the final run directory

## Final Ranking
- Scientific reliability first:
  - `twitter16/hybrid`
  - `twitter15/text_only`
  - `twitter15/hybrid`
  - `twitter16/text_only`
  - `pheme/tda_only`
  - `twitter16/tda_only`
- Raw weighted F1 first:
  - `twitter16/hybrid`
  - `twitter15/text_only`
  - `twitter15/hybrid`
  - `twitter16/text_only`
  - `pheme/tda_only`
  - `twitter16/tda_only`
