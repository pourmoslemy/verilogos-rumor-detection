# Final Recommended Configs

Configs are the readiness-approved YAML files under `results/readiness_rerun_20260429T142746Z/proposed_configs/`.

## twitter15/tda_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter15_tda_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter15
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## twitter15/text_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter15_text_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter15
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## twitter15/hybrid
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter15_hybrid.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter15
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## twitter16/tda_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_tda_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter16
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## twitter16/text_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_text_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter16
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## twitter16/hybrid
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_hybrid.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/acl2017/twitter16
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## pheme/tda_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_pheme_tda_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/pheme/processed_ready_20260429
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## pheme/text_only
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_pheme_text_only.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/pheme/processed_ready_20260429
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## pheme/hybrid
- Config path: `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_pheme_hybrid.yaml`
- Seed: `42`
- Batch size: `16`
- Learning rate: `2e-05`
- Epochs: `15`
- Scheduler: not explicitly configured
- Early stopping / checkpoint criterion: validation loss (default)
- Loss function: cross_entropy (assumed default)
- Preprocessing assumptions: dataset path fixed to /home/ubuntu/colab_deployment/data/pheme/processed_ready_20260429
- Caveats: Use only for the audited dataset/mode pairing. Collapsed runs remain documented but not recommended.

## Summary Recommendations
- Best current publication config: `twitter16/hybrid` via `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_hybrid.yaml`
- Best current stable config: `twitter16/hybrid` via `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_twitter16_hybrid.yaml`
- Best current high-risk/high-reward config: `pheme/tda_only` via `/home/ubuntu/colab_deployment/results/readiness_rerun_20260429T142746Z/proposed_configs/proposed_pheme_tda_only.yaml`
