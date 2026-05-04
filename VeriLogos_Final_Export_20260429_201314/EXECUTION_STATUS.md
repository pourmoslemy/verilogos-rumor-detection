# Execution Status

- Operational readiness: all 9 training runs completed successfully.
- Detached remote session finished cleanly and wrote all expected logs.
- Progress trace:

```text
STARTED 2026-04-29T14:47:14Z
RUNNING 1 twitter15 tda_only 2026-04-29T14:47:14Z
COMPLETED 1 twitter15 tda_only 2026-04-29T14:58:44Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/1_twitter15_tda_only.log
RUNNING 2 twitter16 tda_only 2026-04-29T14:58:44Z
COMPLETED 2 twitter16 tda_only 2026-04-29T15:05:50Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/2_twitter16_tda_only.log
RUNNING 3 pheme tda_only 2026-04-29T15:05:50Z
COMPLETED 3 pheme tda_only 2026-04-29T15:05:59Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/3_pheme_tda_only.log
RUNNING 4 twitter15 text_only 2026-04-29T15:05:59Z
COMPLETED 4 twitter15 text_only 2026-04-29T15:21:05Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/4_twitter15_text_only.log
RUNNING 5 twitter16 text_only 2026-04-29T15:21:05Z
COMPLETED 5 twitter16 text_only 2026-04-29T15:29:45Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/5_twitter16_text_only.log
RUNNING 6 pheme text_only 2026-04-29T15:29:45Z
COMPLETED 6 pheme text_only 2026-04-29T15:33:06Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/6_pheme_text_only.log
RUNNING 7 twitter15 hybrid 2026-04-29T15:33:06Z
COMPLETED 7 twitter15 hybrid 2026-04-29T16:00:54Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/7_twitter15_hybrid.log
RUNNING 8 twitter16 hybrid 2026-04-29T16:00:54Z
COMPLETED 8 twitter16 hybrid 2026-04-29T16:16:35Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/8_twitter16_hybrid.log
RUNNING 9 pheme hybrid 2026-04-29T16:16:35Z
COMPLETED 9 pheme hybrid 2026-04-29T16:21:03Z log=/home/ubuntu/colab_deployment/results/final_training_20260429T144709Z/logs/9_pheme_hybrid.log
ALL_COMPLETED 2026-04-29T16:21:03Z
```

- Copied artifacts include 9 `results.json`, 9 `classification_report.txt`, 9 run logs, confusion matrices, ROC/PR curves, training history images, and metrics comparison images.
- Runtime note: training executed in the audited server environment where CUDA was unavailable during readiness validation, so observed runtimes reflect CPU execution.
