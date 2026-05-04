# Publication Tables

All metrics are traced to `results/final_runs/20260429T142746Z` and the documented `pheme/text_only` closure artifact when applicable.

Binary label contract for all publication tables remains `REAL = 0`, `FAKE = 1`. Unsupported labels such as `unverified` or `unconfirmed` are outside the reported binary claim set. Markdown and LaTeX tables round to four decimals; CSV preserves raw precision.

## Main Results Table
| Dataset | Mode | Reliability | Acc | Weighted F1 | Macro F1 | Fake Recall | Real Recall | ROC-AUC | AP | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Twitter15 | TDA Only | FAILED / UNRELIABLE | 0.5000 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | 0.6625 | 0.6457 | collapse; not eligible for final ranking |
| Twitter15 | Text Only | VALID | 0.8230 | 0.8229 | 0.8229 | 0.7965 | 0.8496 | 0.9020 | 0.8944 | BEST IN DATASET |
| Twitter15 | Hybrid | VALID | 0.8097 | 0.8093 | 0.8093 | 0.7611 | 0.8584 | 0.8955 | 0.8697 | — |
| Twitter16 | TDA Only | WEAK BUT USABLE | 0.4960 | 0.4153 | 0.4171 | 0.1270 | 0.8710 | 0.4521 | 0.4719 | class imbalance warning |
| Twitter16 | Text Only | VALID | 0.7920 | 0.7892 | 0.7890 | 0.9048 | 0.6774 | 0.9183 | 0.9089 | — |
| Twitter16 | Hybrid | VALID | 0.8240 | 0.8239 | 0.8239 | 0.8413 | 0.8065 | 0.9094 | 0.8747 | BEST OVERALL; BEST IN DATASET |
| PHEME | TDA Only | WEAK BUT USABLE | 0.5263 | 0.4447 | 0.4389 | 0.8974 | 0.1351 | 0.5499 | 0.5407 | BEST IN DATASET; class imbalance warning |
| PHEME | Text Only | FAILED / UNRELIABLE | 0.5132 | 0.3481 | 0.3391 | 0.0000 | 1.0000 | 0.4269 | 0.4748 | collapse; not eligible for final ranking |
| PHEME | Hybrid | FAILED / UNRELIABLE | 0.4868 | 0.3188 | 0.3274 | 0.0000 | 1.0000 | 0.5177 | 0.5308 | collapse; not eligible for final ranking |

## Reliable-Only Ranking
| Rank | Dataset | Mode | Reliability | Weighted F1 | Macro F1 | Recall Gap | Artifact Path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Twitter16 | Hybrid | VALID | 0.8239 | 0.8239 | 0.0348 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter16/hybrid |
| 2 | Twitter15 | Text Only | VALID | 0.8229 | 0.8229 | 0.0531 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter15/text_only |
| 3 | Twitter15 | Hybrid | VALID | 0.8093 | 0.8093 | 0.0973 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter15/hybrid |
| 4 | Twitter16 | Text Only | VALID | 0.7892 | 0.7890 | 0.2274 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter16/text_only |
| 5 | PHEME | TDA Only | WEAK BUT USABLE | 0.4447 | 0.4389 | 0.7623 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/pheme/tda_only |
| 6 | Twitter16 | TDA Only | WEAK BUT USABLE | 0.4153 | 0.4171 | 0.7440 | /home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z/twitter16/tda_only |

## Per-Dataset Winner Table
| Dataset | Primary Winner | Weighted F1 | Reliability | Backup | Backup Weighted F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Twitter15 | Text Only | 0.8229 | VALID | Hybrid | 0.8093 | backup retained for contingency only |
| Twitter16 | Hybrid | 0.8239 | VALID | Text Only | 0.7892 | backup retained for contingency only |
| PHEME | TDA Only | 0.4447 | WEAK BUT USABLE | None recommended | NA | backup omitted because no other non-collapsed run exists |

## Collapse Diagnostics Table
| Dataset | Mode | Reliability | One Class | Severe Collapse | Pred Fake | Pred Real | Fake Recall | Real Recall | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Twitter15 | TDA Only | FAILED / UNRELIABLE | true | true | 226 | 0 | 1.0000 | 0.0000 | [[113, 0], [113, 0]] |
| Twitter15 | Text Only | VALID | false | false | 107 | 119 | 0.7965 | 0.8496 | [[90, 23], [17, 96]] |
| Twitter15 | Hybrid | VALID | false | false | 102 | 124 | 0.7611 | 0.8584 | [[86, 27], [16, 97]] |
| Twitter16 | TDA Only | WEAK BUT USABLE | false | false | 16 | 109 | 0.1270 | 0.8710 | [[8, 55], [8, 54]] |
| Twitter16 | Text Only | VALID | false | false | 77 | 48 | 0.9048 | 0.6774 | [[57, 6], [20, 42]] |
| Twitter16 | Hybrid | VALID | false | false | 65 | 60 | 0.8413 | 0.8065 | [[53, 10], [12, 50]] |
| PHEME | TDA Only | WEAK BUT USABLE | false | false | 67 | 9 | 0.8974 | 0.1351 | [[35, 4], [32, 5]] |
| PHEME | Text Only | FAILED / UNRELIABLE | true | true | 0 | 76 | 0.0000 | 1.0000 | [[39, 0], [37, 0]] |
| PHEME | Hybrid | FAILED / UNRELIABLE | true | true | 0 | 76 | 0.0000 | 1.0000 | [[0, 39], [0, 37]] |

## Class-Balance / Recall-Balance Warning Table
| Dataset | Mode | Weighted F1 | Macro F1 | Fake Recall | Real Recall | Recall Gap | Warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Twitter15 | TDA Only | 0.3333 | 0.3333 | 1.0000 | 0.0000 | 1.0000 | HIGH |
| Twitter15 | Text Only | 0.8229 | 0.8229 | 0.7965 | 0.8496 | 0.0531 | LOW |
| Twitter15 | Hybrid | 0.8093 | 0.8093 | 0.7611 | 0.8584 | 0.0973 | LOW |
| Twitter16 | TDA Only | 0.4153 | 0.4171 | 0.1270 | 0.8710 | 0.7440 | HIGH |
| Twitter16 | Text Only | 0.7892 | 0.7890 | 0.9048 | 0.6774 | 0.2274 | LOW |
| Twitter16 | Hybrid | 0.8239 | 0.8239 | 0.8413 | 0.8065 | 0.0348 | LOW |
| PHEME | TDA Only | 0.4447 | 0.4389 | 0.8974 | 0.1351 | 0.7623 | HIGH |
| PHEME | Text Only | 0.3481 | 0.3391 | 0.0000 | 1.0000 | 1.0000 | HIGH |
| PHEME | Hybrid | 0.3188 | 0.3274 | 0.0000 | 1.0000 | 1.0000 | HIGH |

Warning note: stored `classification_report.txt` artifacts predate later label-order cleanup, so ranking claims rely on audited aggregate metrics, confusion matrices, and collapse diagnostics rather than class-name cosmetics.
