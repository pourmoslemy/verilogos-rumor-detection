# Paper-Ready Claims

## Claims Safe for Abstract
- VeriLogos completes a full nine-run evaluation matrix across Twitter15, Twitter16, and PHEME using TDA-only, text-only, and hybrid settings.
- The strongest audited result is `twitter16/hybrid` with weighted F1 = 0.8239.
- On ACL2017 datasets, text-aware models are consistently stronger than TDA-only baselines.
- On PHEME, only `tda_only` remains non-collapsed under the current protocol, and it remains weak.

## Claims Safe for Results Section
- `twitter15/text_only` is the best Twitter15 run with weighted F1 = 0.8229.
- `twitter16/hybrid` is the best Twitter16 and best overall run with weighted F1 = 0.8239.
- `pheme/tda_only` is the best available PHEME run with weighted F1 = 0.4447, but it should be described as weak.
- `twitter15/tda_only`, `pheme/text_only`, and `pheme/hybrid` exhibit collapse or severe recall imbalance and are excluded from winner claims.

## Claims Safe for Discussion
- Dataset-dependent behavior is substantial; gains on ACL2017 do not transfer cleanly to PHEME.
- Collapse diagnostics are necessary because aggregate F1 alone can obscure one-class failure.
- PHEME requires conservative interpretation under the current protocol.

## Claims That Must NOT Be Made
- DO NOT CLAIM: hybrid is best on every dataset.
- DO NOT CLAIM: text-only is reliable on PHEME.
- DO NOT CLAIM: collapsed runs are valid evidence of model superiority.
- DO NOT CLAIM: current PHEME results support deployment-ready rumor detection.
- DO NOT CLAIM: additional tuning rescued `pheme/text_only`.

## Limitations Wording
- Several runs remain scientifically unreliable because they collapse to one predicted class or exhibit extreme recall imbalance.
- Stored final-run reports predate a later label-order cleanup, so publication claims rely on audited aggregate metrics, confusion matrices, and collapse diagnostics rather than class-name cosmetics.
- PHEME conclusions are limited because only one non-collapsed configuration remains usable and it is weak.

## Reproducibility Statement Wording
- Binary label semantics remain `REAL = 0` and `FAKE = 1`; unsupported labels such as `unverified` or `unconfirmed` are outside the binary claim set and are not silently remapped in the publication package.
- All reported claims are traced to frozen artifacts under `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z` plus decision artifacts for `pheme/text_only`.
- No new training was introduced for the publication package; tables and figures are regenerated from existing metrics only.

## Threat-to-Validity Wording
- Small or unstable event-wise splits can induce severe class collapse in some modality/dataset combinations.
- Dataset-specific behavior limits cross-dataset generalization claims.
- Final run directories do not preserve standalone checkpoint weight files, which constrains direct weight redistribution even though metric artifacts remain auditable.
