# Publication-Ready Executive Summary

## Project Status
- Execution status: COMPLETED
- Readiness status carried into this package: READY_FOR_AUDIT_RERUN
- Artifact basis: `/home/ubuntu/colab_deployment/results/final_runs/20260429T142746Z`
- Decision overlay used: `/home/ubuntu/colab_deployment/results/DECISION_PHEME_TEXT_ONLY_UNRESOLVED.json`

## Label Semantics and Class Contract
- Binary publication contract remains `REAL = 0`, `FAKE = 1`.
- Unsupported labels such as `unverified` or `unconfirmed` are outside the binary publication claim set and are not silently remapped in this package.

## Reliable Findings Only
- Best overall scientifically reliable run: `twitter16/hybrid` with weighted F1 = 0.8239.
- Best Twitter15 run: `twitter15/text_only` with weighted F1 = 0.8229.
- Best Twitter16 run: `twitter16/hybrid` with weighted F1 = 0.8239.
- Best available PHEME run: `pheme/tda_only` with weighted F1 = 0.4447, but it remains weak and should be presented conservatively.
- On ACL2017, text-aware models materially outperform TDA-only baselines.

## Collapsed / Unreliable Runs
- `twitter15/tda_only`: failed one-class baseline, weighted F1 = 0.3333.
- `pheme/text_only`: officially closed as unresolved failed baseline; one-class prediction remains after macro-F1 checkpoint selection; weighted F1 = 0.3481 in the closure artifact.
- `pheme/hybrid`: failed one-class baseline, weighted F1 = 0.3188.

## Scientific Limitations
- PHEME remains unstable: only one non-collapsed mode survives and it is weak.
- Final run directories do not preserve standalone checkpoint weight files, so recommendations are artifact-path based.
- Stored final-run `classification_report.txt` files predate later label-order cleanup, so publication claims should emphasize aggregate metrics, confusion matrices, and collapse diagnostics.

## What Can Be Claimed
- VeriLogos achieves strong audited performance on Twitter15 and Twitter16.
- `twitter16/hybrid` is the best overall reliable configuration.
- TDA-only baselines are weaker than text-aware models on ACL2017.
- PHEME is materially harder under the current protocol, and text-aware baselines can collapse.

## What Cannot Be Claimed
- DO NOT CLAIM: hybrid is universally best.
- DO NOT CLAIM: PHEME text-only is fixed or scientifically reliable.
- DO NOT CLAIM: multimodal fusion is validated on PHEME.
- DO NOT CLAIM: collapsed baselines support winner claims.

## Publication Recommendation
- The current evidence base is publishable only with explicit limitation language and explicit exclusion of collapsed runs from scientific ranking claims.
- No new full training is justified before publication.

FINAL PUBLICATION DECISION:
- Publishable now: YES
- Publishable with explicit limitations: YES
- Requires final audit rerun before publication: NO
- Requires new full training before publication: NO
- Best overall reported model: twitter16/hybrid + weighted F1 = 0.8239
- Unreliable runs explicitly excluded from scientific claims: twitter15/tda_only, pheme/text_only, pheme/hybrid
