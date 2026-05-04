# Decision: PHEME Text-Only Closed as Unresolved Baseline

## Final Decision
- `pheme/text_only` is NOT FIXED.
- It is scientifically unreliable under the current protocol.
- It is retained only as a weak unstable negative baseline.
- It should not block final reporting/publication.
- No further open-ended tuning should be performed for this run.

## Evidence
- selected epoch: 1
- validation macro-F1: 0.3763440860215054
- validation weighted-F1: 0.45420837968112715
- validation predicted distribution: all REAL
- validation REAL recall: 1.0
- validation FAKE recall: 0.0
- test accuracy: 0.5131578947368421
- test weighted-F1: 0.3480549199084668
- test macro-F1: 0.3391304347826087
- test confusion matrix: [[39, 0], [37, 0]]
- test REAL recall: 1.0
- test FAKE recall: 0.0
- one_class_predicted: true
- severe_collapse: true
- collapse_resolved: false

## Why Macro-F1 Checkpointing Did Not Resolve Collapse
- The patch worked mechanically.
- The selected checkpoint was chosen using validation macro-F1.
- But the selected checkpoint still predicted only REAL.
- Therefore checkpoint selection alone is insufficient.

## Patch Status
- Keep the macro-F1 checkpointing patch.
- It is accepted for auditability.
- It is backward-compatible.
- It should not be reverted.

## Scientific Interpretation
- The failure is likely due to text-only representation instability, event-wise/split distribution shift, or modality mismatch on PHEME.
- This is not currently treated as a systemic pipeline failure.

## Project Priority Shift
1. Analyze `pheme/tda_only`
2. Confirm `pheme/hybrid`
3. Plan full final rerun
4. Avoid further open-ended tuning of `pheme/text_only`

## Publication/Reporting Guidance
- Report this as a negative/unstable baseline.
- Do not hide the failure.
- Do not over-claim text-only performance on PHEME.
- Emphasize that modality effectiveness depends on dataset structure.

## Audit Trail
- decision artifact creation timestamp: 2026-04-30T06:17:47Z
- approval file path: `/home/ubuntu/colab_deployment/OWNER_PREAPPROVAL_PHEME_TEXT_ONLY_DECISION.md`
- artifact directory if it exists: `/home/ubuntu/colab_deployment/results/pheme_text_only_macro_f1_checkpoint_20260430_055513/`
- config path: `/home/ubuntu/colab_deployment/configs/pheme_text_only_macro_f1_checkpoint.yaml`
- note that no new training was run for this decision memo
- note that no dataset semantics were changed
