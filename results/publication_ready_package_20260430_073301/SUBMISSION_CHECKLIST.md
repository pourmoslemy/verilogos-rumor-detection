# Submission Checklist

## Validations
- [x] Metrics validated against `results.json` and `classification_report.txt`.
- [x] Figures validated against frozen metrics only.
- [x] Tables validated against frozen metrics only.
- [x] Reliable runs separated from failed runs.
- [x] Artifact paths verified.
- [x] Collapsed runs disclosed.
- [x] Limitations disclosed.
- [x] Audit status disclosed.
- [x] Rerun requirement disclosed.
- [x] Reproducibility materials identified.
- [x] Binary label semantics documented (`REAL = 0`, `FAKE = 1`; unsupported `unverified`/`unconfirmed` labels not remapped in publication claims).

## Failed / Unreliable Runs
- [x] `twitter15/tda_only` marked failed/unreliable.
- [x] `pheme/text_only` marked failed/unreliable.
- [x] `pheme/hybrid` marked failed/unreliable.

## Remaining Packaging Checks
- [ ] Decide whether appendix should include reliable confusion-matrix summary.
- [ ] If weight release is required, verify whether checkpoints are stored outside the final run directories.