# Final Tuning Recommendation

## Decision
No new full training is justified before publication under the current evidence base.

## Why
- The nine-run matrix is already complete and auditable.
- The remaining failures are scientifically relevant negative results, not hidden pipeline bugs.
- `pheme/text_only` is formally closed as an unresolved failed baseline.
- Additional open-ended tuning would increase post-hoc selection risk and weaken reproducibility.

## Maximum Safe Scope Before Publication
Only deterministic, low-risk work is justified:
- regenerate figures and tables from frozen metrics,
- verify traceability from manuscript claims back to the audited result paths,
- bundle decision artifacts for failed baselines,
- optionally run an audit-only packaging verification pass without new training.

## Not Recommended Before Publication
- new full training,
- broad hyperparameter searches,
- any split or label semantic changes,
- more `pheme/text_only` tuning,
- cherry-picked reruns to rescue collapsed baselines.

## Ranked Options
1. Scientific necessity: audit-only artifact verification.
2. Expected benefit: packaging consistency and figure/table polish.
3. Reproducibility risk: any new model tuning is high risk and should be deferred to future work.
