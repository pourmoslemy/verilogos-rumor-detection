# Final Findings Executive Summary

- Operational readiness: all 9 final runs completed successfully and all required artifacts were collected.
- Best overall model: `twitter16 / hybrid` with weighted F1 `0.8239`.
- Best per dataset:
  - twitter15: `text_only`
  - twitter16: `hybrid`
  - pheme: `tda_only`
- Hybrid effectiveness:
  - twitter15: does not improve over the best single-mode by `-0.0136` weighted F1.
  - twitter16: improves over the best single-mode by `+0.0348` weighted F1.
  - pheme: does not improve over the best single-mode by `-0.1259` weighted F1.
- Critical scientific warning:
  - `pheme / hybrid` collapses to one predicted class and is not reliable for final scientific conclusions.
  - Additional collapse cases exist in `pheme / text_only` and `twitter15 / tda_only`.
- Recommendation: present operational completion as a success, but present scientific readiness as conditional pending collapse remediation and reruns.
