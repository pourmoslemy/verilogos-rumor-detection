# VeriLogos Fake News Detection - Academic Optimization Plan

**Date:** 2025-04-21
**Status:** Diagnostic Complete - Implementation Ready

---

## Executive Summary

Diagnostic analysis reveals THREE critical bugs causing system failure:

1. **Feature Dimension Mismatch** (50% failure rate)
2. **Severe Class Imbalance** (majority class prediction)
3. **No Stratified Sampling** (distribution mismatch)

Current performance: F1=0.636, Accuracy=0.467 (worse than random)
Target performance: F1≥0.80, Accuracy≥0.75

---

## Phase 1: Critical Bug Fixes (Expected: +25-35% improvement)

### Bug #1: Feature Extraction Inconsistency

**Location:** verilogos/experiments/runner.py, lines 145-165

**Problem:**


**Fix:**


**Impact:** Fixes 50% of experiment failures

---

### Bug #2: Class Imbalance Handling

**Evidence:**
- Recall = 1.0 (predicting all positive)
- Precision = 0.467 (less than half correct)
- Accuracy = 0.467 (worse than random)

**Root Cause:** No class balancing in models

**Fix 1: Add class_weight='balanced' to all models**

Location: verilogos/experiments/models/classical.py



**Fix 2: Implement SMOTE for training data**

Location: verilogos/experiments/runner.py



**Impact:** Expected +20-30% F1 improvement

---

### Bug #3: Non-Stratified Splits

**Location:** verilogos/experiments/datasets/base.py

**Problem:**


**Fix:**


**Impact:** Consistent class distribution across splits

---

## Phase 2: Feature Engineering (Expected: +10-15% improvement)

### Enhancement 1: Better Text Features

**Current:** Basic TF-IDF (max_features=500)
**Upgrade:** Increase to 1000-2000 features + better preprocessing



### Enhancement 2: Enhanced Topology Features

**Current:** Basic Betti numbers
**Add:** Persistence statistics



---

## Phase 3: Model Optimization (Expected: +5-10% improvement)

### Enhancement 1: Hyperparameter Tuning

Use stratified cross-validation for tuning:



### Enhancement 2: Ensemble Stacking



---

## Phase 4: Evaluation Rigor (Required for publication)

### Enhancement 1: Comprehensive Metrics



### Enhancement 2: Cross-Validation



---

## Implementation Priority

### IMMEDIATE (Phase 1 - Critical Bugs)
1. Fix feature extraction bug (runner.py line 165)
2. Add class_weight='balanced' to all models
3. Implement stratified splits in datasets
4. Add SMOTE to training pipeline

**Expected Result:** F1: 0.75-0.80, Accuracy: 0.70-0.75

### SHORT-TERM (Phase 2 - Features)
5. Increase TF-IDF features to 2000
6. Add persistence statistics to topology features
7. Implement feature selection (SelectKBest)

**Expected Result:** F1: 0.80-0.85, Accuracy: 0.75-0.80

### MEDIUM-TERM (Phase 3 - Models)
8. Hyperparameter tuning with GridSearchCV
9. Implement ensemble stacking
10. Add threshold optimization

**Expected Result:** F1: 0.85-0.90, Accuracy: 0.80-0.85

### LONG-TERM (Phase 4 - Rigor)
11. Comprehensive metrics (AUC-ROC, AUC-PR)
12. Cross-validation with confidence intervals
13. Statistical significance testing

**Expected Result:** Publication-ready results with statistical validation

---

## Success Criteria

### Minimum Acceptable Performance
- F1 Score: ≥ 0.80 (all datasets)
- Accuracy: ≥ 0.75 (all datasets)
- AUC-ROC: ≥ 0.85 (all datasets)
- Precision: ≥ 0.75
- Recall: ≥ 0.75

### Target Performance (SOTA)
- F1 Score: ≥ 0.88
- Accuracy: ≥ 0.85
- AUC-ROC: ≥ 0.90

### Academic Rigor Requirements
- Stratified k-fold CV (k≥5)
- Multiple random seeds (≥5)
- Statistical significance (p<0.05)
- No test set contamination
- Reproducible results

---

## Next Steps

1. Implement Phase 1 fixes (critical bugs)
2. Run experiments and validate improvement
3. If F1 ≥ 0.75, proceed to Phase 2
4. If F1 < 0.75, debug and iterate
5. Continue through phases until target met

---

**Author:** Alireza Pourmoslemi (apmath99@gmail.com)
**Date:** 2025-04-21
