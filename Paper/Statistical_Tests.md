# Statistical Significance Testing

This document provides formal statistical hypothesis testing for all main claims comparing SSL pretraining vs. training from scratch at 10% labeled data.

## Methodology

### Test Selection: Welch's t-test

We use **Welch's t-test** (unequal variance t-test) rather than Student's t-test because:
1. Sample sizes are equal (n=5 seeds each) but variances differ substantially between methods
2. IEEE-118 Cascade shows high scratch variance (±0.271) reflecting training instability
3. Welch's t-test is robust to heteroscedasticity and is the recommended default

### Effect Size: Cohen's d

We report **Cohen's d** effect size to quantify practical significance:
- Small effect: d = 0.2
- Medium effect: d = 0.5
- Large effect: d = 0.8

### Seeds Used

All experiments use 5 random seeds: **42, 123, 456, 789, 1337**

---

## Results Summary

| Comparison | Scratch (mean±std) | SSL (mean±std) | t-statistic | p-value | Cohen's d | Significance |
|------------|-------------------|----------------|-------------|---------|-----------|--------------|
| Cascade IEEE-24 (F1↑) | 0.7732 ± 0.0164 | 0.8261 ± 0.0179 | -4.88 | 0.001272 | 3.08 | *** |
| Cascade IEEE-118 (F1↑) | 0.2617 ± 0.2712 | 0.8743 ± 0.0564 | -4.95 | 0.006271 | 3.13 | ** |
| Power Flow IEEE-24 (MAE↓) | 0.0149 ± 0.0005 | 0.0106 ± 0.0003 | 16.39 | 0.000001 | 10.50 | *** |
| Line Flow IEEE-24 (MAE↓) | 0.0084 ± 0.0003 | 0.0062 ± 0.0002 | 13.36 | 0.000006 | 8.58 | *** |

**Significance levels:** * p < 0.05, ** p < 0.01, *** p < 0.001

---

## Detailed Analysis

### 1. Cascade Prediction - IEEE-24 Bus System

**Hypothesis:** SSL pretraining improves F1 score for cascade prediction at 10% labels.

**Per-seed F1 scores:**
| Seed | Scratch | SSL |
|------|---------|-----|
| 42 | 0.7619 | 0.8462 |
| 123 | 0.7660 | 0.8431 |
| 456 | 0.7600 | 0.8269 |
| 789 | 0.7925 | 0.8000 |
| 1337 | 0.7857 | 0.8142 |

**Statistics:**
- Scratch: mean = 0.7732, std = 0.0164
- SSL: mean = 0.8261, std = 0.0179
- Improvement: +6.8% absolute (+8.8% relative)
- Welch's t = -4.88, df ≈ 7.9
- **p = 0.001272** (highly significant)
- **Cohen's d = 3.08** (large effect)

**Conclusion:** SSL pretraining provides a statistically significant improvement in cascade prediction on IEEE-24 (p < 0.01) with a large effect size.

---

### 2. Cascade Prediction - IEEE-118 Bus System

**Hypothesis:** SSL pretraining improves F1 score for cascade prediction at 10% labels.

**Per-seed F1 scores:**
| Seed | Scratch | SSL |
|------|---------|-----|
| 42 | 0.6667 | 0.8621 |
| 123 | 0.0000 | 0.8824 |
| 456 | 0.0000 | 0.8235 |
| 789 | 0.2500 | 0.9032 |
| 1337 | 0.3919 | 0.9000 |

**Statistics:**
- Scratch: mean = 0.2617, std = 0.2712
- SSL: mean = 0.8743, std = 0.0564
- Improvement: +61.3% absolute (+234% relative)
- Welch's t = -4.95, df ≈ 4.4
- **p = 0.006271** (significant)
- **Cohen's d = 3.13** (large effect)

**Notable observation:** Training from scratch shows extreme instability on IEEE-118, with 2 of 5 seeds producing F1=0.0 (complete failure to learn). SSL pretraining provides stable convergence across all seeds.

**Conclusion:** SSL pretraining provides a statistically significant improvement in cascade prediction on IEEE-118 (p < 0.01) with a large effect size. The high scratch variance reflects training instability that SSL pretraining resolves.

---

### 3. Power Flow Prediction - IEEE-24 Bus System

**Hypothesis:** SSL pretraining reduces MAE for voltage magnitude prediction at 10% labels.

**Per-seed MAE values:**
| Seed | Scratch | SSL |
|------|---------|-----|
| 42 | 0.01463 | 0.01051 |
| 123 | 0.01500 | 0.01028 |
| 456 | 0.01458 | 0.01046 |
| 789 | 0.01486 | 0.01069 |
| 1337 | 0.01564 | 0.01101 |

**Statistics:**
- Scratch: mean = 0.0149, std = 0.0005
- SSL: mean = 0.0106, std = 0.0003
- Improvement: 29.1% MAE reduction
- Welch's t = 16.39, df ≈ 6.5
- **p = 0.000001** (highly significant)
- **Cohen's d = 10.50** (very large effect)

**Conclusion:** SSL pretraining provides a highly statistically significant improvement in power flow prediction (p < 0.001) with an exceptionally large effect size.

---

### 4. Line Flow Prediction - IEEE-24 Bus System

**Hypothesis:** SSL pretraining reduces MAE for line power flow prediction at 10% labels.

**Per-seed MAE values:**
| Seed | Scratch | SSL |
|------|---------|-----|
| 42 | 0.00854 | 0.00631 |
| 123 | 0.00820 | 0.00618 |
| 456 | 0.00839 | 0.00618 |
| 789 | 0.00824 | 0.00600 |
| 1337 | 0.00873 | 0.00649 |

**Statistics:**
- Scratch: mean = 0.0084, std = 0.0003
- SSL: mean = 0.0062, std = 0.0002
- Improvement: 26.4% MAE reduction
- Welch's t = 13.36, df ≈ 6.8
- **p = 0.000006** (highly significant)
- **Cohen's d = 8.58** (very large effect)

**Conclusion:** SSL pretraining provides a highly statistically significant improvement in line flow prediction (p < 0.001) with a very large effect size.

---

## Summary of Findings

All four comparisons demonstrate **statistically significant improvements** from SSL pretraining at 10% labeled data:

1. **All p-values < 0.01** - Meeting or exceeding conventional significance thresholds
2. **All Cohen's d > 3.0** - Indicating very large practical effect sizes
3. **Consistent across tasks** - Cascade, Power Flow, and Line Flow all benefit
4. **Consistent across topologies** - Both IEEE-24 and IEEE-118 show improvements

The statistical evidence strongly supports the main claim that physics-guided SSL pretraining provides substantial and reliable improvements in low-label power system prediction tasks.

---

## Computation Code

```python
import numpy as np
from scipy import stats

def welch_ttest_and_cohens_d(group1, group2):
    """Compute Welch's t-test and Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = abs(mean1 - mean2) / pooled_std

    return t_stat, p_value, cohens_d
```

---

*Generated: December 16, 2025*
*Validation: PR29 Statistical Significance Testing*
