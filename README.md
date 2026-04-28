# Stress Detection Using Physiological & Sleep Patterns

**Authors:** Dakshayani Sharma · Ridhima Verma · Samriddhi Saxena  
**Institution:** Manipal Institute of Technology, Manipal — B.Tech Information Technology  
**Format:** IEEE Research Paper (2025)

---

## Overview

An end-to-end machine learning pipeline for non-intrusive stress detection using
multimodal wearable sensor data. The system combines HRV (Heart Rate Variability)
physiological signals with sleep diary behavioral data from 49 participants,
using ensemble learning to classify stress based on GAD-7 survey ground truth.

---

## Key Results

| Metric | Score | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.81 | ±0.04 |
| PR-AUC | 0.74 | — |
| F1-Score | 0.72 | — |
| Recall | 0.76 | — |

Evaluated using 5-fold stratified cross-validation with bootstrap confidence intervals.

---

## Dataset

| Source | Records | Description |
|--------|---------|-------------|
| `sensor_hrv_filtered.csv` | 38,921 | HRV + instantaneous HR with Unix timestamps |
| `sleep_diary.csv` | 1,372 | Sleep duration, latency, WASO, efficiency |
| `survey.csv` | 49 participants | GAD-7 stress labels (score ≥ 10 = stressed) |

**Class distribution:** 16.7% stressed · 83.3% non-stressed (1:5 ratio)  
Dataset not included in repo due to size. Available via original study authors.

---

## Methodology

### Feature Engineering
- **Physiological (9 features):** Recovery index (RMSSD/HR), stress index (HR/RMSSD),
  log-RMSSD, day-over-day HR delta, 3-day rolling means of HR and RMSSD, HR stats (min/max/std)
- **Sleep (4 features):** Sleep duration (hours), sleep latency, WASO, sleep quality score (duration × efficiency)

### Pipeline
1. **Preprocessing** — Unix timestamp → date conversion, HRV daily aggregation, median imputation
2. **Outlier Removal** — Isolation Forest (5% contamination, minority class preserved)
3. **Class Imbalance** — SMOTE per fold + class weighting in both classifiers
4. **Model** — Dual pipeline ensemble:
   - XGBoost on physiological features (weight: 0.6)
   - Random Forest (200 trees) on sleep features (weight: 0.4)
   - Final: `prob = 0.6 × P_physio + 0.4 × P_sleep`
5. **Threshold Selection** — Optimized per fold via Precision-Recall curve (max F1)
6. **Validation** — 5-fold stratified CV + 1000-iteration bootstrap CIs

---

## Top Predictive Features

| Rank | Feature | Type |
|------|---------|------|
| 1 | Recovery Index (RMSSD/HR) | Physiological |
| 2 | RMSSD Mean | Physiological |
| 3 | Sleep Latency | Sleep |

Physiological features contributed ~60% of total importance; sleep features ~40%.

---

## Notebook Outputs

All figures and tables are rendered inline within the notebook and visible directly on GitHub:

- Feature distribution boxplots by stress label
- ROC & PR curve comparison across 3 modalities (physiological, sleep, combined)
- Top 10 feature importances + cumulative importance curve
- Fusion weight optimization sweep (Physio:Sleep ratio)
- Confusion matrix with clinical performance metrics (sensitivity, specificity, Cohen's Kappa)
- Table I: Mean feature values by stress label with statistical significance
- Table II: Per-fold cross-validation metrics (5 folds)

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run via Jupyter
jupyter notebook stress_sleep_analysis.ipynb
```

Update the dataset path in the notebook:
```python
DATASET_DIR = "/path/to/your/data"   # change this line
```

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Matplotlib` · `Seaborn` · `SciPy`

---

## Project Structure

```
Stress-Sleep-Analysis/
├── stress_sleep_analysis.ipynb   # Full ML pipeline
├── requirements.txt
├── .gitignore
└── README.md
```
