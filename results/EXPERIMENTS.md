# Experiments

Cross-validated results per modality. Numbers are mean +/- std across folds.

## Symbolic

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | LogReg (baseline) | 0.640 +/- 0.034 | 0.632 +/- 0.032 | 0.634 +/- 0.036 | 5-fold GroupKFold by song_group; 48 feats |
| 2 | LogReg balanced *(best)* | 0.644 +/- 0.037 | 0.639 +/- 0.040 | 0.640 +/- 0.041 | 5-fold GroupKFold by song_group; 48 feats |
| 3 | LogReg balanced C=2 | 0.635 +/- 0.034 | 0.630 +/- 0.035 | 0.632 +/- 0.037 | 5-fold GroupKFold by song_group; 48 feats |
| 4 | RandomForest | 0.637 +/- 0.038 | 0.630 +/- 0.039 | 0.632 +/- 0.037 | 5-fold GroupKFold by song_group; 48 feats |
| 5 | HistGradientBoosting | 0.631 +/- 0.040 | 0.624 +/- 0.041 | 0.627 +/- 0.042 | 5-fold GroupKFold by song_group; 48 feats |
| 6 | LogReg tuned (nested CV) | 0.639 +/- 0.034 | 0.632 +/- 0.035 | 0.634 +/- 0.037 | nested 5x3 GroupKFold; expanded 48 feats |
| 7 | HistGB tuned (nested CV) | 0.642 +/- 0.038 | 0.636 +/- 0.039 | 0.638 +/- 0.039 | nested 5x3 GroupKFold; expanded 48 feats |

- **1. LogReg (baseline)** — per-class F1: Q1=0.665, Q2=0.681, Q3=0.535, Q4=0.649
- **2. LogReg balanced** — per-class F1: Q1=0.672, Q2=0.680, Q3=0.551, Q4=0.652
- **3. LogReg balanced C=2** — per-class F1: Q1=0.665, Q2=0.671, Q3=0.539, Q4=0.644
- **4. RandomForest** — per-class F1: Q1=0.637, Q2=0.662, Q3=0.566, Q4=0.654
- **5. HistGradientBoosting** — per-class F1: Q1=0.657, Q2=0.665, Q3=0.529, Q4=0.646
- **6. LogReg tuned (nested CV)** — per-class F1: Q1=0.660, Q2=0.682, Q3=0.541, Q4=0.646; best per-fold: C=0.25, class_weight=None
- **7. HistGB tuned (nested CV)** — per-class F1: Q1=0.645, Q2=0.659, Q3=0.580, Q4=0.658; best per-fold: l2_regularization=1.0, learning_rate=0.1, max_iter=400

## Lyrics

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | LogReg word 1-2gram (baseline) | 0.667 +/- 0.017 | 0.659 +/- 0.017 | 0.661 +/- 0.017 | 5-fold StratifiedKFold; TF-IDF features |
| 2 | LogReg balanced | 0.668 +/- 0.017 | 0.662 +/- 0.017 | 0.663 +/- 0.017 | 5-fold StratifiedKFold; TF-IDF features |
| 3 | LinearSVC word+char | 0.678 +/- 0.009 | 0.671 +/- 0.009 | 0.672 +/- 0.009 | 5-fold StratifiedKFold; TF-IDF features |
| 4 | ComplementNB word | 0.670 +/- 0.018 | 0.657 +/- 0.020 | 0.661 +/- 0.018 | 5-fold StratifiedKFold; TF-IDF features |
| 5 | LinearSVC tuned (nested CV) *(best)* | 0.697 +/- 0.007 | 0.690 +/- 0.008 | 0.692 +/- 0.007 | nested 5x3 StratifiedKFold; TF-IDF word+char |

- **1. LogReg word 1-2gram (baseline)** — per-class F1: Q1=0.635, Q2=0.794, Q3=0.598, Q4=0.609
- **2. LogReg balanced** — per-class F1: Q1=0.634, Q2=0.810, Q3=0.598, Q4=0.607
- **3. LinearSVC word+char** — per-class F1: Q1=0.637, Q2=0.829, Q3=0.621, Q4=0.598
- **4. ComplementNB word** — per-class F1: Q1=0.626, Q2=0.788, Q3=0.619, Q4=0.595
- **5. LinearSVC tuned (nested CV)** — per-class F1: Q1=0.658, Q2=0.848, Q3=0.642, Q4=0.615; best per-fold: C=0.5, class_weight=None, ngram_range=(1, 2), sublinear_tf=True

## Audio

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | RandomForest balanced (baseline) | 0.640 +/- 0.014 | 0.395 +/- 0.025 | 0.443 +/- 0.015 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 2 | RandomForest balanced_subsample | 0.640 +/- 0.018 | 0.410 +/- 0.028 | 0.449 +/- 0.019 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 3 | HistGradientBoosting | 0.648 +/- 0.009 | 0.438 +/- 0.011 | 0.464 +/- 0.005 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 4 | LogReg balanced (scaled, top-80) | 0.558 +/- 0.028 | 0.484 +/- 0.028 | 0.488 +/- 0.031 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 5 | RandomForest top-80 features | 0.648 +/- 0.015 | 0.423 +/- 0.016 | 0.458 +/- 0.012 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 6 | Regression-derived quadrant | 0.648 +/- 0.008 | 0.483 +/- 0.017 | 0.490 +/- 0.012 | 5-fold; 2x RF regressor on V/A, threshold at 5.0; 260 feats |
| 7 | RandomForest + SMOTE *(best)* | 0.614 +/- 0.017 | 0.494 +/- 0.025 | 0.493 +/- 0.022 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 8 | HistGB + SMOTE (scaled) | 0.631 +/- 0.018 | 0.471 +/- 0.034 | 0.479 +/- 0.027 | 5-fold StratifiedKFold; 260 openSMILE feats |
| 9 | Regression-derived + tuned thresholds | 0.647 +/- 0.010 | 0.472 +/- 0.018 | 0.487 +/- 0.015 | 5-fold; 2x RF regressor on V/A, learned thresholds; 260 feats |

- **1. RandomForest balanced (baseline)** — per-class F1: Q1=0.726, Q2=0.025, Q3=0.741, Q4=0.089
- **2. RandomForest balanced_subsample** — per-class F1: Q1=0.728, Q2=0.070, Q3=0.739, Q4=0.103
- **3. HistGradientBoosting** — per-class F1: Q1=0.742, Q2=0.104, Q3=0.753, Q4=0.154
- **4. LogReg balanced (scaled, top-80)** — per-class F1: Q1=0.667, Q2=0.269, Q3=0.710, Q4=0.289
- **5. RandomForest top-80 features** — per-class F1: Q1=0.732, Q2=0.080, Q3=0.747, Q4=0.136
- **6. Regression-derived quadrant** — per-class F1: Q1=0.742, Q2=0.163, Q3=0.756, Q4=0.271; Quadrant inferred from predicted valence/arousal rather than direct classification.
- **7. RandomForest + SMOTE** — per-class F1: Q1=0.714, Q2=0.259, Q3=0.721, Q4=0.282
- **8. HistGB + SMOTE (scaled)** — per-class F1: Q1=0.735, Q2=0.203, Q3=0.749, Q4=0.197
- **9. Regression-derived + tuned thresholds** — per-class F1: Q1=0.739, Q2=0.120, Q3=0.757, Q4=0.272; best per-fold: v_thr=4.88, a_thr=5.0

## Fusion

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | Lyrics only | 0.692 +/- 0.018 | 0.677 +/- 0.017 | 0.678 +/- 0.018 | 5-fold StratifiedKFold; TF-IDF word+char -> LinearSVC |
| 2 | Audio only | 0.648 +/- 0.018 | 0.625 +/- 0.019 | 0.629 +/- 0.018 | 5-fold StratifiedKFold; 101 librosa feats, scaled -> LinearSVC |
| 3 | Fused (audio + lyrics) | 0.752 +/- 0.018 | 0.738 +/- 0.018 | 0.738 +/- 0.017 | 5-fold StratifiedKFold; early fusion: TF-IDF + scaled audio -> LinearSVC |
| 4 | Fused + SVD(300) | 0.746 +/- 0.005 | 0.730 +/- 0.006 | 0.731 +/- 0.006 | 5-fold; TF-IDF->SVD(300) + scaled audio -> LinearSVC |
| 5 | Weighted late fusion (nested CV) *(best)* | 0.758 +/- 0.015 | 0.743 +/- 0.015 | 0.744 +/- 0.015 | nested 5x3; per-modality LinearSVC, z-normed scores |

- **1. Lyrics only** — per-class F1: Q1=0.652, Q2=0.848, Q3=0.615, Q4=0.593
- **2. Audio only** — per-class F1: Q1=0.656, Q2=0.857, Q3=0.502, Q4=0.486
- **3. Fused (audio + lyrics)** — per-class F1: Q1=0.749, Q2=0.915, Q3=0.664, Q4=0.622
- **4. Fused + SVD(300)** — per-class F1: Q1=0.734, Q2=0.915, Q3=0.660, Q4=0.611
- **5. Weighted late fusion (nested CV)** — per-class F1: Q1=0.751, Q2=0.928, Q3=0.673, Q4=0.621; best per-fold lyrics weight mode=0.6

## Transformer

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | LinearSVC reference (official split) | 0.672 | 0.665 | 0.665 | MERGE official 70/15/15; TF-IDF word+char -> LinearSVC |
| 2 | DistilBERT fine-tuned (3 epochs) *(best)* | 0.724 | 0.716 | 0.715 | MERGE official 70/15/15; distilbert-base-uncased, max_len=256 |

- **1. LinearSVC reference (official split)** — per-class F1: Q1=0.612, Q2=0.818, Q3=0.639, Q4=0.592
- **2. DistilBERT fine-tuned (3 epochs)** — per-class F1: Q1=0.662, Q2=0.882, Q3=0.659, Q4=0.660

## Llm

| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |
|---|---|---|---|---|---|
| 1 | gpt-4.1-mini (zero-shot) | 0.716 | 0.712 | 0.716 | MERGE official 70/15/15 test (n=384); zero-shot prompt |
| 2 | gpt-4.1-mini (few-shot) | 0.706 | 0.697 | 0.703 | MERGE official 70/15/15 test (n=384); few-shot prompt |
| 3 | gpt-5.5 (zero-shot) *(best)* | 0.773 | 0.764 | 0.770 | MERGE official 70/15/15 test (n=384); zero-shot prompt |
| 4 | gpt-5.5 (few-shot) | 0.763 | 0.751 | 0.760 | MERGE official 70/15/15 test (n=384); few-shot prompt |

- **1. gpt-4.1-mini (zero-shot)** — per-class F1: Q1=0.714, Q2=0.808, Q3=0.714, Q4=0.613
- **2. gpt-4.1-mini (few-shot)** — per-class F1: Q1=0.670, Q2=0.817, Q3=0.728, Q4=0.571
- **3. gpt-5.5 (zero-shot)** — per-class F1: Q1=0.747, Q2=0.870, Q3=0.790, Q4=0.651
- **4. gpt-5.5 (few-shot)** — per-class F1: Q1=0.740, Q2=0.855, Q3=0.798, Q4=0.612
