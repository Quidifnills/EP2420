<div style="text-align: center; padding: 100px 0;">

<h1 style="font-size: 28px;">EP2420 Project 1</h1>

<h2 style="font-size: 22px;">Task I: Data Exploration and Pre-processing</h2>

---

<h3 style="font-size: 18px;">Estimating Service Metrics from Device Measurements</h3>

<br><br>

**Author:** Yiyi Miao

**Date:** October 31, 2025

**Course:** EP2420/EP272V

<br><br>

**Datasets:**
- `2025_JNSM_VoD_flashcrowd_2`
- `2025_JNSM_KV_flashcrowd_2`

</div>

<div style="page-break-after: always;"></div>

---

<h2 style="font-size: 20px;">Question 1: Design Matrix Description and Target Statistics</h2>

<h3 style="font-size: 16px;">1.1 Design Matrix Dimensions</h3>

<h4 style="font-size: 14px;">KV Dataset </h4>

- **Number of sample rows:** 9,722
- **Number of feature columns:** 1,723

<h4 style="font-size: 14px;">VoD Dataset </h4>

- **Number of sample rows:** 18,317
- **Number of feature columns:** 1,670

---

<h3 style="font-size: 16px;">1.2 Target Variable Statistics</h3>

<h4 style="font-size: 14px;">KV Target (ReadsAvg) Statistics</h4>

| Statistic | Value |
|-----------|-------|
| Mean | 55.3 ms |
| Standard Deviation | 3.13 ms |
| Maximum | 152 ms |
| Minimum | 51.1 ms |
| 25th Percentile | 53.3 ms |
| 50th Percentile | 54.1 ms |
| 95th Percentile | 60.5 ms |

<h4 style="font-size: 14px;">VoD Target (DispFrames) Statistics</h4>

| Statistic | Value |
|-----------|-------|
| Mean | 22.0 fps |
| Standard Deviation | 4.33 fps |
| Maximum | 25 fps |
| Minimum | 0 fps |
| 25th Percentile | 24 fps |
| 50th Percentile | 24 fps |
| 95th Percentile | 24 fps |

---

<h3 style="font-size: 16px;">1.3 Target Distribution Visualization</h3>

<div style="display: flex; justify-content: space-around; margin: 20px 0;">

<div style="text-align: center;">

**KV Target Distribution**

<img src="task1_KV_Y.png" alt="KV Target Distribution" width="450"/>

</div>

<div style="text-align: center;">

**VoD Target Distribution**

<img src="task1_VoD_Y.png" alt="VoD Target Distribution" width="450"/>

</div>

</div>

**Observations:**
- The KV target shows a concentrated distribution around 54-55 ms with a long right tail, indicating occasional high response times.
- The VoD target is heavily concentrated at 24 fps (standard video frame rate), with some samples dropping to lower values, suggesting service degradation events.

<div style="page-break-after: always;"></div>

---

<h2 style="font-size: 20px;">Question 2: Data Pre-processing Methods</h2>

<!-- Six different pre-processing methods were applied to both datasets, producing design matrices $X_1, X_2, \ldots, X_6$: -->

<h3 style="font-size: 16px;">Pre-processing Transformations</h3>

| Matrix | Method | Application Axis |
|--------|--------|------------------|
| $X_1$ | L2 Normalization | Feature columns (axis=0) |
| $X_2$ | L2 Normalization | Sample rows (axis=1) |
| $X_3$ | Min-Max Scaling [0,1] | Feature columns |
| $X_4$ | Min-Max Scaling [0,1] | Sample rows |
| $X_5$ | Standardization (μ=0, σ=1) | Feature columns |
| $X_6$ | Standardization (μ=0, σ=1) | Sample rows |

<h3 style="font-size: 16px;">Implementation Code</h3>

```python
# KV Dataset Pre-processing
KV_X_1 = normalize(KV_X0, norm='l2', axis=0)
KV_X_2 = normalize(KV_X0, norm='l2', axis=1)
KV_X_3 = MinMaxScaler().fit_transform(KV_X0)
KV_X_4 = MinMaxScaler().fit_transform(KV_X0.T).T
KV_X_5 = StandardScaler().fit_transform(KV_X0)
KV_X_6 = StandardScaler().fit_transform(KV_X0.T).T

# VoD Dataset Pre-processing
VoD_X_1 = normalize(VoD_X0, norm='l2', axis=0)
VoD_X_2 = normalize(VoD_X0, norm='l2', axis=1)
VoD_X_3 = MinMaxScaler().fit_transform(VoD_X0)
VoD_X_4 = MinMaxScaler().fit_transform(VoD_X0.T).T
VoD_X_5 = StandardScaler().fit_transform(VoD_X0)
VoD_X_6 = StandardScaler().fit_transform(VoD_X0.T).T
```

**Rationale:**
- **Column-wise transformations** ($X_1, X_3, X_5$) normalize each feature independently, suitable when features have different scales.
- **Row-wise transformations** ($X_2, X_4, X_6$) normalize each sample, useful when the magnitude of observations varies significantly.

<div style="page-break-after: always;"></div>

---

<h2 style="font-size: 20px;">Question 3: Feature Selection and Correlation Analysis</h2>

<h3 style="font-size: 16px;">3.1 Methodology</h3>

Using Random Forest Regressor with 50 trees, we selected the top 18 most important features from the original 1,670 features. The correlation matrix was computed for these 18 features plus the target variable, resulting in a 19×19 matrix.

<h3 style="font-size: 16px;">3.2 Correlation Heatmap Analysis</h3>

<h4 style="font-size: 14px;">KV Dataset Correlation Heatmap</h4>

<p align="center">
<img src="task1_KV_correlation_heatmap.png" alt="KV Correlation Heatmap" width="400"/>
</p>

**Key Observations:**

1. High Multicollinearity: The map is overwhelmingly red. This indicates that most of the 18 selected features are strongly and positively correlated with each other. The high degree of correlation between features (multicollinearity) suggests that many features contain redundant information.
2. Outlier Features: There are a few distinct horizontal and vertical stripes of a lighter color (e.g., around feature indices 8, 10, 12), showing that these specific features have a low or negative correlation with the other features.
3. Target Correlations: The last row/column shows correlations between features and the target *ReadsAvg*. Several features exhibit strong positive correlations with target, indicating their high predictive value.

**Engineering Perspective:** The Key-Value store's read performance is closely tied to specific infrastructure bottlenecks. The high multicollinearity suggests over-instrumentation, where multiple sensors capture redundant information about the same underlying system state.

<!-- <div style="page-break-after: always;"></div> -->

<h4 style="font-size: 14px;">VoD Dataset Correlation Heatmap</h4>

<p align="center">
<img src="task1_VoD_correlation_heatmap.png" alt="VoD Correlation Heatmap" width="400"/>
</p>

**Key Observations:**

1. Complex Correlation Patterns: Unlike KV, the VoD heatmap displays a diverse mixture of strong positive (dark red), strong negative (dark blue), and near-zero (white) correlations.

2. Feature Independence: More white and blue regions indicate greater independence between features, suggesting the video streaming infrastructure has more diverse and decoupled monitoring points.

3. Mixed Target Relationships: Correlations with DispFrames vary widely—some features show positive relationships, others negative, and some near-zero. This indicates that video quality depends on multiple, sometimes opposing factors.

**Engineering Perspective:** Video frame rate is influenced by complex, multi-factor interactions. The presence of both positive and negative correlations suggests trade-offs in the system (e.g., higher encoding quality may reduce frame rate, or bandwidth allocation may have inverse relationships). This complexity implies that non-linear models may be necessary for accurate prediction.

**Comparative Insight:** The stark difference between KV (homogeneous, highly correlated) and VoD (heterogeneous, mixed correlations) reflects the fundamental architectural differences: KV operations are primarily compute-bound with predictable dependencies, while video streaming involves diverse resources with complex interdependencies.

<div style="page-break-after: always;"></div>

<h2 style="font-size: 20px;">Question 4: Joint Distribution Analysis</h2>

<h3 style="font-size: 16px;">4.1 Feature Selection for Comparison</h3>

For each dataset, we identified:
- **$F_h$:** Feature with highest absolute correlation to target (best predictor)
- **$F_l$:** Feature with lowest absolute correlation to target (worst predictor)

<h3 style="font-size: 16px;">4.2 KV Dataset Joint Distributions</h3>

<p align="center">
<img src="task1_KV_joint_distributions.png" alt="KV Joint Distributions" width="700"/>
</p>

<h4 style="font-size: 14px;">High Correlation Feature ($F_h$, Correlation: 0.754)</h4>

**Visual Pattern:** The plot reveals a clear, strong positive relationship between $F_h$ and the target. The concentration of density (dark blue regions) forms a diagonal pattern ascending from lower-left to upper-right.

**Interpretation:**
- As $F_h$ increases, ReadsAvg (response time) consistently increases
- The tight clustering along the trend line indicates low variance and high predictability
- This feature likely represents a direct performance bottleneck (e.g., CPU utilization, queue length, or concurrent request count)

**Engineering Insight:** This metric captures a resource that directly constrains read operations. The linear relationship suggests proportional scaling—doubling $F_h$ roughly doubles response time.

<h4 style="font-size: 14px;">Low Correlation Feature ($F_l$, Correlation: 0.219)</h4>

**Visual Pattern:** The plot shows a concentrated vertical blob with no horizontal trend. The data is tightly clustered at low $F_l$ values regardless of target values.

**Interpretation:**
- Changes in $F_l$ provide no information about ReadsAvg
- The target varies across its full range while $F_l$ remains nearly constant
- This feature has minimal variance and no predictive power

**Engineering Insight:** This metric likely monitors a non-critical or over-provisioned resource (e.g., disk I/O on a cache-serving system, or memory on a lightly-loaded node). Its lack of variation suggests it never becomes a bottleneck.

<div style="page-break-after: always;"></div>

<h3 style="font-size: 16px;">4.3 VoD Dataset Joint Distributions</h3>

<p align="center">
<img src="task1_VoD_joint_distributions.png" alt="VoD Joint Distributions" width="700"/>
</p>

<h4 style="font-size: 14px;">High Correlation Feature ($F_h$, Correlation: -0.464)</h4>

**Visual Pattern:** The plot exhibits a clear negative relationship with two distinct density clusters connected by a diagonal band sloping downward from left to right.

**Interpretation:**
- When $F_h$ is low (~4,000), DispFrames clusters near maximum (25 fps)
- When $F_h$ is high (~9,000), DispFrames drops to minimum (~13 fps)
- The inverse relationship suggests $F_h$ represents a cost or load metric

**Engineering Insight:** This feature likely measures resource consumption or system load (e.g., CPU usage, encoding complexity, or network congestion). As this metric increases, the system cannot maintain high frame rates, resulting in degraded video quality. The bimodal distribution suggests the system operates in two regimes: normal (high frame rate, low load) and degraded (low frame rate, high load).

<h4 style="font-size: 14px;">Low Correlation Feature ($F_l$, Correlation: -0.0349)</h4>

**Visual Pattern:** The plot shows two horizontal bands with no relationship to the x-axis. The density is split between high frame rates (~24 fps) and low frame rates (~13 fps), independent of $F_l$.

**Interpretation:**
- $F_l$ values span their range regardless of video quality
- The horizontal banding reflects the bimodal nature of DispFrames
- No predictive relationship exists between $F_l$ and target

**Engineering Insight:** This metric monitors a system component that is not involved in the video delivery pipeline's critical path (e.g., storage I/O on a fully-cached system, or memory usage in a non-bottleneck service). Its independence from frame rate confirms it can be excluded from prediction models.

<h3 style="font-size: 16px;">4.4 Comparative Summary</h3>

The joint distribution analysis validates the correlation-based feature selection:

| Aspect | High Correlation ($F_h$) | Low Correlation ($F_l$) |
|--------|-------------------------|------------------------|
| **Pattern** | Clear directional trend | Random scatter or independence |
| **Predictive Value** | High—can estimate target | None—provides no information |
| **Engineering Role** | Critical bottleneck resource | Non-critical or over-provisioned |
| **Model Utility** | Essential feature | Can be safely removed |

From an engineering perspective, $F_h$ represents key performance indicators directly impacting service quality, while $F_l$ corresponds to secondary metrics with negligible influence on user experience.

---

<div style="text-align: center; padding: 50px 0;">

<h2 style="font-size: 20px;">End of Task I Report</h2>

</div>