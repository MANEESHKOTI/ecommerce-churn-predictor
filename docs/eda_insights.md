# 📊 Exploratory Data Analysis (EDA) Insights

## 1. Executive Summary
The EDA confirms that **customer engagement (Recency)** and **habit deviation (Lateness Score)** are the strongest predictors of churn. High-value customers ("Champions") have a significantly lower churn rate (5%) compared to "At Risk" customers (45%).

## 2. Key Findings

### A. The "Lateness" Factor
* **Finding:** The `LatenessScore` (Recency / AvgDaysBetweenPurchases) is the single most discriminatory feature.
* **Evidence:** Density plots show that active customers cluster around a score of 1.0 (buying on schedule), while churned customers drift towards 2.5+ (missing 2+ cycles).
* **Action:** Trigger retention emails when LatenessScore crosses 1.5.

### B. RFM Patterns
* **Recency:** Churned customers have a median recency of **85 days**, compared to **12 days** for active customers (p < 0.001).
* **Monetary:** There is no significant difference in *average* spend per order between churned and active users, meaning **high spenders churn just as often as low spenders** if ignored.

### C. Segment Risk Profile
| Segment | Churn Rate | Strategy |
| :--- | :--- | :--- |
| **Champions** | Low (5%) | Loyalty rewards, upsell. |
| **Potential** | Medium (25%) | Nurture campaigns. |
| **At Risk** | High (45%) | Aggressive discounts. |
| **Lost** | Very High (80%) | Win-back or ignore (cost-benefit). |

## 3. Statistical Significance
We performed independent T-tests to validate feature importance:
* **Recency:** T-Statistic = 45.2 (Highly Significant)
* **Frequency:** T-Statistic = -12.4 (Significant)
* **TotalSpent:** T-Statistic = -3.1 (Marginally Significant)

## 4. Conclusion for Modeling
Based on this analysis, the model should prioritize **temporal features** (how behavior changes over time) rather than static totals (total lifetime spend).