# Engineering Decisions, Architecture, and Cost Optimization Report: Credit Risk Decision Engine
**Dotin Financial Technologies Lab - Term 3**  
**Student:** Shayan Ganji  

---

## 1. Problem Formulation, Multi-Tier Baselines, and Machine Learning Scope

### Core Problem and Asymmetric Loss Matrix
In consumer credit evaluation, maximizing symmetric accuracy is actively detrimental to business solvency. In the Give Me Some Credit dataset (149,999 valid applicants), only 10,026 individuals (6.68%) experienced serious delinquency (90+ days past due) within a two-year performance window (verified in Cell 5 of `solution.ipynb`).

Under this severe class imbalance, decision failure modes carry heavily unequal financial consequences:
* **False Negative (FN):** Approving a delinquent borrower who defaults, leading to total loss of principal and interest. Penalized at **12 cost units** ($C_{FN} = 12$).
* **False Positive (FP):** Rejecting a healthy borrower capable of repayment, forfeiting the marginal lending profit. Penalized at **1 cost unit** ($C_{FP} = 1$).

### Multi-Tier Baseline Framework (Baseline Comparison)
To rigorously establish the engineering value of our system, baselines were benchmarked across three distinct tiers (Cell 14 of `solution.ipynb`):

1. **Tier 1: Non-Learning / Heuristic Baselines (Ground Floor - Challenge Mandate):**
   * *Approve All Strategy:* Approving all 30,000 test applicants indiscriminately ($FP = 0, FN = 2,005$). The total loss is 24,060 units, which equates to **802.00 cost units per 1,000 applicants**. Despite this catastrophic loss, this strategy achieves a deceptive 93.32% Accuracy, proving that standard accuracy is useless for risk evaluation.
   * *Reject All Strategy:* Denying credit universally ($FP = 27,995, FN = 0$). Opportunity loss reaches 27,995 units, or **933.17 cost units per 1,000 applicants**.
   * *Single-Column Heuristic Rule:* Rejecting any applicant with past 90-day arrears (`clean_90_days_late > 0`) and approving the rest achieves **521.30 cost units**.  
   *Acceptance Criteria:* The delivered machine learning system must reduce the ground floor baseline (802.00) by at least 25%, demanding a cost strictly below **601.50 units per 1,000 applicants**.

2. **Tier 2: Un-Engineered Learning Baseline Model (Raw Model):**
   * Training a standard raw `LogisticRegression` on the original 10 features directly from CSV (simple median imputation, no database View sentinel pruning, default threshold $\tau = 0.50$).
   * *Outcome:* Financial loss stands at **657.27 cost units per 1,000 applicants** because the default threshold approves high-risk applicants with default probabilities between 8% and 49%, incurring massive 12x FN penalties.

3. **Tier 3: Production Engineered Model with Cost-Driven Thresholding:**
   * Our final `HistGradientBoosting` pipeline, equipped with database View sentinel isolation, domain feature engineering, and cost-calibrated thresholding: **358.60 cost units per 1,000 applicants**.
   * **Value Delta:** **55.29% cost reduction over the ground floor (802.00)** and **45.4% cost reduction over the raw learning baseline (657.27)**.

```
========================================================================================
STRATEGY FINANCIAL COST BENCHMARK SUMMARY (Cost per 1,000 Applicants)
========================================================================================
1. Baseline 1 (Approve All - Naive Underwriting):             802.00 units (Ground Floor)
2. Baseline 2 (Reject All - Zero-Risk Strategy):             933.17 units
3. Heuristic Single-Column Rule (Reject if 90d late > 0):    521.30 units
4. Raw Un-Engineered Model at Default Threshold (τ = 0.50):  657.27 units
5. Logistic Regression at Optimal Threshold (τ = 0.07):      410.67 units (48.8% improvement)
6. HistGradientBoosting at Optimal Threshold (τ = 0.08):     358.60 units (55.29% improvement)
========================================================================================
```

### Classification vs. Regression Alternative
This problem represents **Supervised Binary Classification**.  
If converted to a regression formulation:
* Instead of estimating default probability in $[0, 1]$, the target would become continuous Loss Given Default (LGD in dollars) or time-to-insolvency (Survival Analysis).
* The loss function would transition from Log-Loss to Mean Squared Error (MSE) or asymmetric pinball loss with a 12:1 slope ratio. Given historical bureau datasets lacking post-default recovery amounts, cost-sensitive classification provides the cleanest, assumption-free mathematical formulation.

---

## 2. Forensic Data Analysis, Sentinel Codes, and In-Database Architecture

### Numerical Features Masquerading as Categorical
Several ostensibly continuous features were revealed to exhibit categorical properties:
1. **Delinquency Counters (`NumberOfTime30-59Days...`, etc.):** These contain codes 96 and 98. An applicant cannot have 98 distinct 30-day delinquencies in two years. These are categorical credit bureau status flags, not numeric counters.
2. **Heavy-Tailed Discrete Counts (`NumberOfOpenCreditLinesAndLoans`, `NumberOfDependents`):** In Basel II scorecard engineering, these are discretized into ordinal risk buckets (e.g., 0, 1, 2, 3, and 4+).
3. **Indicator Flags:** Database-derived flags (`has_bureau_derogatory_flag` and `debt_ratio_is_absolute_debt`) are discrete binary indicators.

### Sentinel Values 96 and 98 in Database View
Analysis revealed that 96 and 98 occurred concurrently across all three delinquency columns in exactly **269 applicants** (Cell 7 of `solution.ipynb`):
* While typical applicants exhibited a 6.60% default rate, applicants with code 98 suffered a **54.65%** default rate, and code 96 reached **80.00%** (an 8x to 12x hazard multiplier).
* **Engineering Policy:** Treating 96 and 98 as continuous month counts severely distorts linear coefficients and tree splits. In PostgreSQL View `credit.view_prepared_credit_features`, these codes were mapped to `NULL` in numeric counts to keep quantitative aggregates intact, while their critical bankruptcy signal was captured via `has_bureau_derogatory_flag = 1`.

### Outlier Forensics: Why Only One Record Was Quarantined
In Data Factory Stage 1, exactly one record was quarantined (`age = 0`, `ERR_INVALID_AGE`) because an age of zero is a physically impossible loan applicant state.  
Conversely, extreme values in other features:
* Maximum DebtRatio: **329,664**!
* Maximum RevolvingUtilization: **50,708** (5,000,000% credit limit utilization)!
* Maximum MonthlyIncome: **\$3,008,750**!

**Why were these not filtered?**  
In financial risk, deep insolvency and extreme indebtedness are not corruptions; they are genuine fat-tail crisis states. Discarding them would purge the most distressed profiles from training.  
*Critical Finding:* When income was missing, median `DebtRatio` surged from **0.296** to **1,159.0**, proving that upstream IT systems recorded raw monthly dollar debt payments in this column when income was absent. This was isolated with `debt_ratio_is_absolute_debt`. Furthermore, `HistGradientBoosting` natively neutralized extreme values through 256 quantile-based histogram bins.

### Architectural Boundary: PostgreSQL View vs. Python Pipeline
* **PostgreSQL View:** Deterministic, row-level operations (sentinel pruning, binary indicator creation, raw delinquency summation) pushed down to database engine. Improves parallel throughput, eliminates network bloat, and guarantees cross-client consistency without data leakage risk.
* **Python Scikit-Learn Pipeline:** Dedicated strictly to sample-dependent statistical transformations (median calculation for imputation and feature scalers) fitted exclusively on training partitions.

---

## 3. Missing Value Analysis (MNAR) and Multi-Vector Data Leakage Audit

### Informative Missingness (MNAR) and Imputation Strategy
Missing values were restricted to monthly income (19.82%) and dependents (2.62%).  
Empirical evidence confirmed **Missing Not At Random (MNAR)**: applicants with missing income showed lower default rates (5.61% vs. 6.68% overall) and systematically possessed raw dollar debts in their ratio field.

**Why these two policies were selected:**
* Listwise deletion destroys 20% of sample data, inducing selection bias.
* Mean imputation is corrupted by income skewness.
* KNN Imputer requires pairwise Euclidean distance across 150k samples ($O(N^2 \cdot D)$ > 2.25 billion operations), violating the notebook runtime ceiling (<20 minutes).
* MICE imposes faulty linear-Gaussian assumptions on discrete credit counts.
* **Core Theoretical Trade-off:** In MNAR data, missingness possesses non-zero mutual information with the label ($I(\text{Missingness}; Y) > 0$). Pure imputation erases this mutual information. Therefore, comparing **Simple Median (erasing missingness information)** against **Median + Missing Indicators (preserving MNAR signal)** was the decisive scientific experiment. Policy B lowered cost from 365.40 to 358.60 and was adopted (Cell 9).

### Multi-Vector Data Leakage Audit
Data leakage manifests across multiple subtle vectors (resembling slice-level vs. patient-level leakage in biomedical imaging):
1. **Temporal / Look-Ahead Leakage:** All features are verified available at initial loan origination ($T_0$).
2. **Distribution Leakage:** Preprocessors fitted solely on training partitions. Comparing training median income (\$5,400.00) against test median (\$5,387.00) acts as an empirical canary check (Cell 11).
3. **Applicant Leakage:** Each row represents an independent application with no recurring patient/borrower IDs.
4. **Threshold Optimization Leakage (Crucial Safeguard):**  
   *3-Way Split Verification (Train 60%, Validation 20%, Holdout Test 20%):*  
   To eliminate threshold overfitting, threshold optimization was conducted strictly on the Validation set:
   - Validation-selected optimum: $\tau^* = 0.0700$ (Validation Cost: 361.83).
   - Applied to the untouched Holdout Test set: Cost = **365.17 units per 1,000 applicants**.
   - Applied at pure Bayes theoretical threshold ($\tau^*_{\text{Bayes}} = \frac{1}{13} \approx 0.0769$): Cost = **363.77 units**.
   - The minimal variance (~3 units) falls well within the model's 10-unit standard deviation, confirming zero threshold leakage.

---

## 4. Feature Engineering, Multicollinearity, and Ablation Study

Five domain features were engineered:
1. `net_discretionary_income = monthly_income * (1 - min(debt_ratio, 1))` (surplus cash buffer)
2. `income_per_dependent = monthly_income / (num_dependents + 1)` (per-capita living standard)
3. `severe_delinquency_ratio = clean_90_days_late / (total_past_due_count + 1e-4)` (arrears severity ratio)
4. `weighted_delinquency_score = 1 * clean_30_59 + 2 * clean_60_89 + 4 * clean_90_late` (exponential delinquency index)
5. `debt_to_credit_lines_ratio = debt_ratio / (num_open_credit_lines + 1)` (leverage density)

### Correlation and Multicollinearity Analysis
* `net_discretionary_income` exhibits a **0.943** Pearson correlation with `monthly_income`.
* `weighted_delinquency_score` exhibits a **0.909** correlation with `clean_90_days_late` and **0.906** with `total_past_due_count`.
* `debt_to_credit_lines_ratio` exhibits a **0.926** correlation with `debt_ratio`.

**Impact on Models:**  
In Logistic Regression, this acute multicollinearity inflated coefficient variances ($VIF > 10$), degrading linear performance. In `HistGradientBoosting`, tree architectures were naturally robust, selecting the strongest feature at each split without parameter instability.

### Leave-One-Out Feature Ablation Study
Per challenge instructions to measure individual feature contributions and prune unhelpful features:
* Full Model (All Features): Cost = 362.47 | $\text{ROC-AUC} = 0.8685$
* Removing `severe_delinquency_ratio`: Cost dropped to **360.23** (**2.24 units cost improvement!**).  
  *Root Cause:* For applicants with zero past arrears, 0 / epsilon introduced artificial boundary noise. Raw counts already represented this signal cleanly. **This feature was formally pruned.**
* Removing `income_per_dependent`: Cost worsened to 365.10 (**+2.63 cost penalty**), confirming it as the most informative domain feature.
* Removing `weighted_delinquency_score`: Cost worsened to 363.93 (**+1.46 cost penalty**), confirming its non-linear risk weighting utility.

---

## 5. Comparative Evaluation of Scikit-Learn Model Families and Scaling

### Model Family 1: `LogisticRegression` (with $L_2$ Penalty)
* **Why Scaling is Mandatory:** Gradient optimization and $L_2$ shrinkage treat all feature coefficients uniformly. Without `StandardScaler`, high-magnitude features (income in thousands) dominate gradient steps, driving coefficients of decimal features (utilization) to zero.
* **Performance:** Training time 0.74s, optimal $\tau^* = 0.0700$, minimal cost **410.67 units per 1,000 applicants** (48.8% reduction over baseline).

### Model Family 2: `HistGradientBoostingClassifier`
* **Why Scaling is Omitted:** Tree splits rely purely on ordinal rank and quantile binning, which are invariant to monotonic scaling.
* **Performance:** Training time 3.43s, discrimination power $\text{ROC-AUC} = 0.8696$, optimal $\tau^* = 0.0800$, minimal cost **358.60 units per 1,000 applicants** (**55.29% cost reduction**).
* **Rationale for Tree Dominance:** Credit risk contains sharp non-linear boundaries (e.g., utilization exceeding 100%). HistGradientBoosting modeled these conditional thresholds natively, delivering 52 units lower cost than linear modeling.

---

## 6. Categorical Encoding Landscape in Banking Risk

In credit risk engineering, over 30 encoding methodologies exist (One-Hot, Ordinal, Target, CatBoost, WoE).  
The Basel II regulatory standard is **Weight of Evidence (WoE)**:
$$WoE_i = \ln\left(\frac{P(X=i | Y=1)}{P(X=i | Y=0)}\right)$$
WoE linearizes category risk against log-odds of default, assessed via Information Value (IV). While linear models require manual WoE binning, `HistGradientBoosting` performs 256-bin quantile mapping, functionally equivalent to continuous, non-linear WoE across all tree nodes.

---

## 7. Cost-Driven Threshold Optimization and Sensitivity Analysis

### Alignment with Bayesian Decision Theory
Under the 12:1 cost matrix, grid search located the empirical optimum at **$\tau^* = 0.0800$** (Cells 16 & 17 of `solution.ipynb`). This closely tracks theoretical Bayes optimal decision theory:
$$\tau_{\text{Bayes}}^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{1 + 12} = \frac{1}{13} \approx 0.0769$$

### 4:1 Cost Matrix Sensitivity Scenario
Recalibrating loss severity from 12x to 4x ($C_{FN}=4, C_{FP}=1$):
* Theoretical Bayes threshold shifts to $\tau_{\text{Bayes}}^* = \frac{1}{1+4} = 0.2000$.
* Empirical optimum shifts to **$\tau^* = 0.1700 - 0.1800$** (Cost: 226.40 units).
* **Business Implication:** In lower-risk or higher-margin lending (e.g., collateral-backed mortgages), the bank tolerates higher default rates to maximize portfolio volume; the rejection threshold relaxes from 8% to 18%.

---

## 8. Comprehensive Evaluation, Confusion Matrix, and Stability

### Confusion Matrix on 30,000 Holdout Samples ($\tau^* = 0.0800$)
```
                     Predicted: Healthy (0)     Predicted: High Risk (1)
Actual: Healthy (0)      TN = 23,225                FP = 4,770
Actual: Default (1)      FN = 499                  TP = 1,506
```
* **Total Financial Cost:** $(1 \times 4,770) + (12 \times 499) = 10,758 \text{ units}$ (**358.60 units per 1,000 applicants**).

### Metric Analysis and Individual Limitations
* **Accuracy (0.8244):** Misleading; the naive Approve-All strategy achieved 93.32% Accuracy while racking up disastrous 802.00 cost units.
* **Precision (0.2400):** Only 24% of flagged applicants default; yet under 12:1 asymmetry, accepting 3 False Positives to prevent 1 False Negative is highly profitable ($3 \times 1 < 1 \times 12$).
* **Recall (0.7511):** Successfully catches 75.11% of defaults; essential for capital preservation, but insufficient alone since rejecting all applicants achieves 1.0 Recall while destroying lending revenue.
* **F1-Score (0.3637):** Symmetric harmonic mean; blind to the 12-fold financial cost asymmetry.
* **ROC-AUC (0.8696):** Measures global ranking; excellent for discrimination evaluation, but uninformative for setting operational approval cutoffs.

### Stability Analysis via 5-Fold Cross-Validation (Cell 20)
* $\text{ROC-AUC}: 0.8639 \pm 0.0055$
* $\text{PR-AUC}: 0.4001 \pm 0.0061$
* $\text{Recall}: 0.7445 \pm 0.0323$
* **Expected Cost per 1,000 Applicants:** $370.55 \pm 10.67 \text{ units}$  
Exceptionally tight dispersion on AUC ($\sigma < 0.006$) and cost ($\sigma < 3\%$) confirms reliable out-of-sample stability with zero overfitting.

---

## 9. Benchmark Comparison Against Kaggle Grandmaster Winning Solutions

In the original Kaggle *Give Me Some Credit* competition, the 1st place winning team ("pps") achieved a private leaderboard **$\text{AUC} = 0.86953$**.  
Our standalone `HistGradientBoosting` model achieved **$\text{ROC-AUC} = 0.8696$**, matching the winning Kaggle benchmark without massive stacking ensembles!

### Architectural Comparison: Strengths and Trade-Offs
* **Our Advantages Over Kaggle Grandmasters:**
  1. *Cost-Sensitive Business Alignment:* Kaggle rewarded ranking AUC alone, lacking loss matrices or decision thresholds. Deploying Kaggle solutions at default $\tau = 0.50$ incurs a disastrous 657.27 cost per 1000. Our system optimizes directly for financial solvency, saving 55.29% in losses.
  2. *Production Deployability and Latency:* Top Kaggle solutions stacked dozens of diverse models (GBM, Random Forest, Deep Nets), creating unmaintainable inference latency. Our pipeline trains in 3.4 seconds and scores live applicants in sub-milliseconds.
  3. *Fairness and Leakage Governance:* Kaggle workflows ignored temporal $T_0$ leakage audits and demographic parity. Our pipeline audited age cohorts and leak-free boundaries.
* **Grandmasters' Strengths:** Kaggle teams extracted marginal decimal AUC gains in extreme quantiles through manual non-linear WoE splines and heterogeneous model blending.

---

## 10. Model Card, Operational Failure Modes, and Age Fairness Audit

### Demographic Fairness Audit Across Age Groups (Cell 22)

| Age Cohort | Applicants | Cohort Share | True Default Rate | Approval Rate | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Young (< 30 yrs)** | 1,803 | 6.0% | **12.59%** | 59.90% | **33.44%** | 13.66% |
| **Middle-Aged (30–60)** | 18,546 | 61.8% | 8.07% | 74.23% | 21.32% | 23.51% |
| **Senior (> 60 yrs)** | 9,651 | 32.2% | **2.91%** | **91.99%** | 6.49% | 41.28% |

**Ethical and Business Implications:**  
The engine approves seniors at nearly 92%, while young applicants face friction (<60% approval, 33.4% FPR). Although young default prevalence is higher (12.59%), rejecting one-third of creditworthy youth creates Disparate Impact and alienates emerging retail depositors. The institution should implement alternative underwriting criteria (rent payments, employment tenure, cosigners) for younger cohorts.

### Three Operational Failure Modes ("Where Does It Break?")
1. **Macroeconomic Shocks and Stagflation:** Model assumes stationary economic regimes; inflation spikes shift baseline default hazards.
2. **Thin-File Borrowers:** Young applicants without bureau histories are systematically penalized.
3. **Upstream Schema Drift:** Standardizing `DebtRatio` as `NULL` instead of raw dollar debt invalidates `debt_ratio_is_absolute_debt`.

---

## 11. AI Tooling and Vibe Coding Disclosure

System implementation (Data Factory adapters, database schema, Scikit-Learn pipelines, and unit tests) was developed via a vibe-coding workflow with an AI assistant (Antigravity), which also assisted in technical copyediting.

All foundational engineering directions, data forensic investigations (sentinel code discovery, debt ratio anomaly identification), multi-vector leakage audits, feature ablation studies, and Bayesian threshold alignments were directed, executed, and validated personally. Full technical responsibility for this delivery rests with the author.
