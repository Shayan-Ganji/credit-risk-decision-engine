# Engineering Decisions, Architecture, and Cost Optimization Report: Credit Risk Decision Engine
**Dotin Financial Technologies Lab - Term 3**  
**Student:** Shayan Ganji  

---

## 1. Problem Formulation, Multi-Tier Baselines, and Machine Learning Scope

### Core Problem and Asymmetric Loss Matrix
In consumer credit evaluation, maximizing symmetric accuracy is actively detrimental to business solvency. In the credit dataset comprising 149,999 valid applicants, only 10,026 individuals (6.68%) experienced serious delinquency (90+ days past due) within a two-year performance window (verified in Cell 5 of `solution.ipynb`).

Under this severe class imbalance, decision failure modes carry heavily unequal financial consequences:
* **False Negative (FN):** Approving a delinquent borrower who defaults, leading to total write-off of principal and interest. Penalized at **12 cost units** ($C_{FN} = 12$).
* **False Positive (FP):** Rejecting a healthy borrower capable of repayment, forfeiting marginal lending profit. Penalized at **1 cost unit** ($C_{FP} = 1$).

### Multi-Tier Baseline Framework
To establish the operational impact of our engineering pipeline, performance was evaluated across three distinct baseline tiers (Cell 14 of `solution.ipynb`):

1. **Tier 1: Non-Learning / Heuristic Baselines (Ground Floor):**
   * *Approve All Strategy:* Approving all 30,000 holdout test applicants indiscriminately ($FP = 0, FN = 2,005$). Total financial penalty reaches 24,060 units, which equates to **802.00 cost units per 1,000 applicants**. Despite this heavy loss, this naive strategy achieves a deceptive 93.32% Accuracy, confirming that standard accuracy is completely unsuited for imbalanced risk evaluation.
   * *Reject All Strategy:* Denying credit universally ($FP = 27,995, FN = 0$). Opportunity loss reaches 27,995 units, translating to **933.17 cost units per 1,000 applicants**.
   * *Single-Column Heuristic Rule:* Rejecting any applicant with past 90-day arrears (`clean_90_days_late > 0`) and approving remaining applicants yields **521.30 cost units**.  
   *Acceptance Ceiling:* The candidate machine learning system must reduce the ground floor baseline (802.00) by at least 25%, requiring an expected cost strictly below **601.50 units per 1,000 applicants**.

2. **Tier 2: Basic Learning Baseline Without Feature Engineering:**
   * Training a standard Logistic Regression model on the original 10 features directly loaded from the database (simple median imputation, no database View sentinel pruning, default threshold $\tau = 0.50$).
   * *Outcome:* Financial loss stands at **657.27 cost units per 1,000 applicants**, as the default 0.50 cutoff approves applicants with default probabilities between 8% and 49%, incurring severe 12x FN penalties.

3. **Tier 3: Production Engineered Model with Cost-Driven Thresholding:**
   * Our final `HistGradientBoosting` pipeline, integrated with database View sentinel isolation, domain feature engineering, and cost-calibrated thresholding: **358.60 cost units per 1,000 applicants**.
   * **Value Delta:** **55.29% cost reduction over the ground floor (802.00)** and **45.4% cost reduction over the basic learning baseline (657.27)**.

```
========================================================================================
STRATEGY FINANCIAL COST BENCHMARK SUMMARY (Cost per 1,000 Applicants)
========================================================================================
1. Baseline 1 (Approve All - Naive Underwriting):             802.00 units (Ground Floor)
2. Baseline 2 (Reject All - Zero-Risk Strategy):             933.17 units
3. Heuristic Single-Column Rule (Reject if 90d late > 0):    521.30 units
4. Basic Learning Model at Default Threshold (τ = 0.50):     657.27 units
5. Logistic Regression at Optimal Threshold (τ = 0.07):      410.67 units (48.8% improvement)
6. HistGradientBoosting at Optimal Threshold (τ = 0.08):     358.60 units (55.29% improvement)
========================================================================================
```

### Classification vs. Regression Formulation
This task is formulated as **Supervised Binary Classification**.  
If converted to a regression problem, the target would become continuous Loss Given Default (LGD in dollars) or time-to-insolvency (Survival Analysis). The loss function would transition from Log-Loss to quantile/pinball regression with an asymmetric 12:1 slope ratio. Given historical datasets that track binary delinquency states rather than recovery amounts, cost-sensitive classification provides the most direct and mathematically sound formulation.

---

## 2. Data Structure, Sentinel Codes, Extreme Outliers, and Database Architecture

### Numerical Features Masquerading as Categorical
Initial exploratory analysis revealed that several ostensibly continuous features exhibit categorical properties:
1. **Delinquency Counters (`NumberOfTime30-59Days...`, etc.):** These contain anomalous codes 96 and 98. In credit bureau conventions, these are categorical condition codes representing judicial collection or administrative repossession, not numeric day counts.
2. **Heavy-Tailed Discrete Counts (`NumberOfOpenCreditLinesAndLoans`, `NumberOfDependents`):** In retail credit scoring, these discrete counts behave ordinally and are traditionally analyzed in risk brackets.
3. **Derived Binary Indicators:** Flags generated in the database (`has_bureau_derogatory_flag` and `debt_ratio_is_absolute_debt`) represent qualitative binary states.

### Sentinel Values 96 and 98 in Database View
Statistical inspection revealed that codes 96 and 98 appeared concurrently across all three delinquency columns in exactly **269 applicants** (Cell 7 of `solution.ipynb`):
* While typical applicants exhibited a 6.60% default rate, applicants with code 98 suffered a **54.65%** default rate, and code 96 reached **80.00%** (an 8x to 12x hazard multiplier).
* **Engineering Policy:** Treating 96 and 98 as continuous counts severely distorts linear coefficients and tree splits. In PostgreSQL View `credit.view_prepared_credit_features`, these codes were mapped to `NULL` in numeric counts to keep quantitative statistics intact, while their critical bankruptcy signal was captured via `has_bureau_derogatory_flag = 1`.

### Outlier Handling and System Anomaly Isolation
During initial ingestion, exactly one record was quarantined due to an age of zero (`ERR_INVALID_AGE`), representing a physically impossible applicant state.  
In other features, extreme values were observed (DebtRatio up to 329,664 and RevolvingUtilization up to 50,708). In credit risk management, severe indebtedness or over-limit cards are not systemic corruptions; they represent genuine fat-tail distress events, and deleting them would remove the most critical high-risk profiles.  

Cross-tabulation revealed that for records with missing income, the median DebtRatio jumped from **0.296** to **1,159.0**, proving that when income was unrecorded, upstream banking systems stored raw monthly dollar debt payments in this field. This discrepancy was isolated via `debt_ratio_is_absolute_debt`. Furthermore, `HistGradientBoosting` natively neutralized extreme values through 256 quantile-based histogram bins.

### Architectural Boundary: PostgreSQL View vs. Python Pipeline
* **PostgreSQL View:** Deterministic, row-level operations (sentinel pruning, binary indicator creation, raw delinquency summation) were pushed down to the database engine (Cell 3 of `solution.ipynb`). This optimizes network bandwidth and enforces unified business rules across consumers without introducing cross-row distribution leakage.
* **Python Scikit-Learn Pipeline:** Dedicated strictly to sample-dependent statistical transformations (such as median calculation for imputation and feature scalers) fitted exclusively on training partitions.

---

## 3. Missing Value Analysis (MNAR) and Multi-Vector Data Leakage Audit

### Informative Missingness (MNAR) and Imputation Policy
Missing values were confined to monthly income (19.82%) and dependents (2.62%).  
Empirical analysis confirmed **Missing Not At Random (MNAR)**: applicants without recorded income showed distinct default rates (5.61%) compared to the overall population, and their debt ratios reflected monthly dollar liabilities, indicating informal employment or non-disclosure.

Listwise deletion was rejected to prevent discarding 20% of data and inducing selection bias. Mean imputation was unsuitable due to acute income skewness, and distance-based methods like KNN were computationally prohibitive on 150,000 records within notebook time budgets.  
Because missingness carries mutual information with the label ($I(\text{Missingness}; Y) > 0$), two policies were evaluated empirically (Cell 9 of `solution.ipynb`):
* **Policy A (Simple Median Imputation):** Missing values filled with median, yielding 365.40 cost units.
* **Policy B (Median Imputation with Missingness Indicators):** Preserving binary missingness indicators alongside median filling, which lowered expected cost to **358.60 units per 1,000 applicants** and was selected as production standard.

### Multi-Vector Data Leakage Audit
Data leakage safeguards were verified across multiple vectors:
1. **Temporal Availability:** All features reflect verified data available at loan application time ($T_0$), with zero contamination from the two-year outcome window.
2. **Distribution Leakage:** Data partitioning (80/20) was executed prior to fitting transformers. Comparing training median income (\$5,400.00) against holdout test median (\$5,387.00) verified that evaluation partitions remained unobserved during fitting (Cell 11).
3. **Threshold Optimization Verification (3-Way Split):**  
   To prevent threshold overfitting, an experimental 3-way split was conducted (Sub-Train 60%, Validation 20%, Holdout Test 20%):
   - The optimal threshold was identified strictly on Validation: $\tau^* = 0.0700$ (Validation Cost: 361.83).
   - Applied to the untouched Holdout Test set, cost settled at **365.17 units per 1,000 applicants**.
   - Applying the theoretical Bayes threshold ($\tau^*_{\text{Bayes}} = \frac{1}{13} \approx 0.0769$) yielded **363.77 units**.  
   This tight consistency confirms that the threshold generalizes reliably without leakage.

---

## 4. Feature Engineering, Multicollinearity, and Ablation Study

Five domain features were engineered based on financial principles:
1. `net_discretionary_income = monthly_income * (1 - min(debt_ratio, 1))` (surplus cash buffer)
2. `income_per_dependent = monthly_income / (num_dependents + 1)` (per-capita living standard)
3. `severe_delinquency_ratio = clean_90_days_late / (total_past_due_count + 1e-4)` (arrears severity ratio)
4. `weighted_delinquency_score = 1 * clean_30_59 + 2 * clean_60_89 + 4 * clean_90_late` (exponential delinquency index)
5. `debt_to_credit_lines_ratio = debt_ratio / (num_open_credit_lines + 1)` (leverage density)

### Correlation and Multicollinearity Analysis
Pearson correlation assessment demonstrated:
* `net_discretionary_income` exhibits a **0.943** correlation with `monthly_income`.
* `weighted_delinquency_score` exhibits a **0.909** correlation with `clean_90_days_late` and **0.906** with `total_past_due_count`.
* `debt_to_credit_lines_ratio` exhibits a **0.926** correlation with `debt_ratio`.

In Logistic Regression, this acute multicollinearity inflated coefficient variances ($VIF > 10$), constraining linear performance. In `HistGradientBoosting`, tree splits naturally accommodated collinear features by selecting the optimal split candidate without parameter degradation.

### Leave-One-Out Feature Ablation Study
To measure individual feature contributions, a Leave-One-Out ablation experiment was conducted:
* Full Model (All Features): Cost = 362.47 | $\text{ROC-AUC} = 0.8685$
* **Removing `severe_delinquency_ratio`:** Cost improved to **360.23** (**2.24 units cost improvement**).  
  Dividing by epsilon in zero-delinquency cases created artificial boundary noise, while base delinquency counts already captured the underlying signal; this feature was consequently pruned.
* **Removing `income_per_dependent`:** Cost worsened to **365.10** (**+2.63 cost penalty**), confirming its strong discriminatory power in isolating vulnerable households and warranting retention.
* **Removing `weighted_delinquency_score`:** Cost worsened to **363.93** (**+1.46 cost penalty**), confirming the validity of exponential weighting.

---

## 5. Comparative Evaluation of Scikit-Learn Model Families and Scaling

### Model Family 1: `LogisticRegression` (with $L_2$ Regularization)
* **Scaling Requirement Analysis:** Gradient descent optimization and uniform $L_2$ shrinkage require normalized feature scales. Without `StandardScaler`, high-magnitude features (income in thousands) dominated gradients, driving decimal coefficients (utilization) toward zero. Feature scaling was therefore **mandatory**.
* **Performance:** Training duration 0.74s, optimal threshold $\tau^* = 0.0700$, expected cost **410.67 units per 1,000 applicants**.

### Model Family 2: `HistGradientBoostingClassifier`
* **Scaling Requirement Analysis:** Tree-based partitioning relies on ordinal ranking and quantile histogram binning, which are invariant to monotonic scaling. Scaling was intentionally omitted, eliminating unnecessary computational overhead.
* **Performance:** Training duration 3.43s, discrimination power $\text{ROC-AUC} = 0.8696$, optimal threshold $\tau^* = 0.0800$, expected cost **358.60 units per 1,000 applicants** (**55.29% cost reduction** over baseline).
* **Tree Architecture Rationale:** Retail credit risk contains sharp non-linear boundaries (such as risk escalation when utilization exceeds 100%). HistGradientBoosting captured these interactions natively within tree nodes, delivering 52 units lower cost than linear modeling.

---

## 6. Cost-Driven Threshold Optimization and Sensitivity Analysis

### Alignment with Bayesian Decision Theory
Classifiers output continuous probabilities. Applying the default library threshold ($\tau = 0.50$) assumes equal error costs, resulting in an institutional loss of **657.27 units per 1,000 applicants**.  
Evaluating the 12:1 cost matrix through grid search established the empirical optimum at **$\tau^* = 0.0800$** (Cells 16 and 17 of `solution.ipynb`), tracking theoretical Bayesian decision theory:
$$\tau_{\text{Bayes}}^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{1 + 12} = \frac{1}{13} \approx 0.0769$$

### 4:1 Cost Matrix Sensitivity Scenario
Recalibrating loss severity from 12x to 4x ($C_{FN}=4, C_{FP}=1$):
* Theoretical Bayes threshold shifts to $\tau_{\text{Bayes}}^* = \frac{1}{1+4} = 0.2000$.
* Empirical grid search established the revised optimum at **$\tau^* = 0.1700 - 0.1800$** (Cost: 226.40 units).
* **Commercial Interpretation:** When default write-offs carry lower severity (e.g., in secured lending or higher-margin products), the institution tolerates higher default volume to expand portfolio acquisition; the rejection threshold relaxes from 8% to 18%.

---

## 7. Comprehensive Model Evaluation, Confusion Matrix, and Stability

### Confusion Matrix on 30,000 Holdout Samples ($\tau^* = 0.0800$)
Evaluation on untouched test partitions yielded the following absolute counts (Cell 19 of `solution.ipynb`):
```
                     Predicted: Healthy (0)     Predicted: High Risk (1)
Actual: Healthy (0)      TN = 23,225                FP = 4,770
Actual: Default (1)      FN = 499                  TP = 1,506
```
* **Total Financial Cost:** $(1 \times 4,770) + (12 \times 499) = 10,758 \text{ units}$ (**358.60 units per 1,000 applicants**).

### Metric Analysis and Limitations
* **Accuracy (0.8244):** Reduced from naive 93.32% baseline because 4,770 healthy applicants are rejected to avert defaults; this decline is economically justified by overall loss minimization.
* **Precision (0.2400):** Only 24% of flagged applicants default; yet under 12:1 asymmetry, accepting 3 False Positives to prevent 1 False Negative is profitable ($3 \times 1 < 1 \times 12$).
* **Recall (0.7511):** Successfully intercepts 75.11% of defaults; essential for capital preservation, but insufficient alone since blanket rejections achieve 1.0 Recall while eliminating lending volume.
* **F1-Score (0.3637):** Symmetric harmonic mean; unsuited for banking because it is blind to the 12-fold financial loss asymmetry.
* **ROC-AUC (0.8696):** Measures ranking quality across thresholds, but provides no guidance on setting the actual operational decision boundary.

### Stability Analysis via 5-Fold Cross-Validation (Cell 20)
* $\text{ROC-AUC}: 0.8639 \pm 0.0055$
* $\text{PR-AUC}: 0.4001 \pm 0.0061$
* $\text{Recall}: 0.7445 \pm 0.0323$
* **Expected Cost per 1,000 Applicants:** $370.55 \pm 10.67 \text{ units}$  
Tight standard deviation on AUC (<0.006) and close alignment with test holdout performance ($358.60$) confirm reliable out-of-sample stability with zero overfitting.

---

## 8. Model Card, Operational Failure Modes, and Demographic Fairness Audit

### Demographic Fairness Audit Across Age Groups (Cell 22)

| Age Cohort | Applicants | Cohort Share | True Default Rate | Approval Rate | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Young (< 30 yrs)** | 1,803 | 6.0% | **12.59%** | 59.90% | **33.44%** | 13.66% |
| **Middle-Aged (30–60)** | 18,546 | 61.8% | 8.07% | 74.23% | 21.32% | 23.51% |
| **Senior (> 60 yrs)** | 9,651 | 32.2% | **2.91%** | **91.99%** | 6.49% | 41.28% |

**Ethical and Commercial Implications:**  
The engine approves seniors at nearly 92%, while young applicants face friction (<60% approval, 33.4% FPR). While young default prevalence is higher (12.59%), rejecting one-third of creditworthy youth creates Disparate Impact and alienates emerging retail depositors. The institution should implement alternative underwriting criteria (rent payments, employment tenure, cosigners) for younger cohorts.

### Operational Failure Modes and System Boundaries
1. **Macroeconomic Shocks and Stagflation:** Model assumes stationary economic conditions; sharp inflation spikes shift baseline default hazards.
2. **Thin-File Borrowers:** Young applicants without bureau histories are systematically penalized due to absence of credit data.
3. **Upstream Schema Drift:** Standardizing `DebtRatio` as `NULL` instead of storing raw dollar debt invalidates the logic powering `debt_ratio_is_absolute_debt`.

---

## 9. Software Engineering, Executable Artifacts, and CLI

* **Modular Python Package:** Production modeling logic is encapsulated within `src/credit_risk/`, reading records via SQLAlchemy from PostgreSQL rather than raw flat files.
* **Model Artifact Serialization:** The fitted pipeline bundle, including the calibrated decision threshold ($\tau^* = 0.0800$), is persisted in `data/curated/models/credit_decision_pipeline.joblib`.
* **Command-Line Interface (CLI):** Implemented via `python -m credit_risk.cli`, supporting `--train-and-evaluate` for model training and `--predict <path>` for batch applicant scoring.
* **Offline Unit Test Suite:** 10 offline tests in `tests/test_credit_risk.py` validate mathematical formulas, data leakage safeguards, and pipeline execution in 0.30s.

---

## 10. AI Tooling and Vibe Coding Disclosure

System implementation (Data Factory adapters, database schema, Scikit-Learn pipelines, and unit tests) was developed via a vibe-coding workflow with an AI assistant (Antigravity), which also assisted in technical copyediting.

All foundational engineering directions, data forensic investigations (sentinel code discovery, debt ratio anomaly identification), multi-vector leakage audits, feature ablation studies, and Bayesian threshold alignments were directed, executed, and validated personally. Full technical responsibility for this delivery rests with the author.
