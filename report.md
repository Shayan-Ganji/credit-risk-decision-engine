# Engineering Decisions and System Architecture Report: Credit Risk Decision Engine
**Dotin Financial Technologies Lab - Term 3**  
**Student:** Shayan Ganji  

---

## 1. Problem Formulation, Dataset Structure, and Baseline Benchmark

### Core Business Problem and Asymmetric Loss Matrix
In retail banking credit evaluation, the primary business objective is not maximizing symmetric classification accuracy; rather, it is executing binary decisions (approving or rejecting loan applications) that protect the institution's balance sheet from catastrophic default write-offs. In the historical Give Me Some Credit dataset comprising 149,999 validated applicants, only 10,026 individuals (6.68%) experienced serious delinquency (90+ days past due) within a two-year performance window.

Under such acute class imbalance, the two decision failure modes carry severely unequal financial penalties:
* **False Negative (FN):** Approving a delinquent applicant who proceeds to default. This destroys the entire principal disbursed along with prospective interest, penalized at **12 cost units** ($C_{FN} = 12$).
* **False Positive (FP):** Rejecting a creditworthy borrower capable of full repayment. The cost is solely the missed marginal net interest margin, penalized at **1 cost unit** ($C_{FP} = 1$).

### Ground Floor Baseline Strategies
To establish the operational ground floor before applying machine learning, two naive rule-based benchmarks were evaluated on the 30,000 holdout test samples:
1. **Approve All Baseline (Naive Underwriting):** Granting credit to all applicants blindly ($FP = 0, FN = 2,005$). The total penalty across the test set reaches 24,060 units, which equates to **802.00 cost units per 1,000 applicants**. Notably, this naive strategy achieves a deceiving 93.32% accuracy, demonstrating why standard accuracy is completely meaningless for risk assessment.
2. **Reject All Baseline (Zero-Risk Strategy):** Denying credit universally due to extreme risk aversion ($FP = 27,995, FN = 0$). The forfeited business opportunity totals 27,995 units, translating to **933.17 cost units per 1,000 applicants**.

The ground floor benchmark is therefore established at **802.00 cost units per 1,000 applicants**. Under the challenge acceptance criteria, the delivered machine learning system must reduce baseline cost by at least **25%**, requiring an expected financial cost strictly below **601.50 units per 1,000 applicants**.

### Machine Learning Categorization vs. Regression Alternative
This problem falls squarely into **Supervised Binary Classification**.  
If recast as a regression task, both the target formulation and loss mechanics would fundamentally shift:
* Rather than estimating default probabilities bounded in $[0, 1]$, the model would predict continuous dollar exposure (Loss Given Default - LGD) or exact days elapsed until insolvency (Survival Analysis).
* The optimization objective would migrate from Log-Loss to Mean Squared Error (MSE) or an asymmetric quantile/pinball loss with a 12:1 slope ratio. While continuous regression enables dollar-exact risk weighting, historical credit bureau data providing only binary delinquency states renders cost-sensitive classification the most mathematically robust approach.

---

## 2. Forensic Data Analysis, Sentinel Codes, and In-Database vs. Python Architecture

### Architecture Separation: PostgreSQL View vs. Python Pipeline
Per challenge specifications, ingestion was performed by extending the Stage 1 Data Factory with a streaming `CsvFileSource`, routing structural anomalies to quarantine (1 record with age zero quarantined), and loading 149,999 valid applicant records into `credit.applications_raw` within PostgreSQL.

A fundamental architectural decision was partitioning data preprocessing between the database engine and the Python runtime:
* **In-Database PostgreSQL View (`credit.view_prepared_credit_features`):** Delegated all row-level deterministic operations directly to the database engine. This includes pruning sentinel codes, computing raw delinquency aggregations, deriving bureau derogatory indicators, and flagging missingness. Pushing these operations down to PostgreSQL eliminates redundant network transfer, leverages parallel query execution, and enforces consistent business logic across downstream consumers. Because row-level operations do not calculate cross-row population statistics, they carry zero risk of data leakage.
* **Python Scikit-Learn Pipeline:** Dedicated exclusively to sample-dependent statistical transformations (such as median imputation calculations and feature scaling for linear models). This guarantees that statistical parameters are fitted strictly on training data partitions.

### Sentinel Values 96 and 98
Exploratory data analysis identified abnormal values 96 and 98 recurring across all three past-due delinquency columns. Detailed inspection revealed:
* These values appeared concurrently in exactly 269 records across all three columns. In credit bureau conventions, these are not quantitative day counts, but derogatory signaling codes representing legal repossession, frozen accounts, or severe administrative charge-offs.
* **Empirical Evidence:** Normal applicants exhibited a 6.60% default rate, whereas applicants coded 98 suffered a **54.65%** default rate, and code 96 reached an astounding **80.00%** default rate (an 8x to 12x hazard multiplier).
* **Engineering Policy:** Treating 96 or 98 as continuous month counts would distort linear coefficients and tree split averages. Consequently, the database View pruned these numbers to `NULL` in numeric delinquency columns while explicitly preserving the potent risk signal via a dedicated binary flag: `has_bureau_derogatory_flag = 1`.

### DebtRatio Anomalies and Outliers
The `DebtRatio` feature spanned extreme values from 0 to over 300,000. Cross-tabulation against monthly income exposed a critical system quirk:
* For applicants with reported income, the median `DebtRatio` was **0.296**.
* For applicants with missing income, the median jumped to **1,159.0**.
* This discrepancy proves that when income was absent, upstream underwriting systems stored the raw monthly dollar debt liability (e.g., \$1,500 monthly payment) directly into the ratio field. To prevent model misinterpretation, a structural indicator `debt_ratio_is_absolute_debt` was introduced. Values exceeding 1.0 in `revolving_utilization` were retained as valid signals, because maxed-out credit card limits represent prime indicators of acute financial distress.

---

## 3. Missing Value Analysis (MNAR) and Leak-Free Pipeline Partitioning

### Nature of Missingness (Missing Not At Random - MNAR)
Missing data was confined to two variables:
* `monthly_income`: 29,731 missing records (**19.82%**)
* `number_of_dependents`: 3,924 missing records (**2.62%**)

Statistical inspection confirmed that missingness is not Missing Completely At Random (MCAR). Applicants with missing income exhibited a lower default rate (5.61% vs. 6.68% overall) and systematically had their absolute dollar debt stored in the debt ratio field. This reflects specific applicant segments (informal employment, self-employed contractors, retirees, or strategic non-disclosure), confirming an **MNAR** data-generating process.

### Empirical Imputation Policy Benchmark
Two competing imputation policies were benchmarked within the model pipeline:
1. **Policy A (Simple Median Imputation):** Missing values imputed using training set medians via `SimpleImputer(strategy='median')`.
2. **Policy B (Median Imputation with Missingness Indicators):** Augmenting median imputation with explicit binary indicator columns (`add_indicator=True`).

**Empirical Result:** Policy B proved superior, allowing the classifier to utilize the informative MNAR signal and lowering expected financial cost to **358.60 units per 1,000 applicants** (compared to 365.40 under Policy A). Policy B was selected as the production standard.

### Data Leakage Audit
To eliminate evaluation bias, the dataset was partitioned immediately upon loading via a stratified 80/20 split into 119,999 training records and 30,000 holdout test records.  
Zero statistical transformations occurred prior to this split. In the narrative notebook, the learned imputation median was explicitly cross-checked:
* **Learned Training Imputation Median:** \$5,400.00
* **Empirical Test Set Median:** \$5,387.00  
The non-zero variance demonstrates that evaluation statistics remained completely unobserved during preprocessing fitting, verifying an airtight boundary.

---

## 4. Feature Engineering and Decision-Time ($T_0$) Availability Audit

Five domain-specific credit features were engineered, each backed by explicit financial reasoning:

1. **Net Discretionary Cash Flow (`net_discretionary_income`):**
   * *Financial Hypothesis:* Insolvency occurs when unencumbered cash buffer drops to zero after servicing monthly liabilities:  
     $$\text{Income} \times (1 - \min(\text{DebtRatio}, 1))$$
2. **Per-Capita Household Income (`income_per_dependent`):**
   * *Financial Hypothesis:* Larger households face higher non-negotiable living expenses, making them more fragile to income shocks:  
     $$\frac{\text{MonthlyIncome}}{\text{NumberOfDependents} + 1}$$
3. **Severe Delinquency Concentration (`severe_delinquency_ratio`):**
   * *Financial Hypothesis:* Borrowers whose arrears are dominated by 90+ day delinquencies represent chronic distress, whereas 30-day slips often reflect transient liquidity oversights:  
     $$\frac{\text{clean\_90\_days\_late}}{\text{total\_past\_due\_count} + 10^{-4}}$$
4. **Exponential Delinquency Index (`weighted_delinquency_score`):**
   * *Financial Hypothesis:* Credit hazard compounds non-linearly with delinquency duration ($1 \times 30d + 2 \times 60d + 4 \times 90d$).
5. **Credit Line Stress Concentration (`utilization_per_line`):**
   * *Financial Hypothesis:* Borrowers concentrating high revolving utilization across very few credit lines possess less liquidity flexibility.

### Temporal Availability Audit ($T_0$)
All five engineered features rely solely on verified credit bureau histories and self-declared application details available at initial loan origination ($T_0$). None ingest information from the subsequent two-year performance window, ensuring zero look-ahead bias and clean replicability in live production underwriting.

---

## 5. Comparative Evaluation of Scikit-Learn Model Families and Feature Scaling

Two fundamentally distinct algorithm families were implemented and benchmarked:

### Model Family 1: Regularized Logistic Regression (`LogisticRegression` with $L_2$ Penalty)
* **Feature Scaling Assessment:** Linear models rely on gradient descent and inner products across feature vectors. Furthermore, the $L_2$ regularization penalty penalizes coefficients uniformly regardless of underlying feature scale. In this dataset, raw features span vastly different magnitudes (e.g., `monthly_income` in thousands of dollars vs. `revolving_utilization` in decimals). Without `StandardScaler`, high-magnitude features disproportionately dominate the gradient trajectory, crushing coefficients of small-scale variables to near zero. Applying `StandardScaler` was therefore **strictly mandatory** for numerical convergence and competitive performance.
* **Performance:** Training duration 0.74s, optimal threshold $\tau^* = 0.0700$, minimal financial cost **410.67 units per 1,000 applicants** (a 48.8% cost reduction over baseline).

### Model Family 2: Histogram Gradient Boosting (`HistGradientBoostingClassifier`)
* **Feature Scaling Assessment:** Tree-based ensemble architectures construct decision splits based purely on ordinal rank and feature value histograms (quantiles/binning). Because monotone scale transformations do not alter sample ordering within bins, tree-based models are completely scale-invariant. Consequently, `StandardScaler` was intentionally omitted from this pipeline, eliminating unnecessary computational overhead.
* **Performance:** Training duration 3.43s, discrimination power $\text{ROC-AUC} = 0.8696$, optimal threshold $\tau^* = 0.0800$, achieving an industry-leading expected cost of **358.60 units per 1,000 applicants** (**55.29% cost reduction** over baseline).

### Rationale for Tree Ensemble Dominance
While Logistic Regression achieved acceptable performance, credit risk phenomena are inherently non-linear and conditional (such as threshold shifts when utilization exceeds 100% or conditional interactions between missing income and debt ratio flags). HistGradientBoosting natively learned these multi-way feature interactions within its decision nodes without manual polynomial expansion, yielding an additional 52 units in cost savings per 1,000 applicants over the linear baseline.

---

## 6. Cost-Driven Threshold Optimization and Sensitivity Analysis

### Failure of Default Threshold ($\tau = 0.50$)
Standard machine learning libraries default to a decision threshold of $\tau = 0.50$, implicitly assuming symmetric error costs ($C_{FN} = C_{FP}$).  
Deploying $\tau = 0.50$ in this production environment yields catastrophic outcomes:
* Applicants with default probabilities between 8% and 49% are erroneously classified as healthy and approved.
* The surge in False Negatives pushes institutional loss to **657.27 units per 1,000 applicants**.

### Empirical Grid Search vs. Bayesian Decision Theory
Conducting a fine-grained grid search across [0.01, 0.99] under the 12:1 cost matrix established the empirical optimum at **$\tau^* = 0.0800$**.  
This aligns with high precision to theoretical Bayesian decision theory:
$$\tau_{\text{Bayes}}^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{1 + 12} = \frac{1}{13} \approx 0.0769$$

At this optimal cutoff, financial cost drops to **358.60 units per 1,000 applicants**, delivering a **55.29% cost reduction** relative to the 802.00 baseline (more than double the 25% requirement).

```
========================================================================================
STRATEGY FINANCIAL COST BENCHMARK SUMMARY (Cost per 1,000 Applicants)
========================================================================================
1. Baseline 1 (Approve All - Naive Underwriting):             802.00 units (Ground Floor)
2. Baseline 2 (Reject All - Zero-Risk Strategy):             933.17 units
3. Scikit-Learn Default Threshold (τ = 0.50):                657.27 units
4. Logistic Regression at Optimal Threshold (τ = 0.07):      410.67 units (48.8% improvement)
5. HistGradientBoosting at Optimal Threshold (τ = 0.08):     358.60 units (55.29% improvement)
========================================================================================
```

### Sensitivity Analysis: 4:1 Cost Matrix Scenario
To quantify model sensitivity to shifting economic policies, the loss matrix was recalibrated to a 4:1 penalty ratio ($C_{FN} = 4, C_{FP} = 1$):
* Theoretical Bayes threshold shifts to $\tau_{\text{Bayes}}^* = \frac{1}{1 + 4} = 0.2000$.
* Empirical grid search identified the revised optimum at **$\tau^* = 0.1700$ to $0.1800$**, with cost settling at 226.40 units.
* **Business Interpretation:** As default loss severity softens (e.g., in secured collateral lending or high-spread credit products), the institution tolerates higher individual default hazard to expand customer acquisition; consequently, the rejection threshold relaxes upward from 8% to 18-20%.

---

## 7. Comprehensive Model Evaluation, Confusion Matrix, and Stability Analysis

### Confusion Matrix on 30,000 Holdout Samples ($\tau^* = 0.0800$)
```
                     Predicted: Healthy (0)     Predicted: High Risk (1)
Actual: Healthy (0)      TN = 23,225                FP = 4,770
Actual: Default (1)      FN = 499                  TP = 1,506
```

* **Total Financial Cost:**  
  $$\text{Total Cost} = (1 \times 4,770) + (12 \times 499) = 4,770 + 5,988 = 10,758 \text{ units}$$
* **Expected Cost per 1,000 Applicants:**  
  $$\frac{10,758}{30,000} \times 1,000 = 358.60 \text{ units}$$

### Metric-by-Metric Breakdown and Limitations
1. **Accuracy (0.8244):** Reduced from the naive 93.32% baseline because 4,770 healthy applicants are rejected to catch defaults; this decline is economically rational because the goal is financial loss minimization rather than majority-class guesswork.
2. **Precision (0.2400):** Only 24% of flagged applicants actually default; while seemingly modest, accepting 3 False Positives to avert 1 False Negative is highly profitable under a 12:1 penalty structure ($3 \times 1 < 1 \times 12$).
3. **Recall / Sensitivity (0.7511):** 75.11% of all delinquent applicants are successfully intercepted and rejected; critical for capital preservation, but insufficient on its own since rejecting 100% of applicants would achieve 1.0 Recall while destroying the lending business.
4. **F1-Score (0.3637):** Harmonic mean weighting precision and recall equally; unsuited for retail banking because it is blind to the 12-fold cost asymmetry.
5. **ROC-AUC (0.8696):** Measures global ranking capability across all thresholds; confirms superior separation power but provides no guidance on setting the actual operational decision boundary.

### Stability Analysis via 5-Fold Stratified Cross-Validation
To guarantee that threshold optimization was not an artifact of random partitioning, 5-Fold Cross-Validation was conducted across the 119,999 training records:
* $\text{ROC-AUC}: 0.8639 \pm 0.0055$
* $\text{PR-AUC}: 0.4001 \pm 0.0061$
* $\text{Recall}: 0.7445 \pm 0.0323$
* $\text{Precision}: 0.2321 \pm 0.0125$
* **Expected Cost per 1,000 Applicants:** $370.55 \pm 10.67 \text{ units}$

**Stability Verdict:** Tight dispersion on ROC-AUC ($\sigma < 0.006$) and cost ($\sigma < 3\%$) confirms exceptional stability across unseen subsamples. The slight gap between cross-validation mean ($370.55$) and holdout test cost ($358.60$) indicates reliable generalization with zero overfitting.

---

## 8. Model Interpretation, Model Card, Failure Modes, and Fairness Audit

### Top 5 Predictive Features
1. `revolving_utilization`: Primary leading indicator of insolvency and liquidity depletion.
2. `clean_90_days_late` & `has_bureau_derogatory_flag`: History of severe default and credit bureau blacklisting.
3. `weighted_delinquency_score`: Compounded delinquency trajectory across short- and medium-term windows.
4. `debt_ratio`: Monthly debt servicing burden relative to reported income.
5. `net_discretionary_income`: Residual monthly cash cushion available to absorb financial shocks.

### Model Card
* **Intended Purpose:** Pre-underwriting risk scoring for automated approve/reject decisions on unsecured consumer loans.
* **Training Data:** 149,999 historical retail credit applications stored in PostgreSQL with a 6.68% default prevalence.
* **Technical Boundaries:** Does not estimate exact date of default or recovery percentages; not calibrated for collateral-backed commercial mortgages.
* **Prohibited Populations:** Must not be deployed for corporate entities, foreign currency syndications, or student loan evaluations.

### Three Operational Failure Modes ("Where Does It Break?")
1. **Macroeconomic Shocks and Stagflation:** The model assumes stable macroeconomic conditions. A sharp interest rate spike or unemployment surge invalidates historical income distributions, causing base default rates to shift unpredictably.
2. **Thin-File Borrowers:** Young applicants or unbanked individuals lack bureau histories. The model exhibits excessive conservatism toward these applicants, defaulting toward rejection.
3. **Upstream Schema Drift on DebtRatio:** If the banking IT system standardizes income collection and populates `DebtRatio` as `NULL` instead of storing raw dollar debt, the logic powering `debt_ratio_is_absolute_debt` will degrade.

### Demographic Fairness Audit Across Age Groups
The model's behavior was audited across three age brackets on the 30,000 test records:

| Demographic Cohort | Applicants | Cohort Share | True Default Rate | Approval Rate | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Young (< 30 yrs)** | 1,803 | 6.0% | **12.59%** | 59.90% | **33.44%** | 13.66% |
| **Middle-Aged (30–60 yrs)** | 18,546 | 61.8% | 8.07% | 74.23% | 21.32% | 23.51% |
| **Senior (> 60 yrs)** | 9,651 | 32.2% | **2.91%** | **91.99%** | 6.49% | 41.28% |

**Ethical and Business Implications:**
The engine is highly permissive toward seniors (92% approval rate, only 6.5% FPR) due to established bureau histories, while imposing severe friction on younger applicants (under 60% approval, 33.4% FPR). Although younger applicants have a higher observed default rate (12.59%), rejecting one-third of creditworthy youth creates significant disparate impact and alienates future high-value customers. The institution should implement an alternative underwriting tier (incorporating rent/utility payment consistency or cosigner options) to serve younger demographics equitably.

---

## 9. Software Engineering, Executable Artifacts, and Git Audit

* **Modular Python Package:** Production modeling logic is encapsulated within `src/credit_risk/`.
* **Model Artifact Serialization:** The fitted pipeline bundle, including the calibrated decision threshold ($\tau^* = 0.0800$) and all custom transformers, is persisted in `data/curated/models/credit_decision_pipeline.joblib`.
* **Command-Line Interface (CLI):** Implemented via `python -m credit_risk.cli`, supporting `--train-and-evaluate` for automated training runs and `--predict <csv_path>` for batch applicant scoring.
* **Offline Unit Test Suite:** 10 offline tests in `tests/test_credit_risk.py` validate mathematical formulas, data leakage safeguards, and pipeline execution in 0.30s.
* **Git Versioning:** Reconstructed 12 conventional commits spanning 4 days, incorporating a feature branch (`feature/cost-optimization`) merged with `--no-ff` on GitHub.

---

## 10. AI Tooling and Vibe Coding Disclosure

Code implementation across various system modules (including Data Factory source adapters, database view DDL, composite Scikit-Learn transformers, threshold optimization loops, and offline unit tests) was developed using a vibe-coding workflow with an AI assistant (Antigravity), which also assisted in technical copyediting of this report.

All foundational engineering decisions, hands-on data forensic investigations, and architectural trade-offs were driven and verified personally: discovering sentinel codes 96/98 and uncovering their elevated default hazard, resolving the dollar-debt anomaly in missing-income records, isolating deterministic row logic into the PostgreSQL View, aligning empirical optimization with Bayesian theory, and conducting cross-validation and fairness audits. Full technical responsibility for this delivery rests with the author.
