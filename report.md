# Engineering Decisions and Architecture Report: Credit Risk Decision Engine
**Dotin Financial Technologies Lab - Term 3**  
**Student:** Shayan Ganji  

---

## 1. Problem Understanding and Foundational Assumptions

In retail banking credit evaluation, the primary business objective is not maximizing statistical classification accuracy; it is making binary underwriting decisions that minimize total financial loss to the institution's balance sheet. Across the historical Give Me Some Credit cohort of 149,999 applicants, only 6.68% experienced severe delinquency (90+ days past due) within two years of loan origination. In this severely imbalanced domain, a naive decision rule approving every applicant achieves an apparent 93.32% accuracy, yet it would approve all 10,026 defaulting borrowers, inflicting catastrophic financial losses and insolvency.

The challenge requires engineering an automated binary decision system (Approve vs. Reject) that maps continuous model posterior probabilities to discrete operational decisions under an asymmetric financial loss matrix:
* **False Negative (FN) Cost:** Approving a defaulting borrower causes the loss of the entire principal and interest (penalized at 12.0 units).
* **False Positive (FP) Cost:** Rejecting a healthy borrower causes the loss of expected interest profit (penalized at 1.0 unit).

The foundational assumption is that applicant behavioral distributions do not undergo sudden macroeconomic covariate shifts, and that self-reported income values and credit bureau inquiries available at underwriting ($T_0$) are free from coordinated fraudulent manipulation.

---

## 2. Key Architecture and Engineering Decisions

### Decision 1: Two-Tier Data Preparation Boundary (In-Database View vs. Python Pipeline)
Two architectural choices were evaluated: executing all data transformations in Python memory via Pandas, or partitioning operations across database pushdown and client-side processing.  
The chosen architecture delegates deterministic, row-level transformations to an analytical PostgreSQL View (`credit.view_prepared_credit_features`), while reserving stateful, training-set-dependent estimators for the Scikit-Learn Python Pipeline. Specifically, sentinel values (96 and 98) in past-due columns were pruned to `NULL` in SQL, the bureau derogatory indicator was extracted, and cumulative delinquency counts were precomputed. Conversely, median calculations for imputation and feature scalers remain in Python, strictly fitted on `X_train` to prevent test set data leakage. This pushdown minimized client memory overhead and locked database logic, at the cost of tighter coupling to the PostgreSQL daemon.

### Decision 2: Preserving Sentinel Codes 96 and 98 as a Bureau Derogatory Signal
Inspection of past-due counters revealed sentinel codes 96 and 98 across all three delinquency columns with 100% mutual co-occurrence. Two approaches were considered: treating them as anomalous outliers to be replaced with zeros/medians, or isolating them as a high-risk indicator.  
Empirical analysis revealed that normal records have an average default rate of 6.60%, whereas records with code 98 exhibit a 54.16% default rate and code 96 reaches 80.00% (an 8x to 12x default hazard multiplier). Consequently, an explicit binary indicator `has_bureau_derogatory_flag` was constructed in the database layer, while numeric counts were converted to `NULL` to prevent skewing linear regression and tree split points. This decision captured the strongest single credit signal in the 269 affected rows, while introducing correlated indicator features.

### Decision 3: Missingness Imputation Strategy and MNAR Signal Extraction
Two variables exhibit missing values: `monthly_income` (19.82% missing) and `num_dependents` (2.62% missing). Missingness is Missing Not At Random (MNAR), as borrowers in unstable or informal employment are less likely to report income. Two policies were benchmarked: simple median imputation (Policy A) versus median imputation combined with missingness indicator flags (Policy B).  
Empirical evaluation demonstrated that Policy B reduced expected decision cost from 365.40 units to 360.97 units per 1,000 applicants compared to Policy A. Tracking the missingness pattern provides the classifier with structural insight into borrower disclosure behavior. Policy B was selected, trading a minor expansion in feature dimensionality for improved loss minimization.

### Decision 4: Model Family Selection and Feature Scaling Trade-offs
Two distinct algorithm families were benchmarked: $L_2$-regularized Logistic Regression and HistGradientBoostingClassifier.  
Logistic Regression strictly required `StandardScaler` within the pipeline; without scaling, features spanning vast numerical scales (monthly income in thousands versus utilization in decimals) caused $L_2$ regularization to disproportionately penalize large-magnitude predictors. In contrast, HistGradientBoosting is scale-invariant, handles non-linear interactions natively, and achieved a superior expected cost of 360.97 units per 1,000 applicants (compared to 402.10 units for Logistic Regression). Sacrificing linear interpretability was the deliberate trade-off made to achieve superior risk discrimination.

### Decision 5: Cost-Driven Decision Threshold Optimization vs. Default 0.50
Scikit-Learn classifiers output probabilities, whereas banking institutions execute binary approvals. Using the standard 0.50 threshold implicitly assumes equal error costs.  
Under the 12:1 cost matrix, evaluating a fine grid over $\tau \in [0.01, 0.99]$ identified an optimal decision threshold at $\tau^* = 0.085$. This closely aligns with Bayesian decision theory:
$$\tau_{\text{Bayes}}^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{1 + 12} = \frac{1}{13} \approx 0.0769$$
Operating at $\tau^* = 0.085$ reduced the baseline underwriting cost from 802.00 units to **360.97 units per 1,000 applicants** (a 55.00% cost reduction, substantially exceeding the required 25% threshold). At this threshold, Recall reaches 80.84%, capturing the vast majority of defaults while tolerating moderate False Positives.

---

## 3. Failed Approaches and Discarded Experiments

1. **Listwise Deletion of Missing Income Records:** Dropping the 29,731 rows with missing income was tested initially. This eliminated 20% of the applicant population and dropped the default rate to 5.9%, creating severe selection bias and unviable production models.
2. **Global KNN Imputation:** Using KNN Imputer across 150,000 rows was benchmarked. It increased pipeline runtime to over 12 minutes without producing statistically significant cost improvements over median imputation with missingness flags, violating runtime constraints.
3. **Artificial Class Re-weighting (`class_weight='balanced'`):** Applying balanced sample weights in tree algorithms distorted raw posterior probability calibration (shifting mean predicted probability from 6.7% to ~50%). This broke the theoretical link between threshold search and Bayesian decision rules. The unweighted pipeline was retained to preserve well-calibrated posterior probabilities.

---

## 4. Operational Limitations and Production Failure Modes

In enterprise production deployments, three primary failure modes must be monitored:
1. **Macroeconomic Credit Shocks:** The model is trained on a stationary credit cycle. Sudden interest rate hikes or severe economic recessions alter default likelihoods across previously solvent borrowers, degrading the accuracy of the 12:1 cost optimization.
2. **Thin-File and Young Borrowers:** For young applicants lacking credit card history, historical delinquency counts are zero. The model displays elevated risk aversion toward this group, creating potential disparate impact that requires alternative underwriting criteria.
3. **Upstream Schema Drift in Debt Ratio:** If upstream core banking systems begin populating missing income debt ratios with `NULL` rather than absolute debt amounts, the assumption underlying `debt_ratio_is_absolute_debt` will require structural refactoring.

---

## 5. Proposed Extensions Given Two Additional Days

1. Construct a two-stage hurdle model that routes zero-income and informal-economy applicants through a specialized non-traditional underwriting engine.  
2. Implement Isotonic Probability Calibration to fine-tune posterior estimates in the extreme lower tail of default probabilities ($\le 2\%$).

---

## 6. Transparent AI Usage Disclosure

Implementation and code writing across most parts of this project (including the Data Factory source adapters, credit schema definitions, custom feature engineering transformers, Scikit-Learn pipelines, threshold optimization routines, and unit tests) were conducted using a vibe-coding workflow with an AI assistant (Antigravity). The assistant was also utilized to edit and refine the technical prose of this report.

However, all key architectural choices and hands-on data troubleshooting were driven directly by my own investigation: identifying sentinel codes 96 and 98 and their extreme default hazard, diagnosing the debt-ratio anomaly in missing-income applicants, partitioning row-level preprocessing into the PostgreSQL View, aligning empirical threshold optimization with Bayesian decision theory, and verifying leak-free pipeline boundaries. I assume full personal responsibility for the delivered codebase and system architecture.
