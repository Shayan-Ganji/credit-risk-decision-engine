-- ============================================================================
-- Stage 3: In-Database Preprocessing & Feature Extraction View
-- Schema: credit
-- View: view_prepared_credit_features
--
-- Engineering Objectives:
-- 1. Row-level sentinel pruning: Map bureau placeholder codes (96, 98) to NULL
-- 2. Derogatory bureau indicator: Capture severe delinquency flag (54-80% default rate)
-- 3. Missingness indicators: Preserve missingness as predictive signal for credit scoring
-- 4. Precalculated total past-due count: In-database row pushdown
-- 5. Disentangle DebtRatio anomalies: Flag absolute monthly debt vs true ratio
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS credit;

CREATE OR REPLACE VIEW credit.view_prepared_credit_features AS
SELECT
    -- Target Variable
    serious_dlq_in_2yrs,

    -- Core Quantitative Features
    revolving_utilization,
    age,
    debt_ratio,
    monthly_income,
    num_open_credit_lines,
    num_real_estate_loans,
    num_dependents,

    -- In-Database Cleaned Past-Due Counters (Pruning Sentinels 96 and 98 to NULL)
    CASE WHEN num_30_59_days_past_due IN (96, 98) THEN NULL ELSE num_30_59_days_past_due END AS clean_30_59_days_past_due,
    CASE WHEN num_60_89_days_past_due IN (96, 98) THEN NULL ELSE num_60_89_days_past_due END AS clean_60_89_days_past_due,
    CASE WHEN num_90_days_late IN (96, 98) THEN NULL ELSE num_90_days_late END AS clean_90_days_late,

    -- Precomputed Total Past Due Count (Deterministic Pushdown)
    COALESCE(CASE WHEN num_30_59_days_past_due IN (96, 98) THEN NULL ELSE num_30_59_days_past_due END, 0) +
    COALESCE(CASE WHEN num_60_89_days_past_due IN (96, 98) THEN NULL ELSE num_60_89_days_past_due END, 0) +
    COALESCE(CASE WHEN num_90_days_late IN (96, 98) THEN NULL ELSE num_90_days_late END, 0) AS total_past_due_count,

    -- Bureau Derogatory Code Flag (Preserving the 54-80% Default Signal)
    CASE 
        WHEN num_30_59_days_past_due IN (96, 98) 
          OR num_60_89_days_past_due IN (96, 98) 
          OR num_90_days_late IN (96, 98) 
        THEN 1 
        ELSE 0 
    END AS has_bureau_derogatory_flag,

    -- Missingness Indicators (Signal Preservation)
    CASE WHEN monthly_income IS NULL THEN 1 ELSE 0 END AS monthly_income_is_missing,
    CASE WHEN num_dependents IS NULL THEN 1 ELSE 0 END AS num_dependents_is_missing,

    -- DebtRatio Anomaly Flag (Disentangling Ratio vs. Absolute Debt)
    CASE WHEN monthly_income IS NULL AND debt_ratio > 1 THEN 1 ELSE 0 END AS debt_ratio_is_absolute_debt,

    -- Demographic Age Group for Fairness Auditing
    CASE 
        WHEN age < 30 THEN 'young'
        WHEN age BETWEEN 30 AND 59 THEN 'middle_age'
        ELSE 'senior'
    END AS age_group

FROM credit.applications_raw;

