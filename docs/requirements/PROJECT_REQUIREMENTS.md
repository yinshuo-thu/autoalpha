# Alpha Quantitative Challenge - Core Requirements and Data Specification

This document outlines the core requirements, timeline, evaluation criteria, and dataset specifications for the Scientech Alpha Quantitative Challenge. It serves as the primary reference guide for AI development and quantitative modeling on this project.

## 1. Challenge Overview

This initiative is designed to bridge the gap between academia and industry, offering hands-on experience in quantitative alpha development. The program emphasizes exploratory learning using real-world, anonymized data from global equity markets.

### Timeline
- **March 10**: Project Kickoff (Data compliance and NDA signing)
- **March 12**: First Mentor Meeting
- **March 16**: In-Sample Testing Begins (Max 200 test runs per team)
- **May 15**: Out-of-Sample (OOS) Testing Begins (Max 20 test runs per team)
- **June 5**: Final Presentation & Awards (Top 3 teams based on OOS performance)

## 2. Evaluation System

The platform evaluates submitted alphas strictly based on intraday predictive performance and consistency. Alphas must be submitted at **15-minute frequency**.

### Core Evaluation Metrics
- **IC (Information Coefficient)**: Mean daily correlation between your alpha and the label. Target: `IC > 0.6` (scaled x100).
- **IR (Information Ratio)**: Annualized consistency of IC over time. Target: `IR > 2.5`.
- **Turnover (tvr)**: Mean daily turnover, estimating trading cost penalty. Target: `tvr < 400`.
- **Score Formula**: `score = (IC - 0.0005 * tvr) * sqrt(IR) * 100`

### Quality Gates (Elimination Criteria)
Submissions failing any gate will be scored `0`:
1. **Coverage**: Must evaluate every trading day.
2. **Predictive Power**: `IC > 0.6`.
3. **Consistency**: `IR > 2.5`.
4. **Turnover**: `tvr < 400`.
5. **Concentration Limits**: Largest single-security position weight `maxx < 50 bps`, absolute min `minn < 50 bps`. Average max `< 20 bps`.

## 3. Data Specification

All data is stored in Apache Parquet format (`.pq`). Each Parquet file preserves schemas inherently:
- **`basic_pv`**: 1-minute OHLCV bar data with VWAP and trade counts. UTC timestamps.
- **`universe`**: The daily list of valid, tradeable securities.
- **`resp` (Labels)**: The next-day return mapped to each 15-minute bar. **Local evaluation ONLY.**
- **`trading_restriction`**: 15-minute binary masks of restrictions (1 = short restricted, 2 = long restricted, 3 = both). **Local evaluation ONLY.**

---

## 4. Dataset Contents & Statistics

Below are the detected statistics for the current In-Sample core data:

#### Directory Footprint
- **eq_data_stage1**: 729 files, 47.07 GB
- **eq_resp_stage1**: 726 files, 0.58 GB
- **eq_trading_restriction_stage1**: 726 files, 0.10 GB
- **resp**: 726 files, 0.58 GB

#### Date Range & Coverage
- **Total Trading Days Covered (basic_pv)**: 726
- **Date Range**: 2022-01-04 to 2024-12-31

#### Universe Metrics
- **Sample Universe Days**: 242 (from one file)
- **Average Securities per Day**: 4748

---

## 5. Development Guidelines
- Always map predictions against the **Universe** per trading day.
- Output files should ideally have the same layout as the `resp` file (i.e. `{yyyy}/{mm}/{dd}/data.pq`) containing 15-minute snapshots of cross-sectional rank or alpha value.
- Do NOT feature-engineer using the `resp` or `trading_restriction` records, as these leak future information and are masked in Out-Of-Sample pipelines.

---

## 6. Testing Phases & Restrictions

### Phase: In-Sample Test (Development)
- **Period**: 2022-01-01 to 2024-12-31
- **Starts**: Mar 16, 2026 9:00:00 AM CST
- **Ends**: Jun 10, 2026 6:00:00 PM CST
- **Note**: Response data accessible for iteration. Leaderboards hidden. Submissions capped at 200 total / 200 per day.

### Phase: Out-of-Sample Test (OOS)
- **Period**: 2025-01-01 to 2025-12-31
- **Starts**: May 15, 2026 9:00:00 AM CST
- **Ends**: Jun 10, 2026 6:00:00 PM CST
- **Note**: Generalization test. No response/restriction data provided. Submissions capped at 20 total / 20 per day.

---

## 7. Submission Guidelines

Your final alpha prediction must be uploaded as a single **`.pq`** (Parquet) file, optionally wrapped in a `.zip` or `.tar.gz` archive. Max file size is **2 GB**.

### Data Formatting Requirements
The submission DataFrame must uniquely map predictions to `(date, datetime, security_id)`. These can be individual columns or constitute a `MultiIndex` with `alpha` as the sole value column.
- **date** (`date` or `string`): `yyyy-mm-dd` (e.g., `2022-01-04`).
- **datetime** (`timestamp` or `string`): `yyyy-mm-dd hh:mm:ss` (UTC timezone).
- **security_id** (`int`): Valid Universe member for the date.
- **alpha** (`float`): Float scalar `[-1.0, 1.0]`, representing directional tilt (positive = outperform). Unmapped predictions should be `NaN`.

### Temporal & Validation Rules
- **Frequency Grid**: Must snap to exact 15-minute intervals synchronized with standard evaluation boundaries (`01:45`, `02:00`... `03:30` UTC & `05:15`... `07:00` UTC). Submitting downscaled arrays (e.g., `1-min`) will result in unnecessary bloat.
- **Full Coverage Requirement**: To ensure the `cover_all = 1` constraint passes, the predictions must continuously span every single valid trading day from the earliest to the latest date. Gaps equate to all-NaNs.
- **Strict Format**: Any OOB datetimes, schema breaks, or alpha violations (`> 1.0`) trigger instant rejection of the entire file.
