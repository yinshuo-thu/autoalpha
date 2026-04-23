# Manual Factor Visual Summary

## Files
- `manual_factor_dashboard.png`
- `manual_factor_metric_distributions.png`
- `manual_factor_rank_vs_metrics.png`


## Quick Read
- Best factor by score: `manual_alpha_001` (range_location) with `Score=7.662`.
- Strongest family cluster: `range-location / bar-structure` signals.
- Lowest-turnover winners come from `ema_spread`, which helps score by keeping turnover near zero.
- Performance is not a single-factor fluke: 26 factors passed gate, spanning 16 families.

## Top 5 By Score
| factor_name      | family                     |   Score |        IC |       IR |   Turnover |
|:-----------------|:---------------------------|--------:|----------:|---------:|-----------:|
| manual_alpha_001 | range_location             | 7.66234 | 0.0185726 | 35.4014  |  11.389    |
| manual_alpha_002 | range_conditioned_location | 6.49139 | 0.0170753 | 31.5868  |  11.0504   |
| manual_alpha_003 | wick_imbalance             | 5.52228 | 0.0146096 | 37.666   |  11.2233   |
| manual_alpha_004 | zscore_body_fraction       | 5.34818 | 0.014052  | 45.8892  |  12.3141   |
| manual_alpha_005 | ema_spread                 | 4.99384 | 0.0211577 |  5.80949 |   0.877708 |

## Lowest Turnover Factors
| factor_name      | family            |   Turnover |   Score |
|:-----------------|:------------------|-----------:|--------:|
| manual_alpha_005 | ema_spread        |   0.877708 | 4.99384 |
| manual_alpha_008 | ema_spread        |   1.78473  | 4.40851 |
| manual_alpha_015 | close_zscore      |   3.60052  | 2.77842 |
| manual_alpha_024 | close_zscore      |   4.66117  | 1.82254 |
| manual_alpha_026 | multi_horizon_mix |   7.06276  | 0.75497 |

## Best Families
| family_label                    |   factor_count |   best_score |   mean_score |   mean_turnover |
|:--------------------------------|---------------:|-------------:|-------------:|----------------:|
| Close Location In Range         |              1 |      7.66234 |      7.66234 |        11.389   |
| Range Location x Range Surprise |              2 |      6.49139 |      5.66658 |        10.9995  |
| Wick Imbalance                  |              1 |      5.52228 |      5.52228 |        11.2233  |
| Body Fraction Z-Score           |              2 |      5.34818 |      5.02593 |        12.2786  |
| Short-Long EMA Spread           |              2 |      4.99384 |      4.70118 |         1.33122 |
| Open-Close Return               |              1 |      3.44457 |      3.44457 |        11.7052  |
| Body x Range Surprise           |              2 |      3.08194 |      2.94995 |        11.8541  |
| Body Fraction                   |              1 |      2.8372  |      2.8372  |        11.9145  |
