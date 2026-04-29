"""
quick_test.py — Single-Factor Quick Evaluation on Real Data

Pipeline: parse → validate → compute → evaluate → classify
Returns a structured JSON result with all metrics.

Usage:
    python quick_test.py "rank(sub(div(close_trade_px, vwap), 1))"
    python quick_test.py "ts_zscore(volume, 20)" --postprocess rank
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from formula_parser import parse_formula, ParseError, ast_to_string
from formula_validator import validate_formula
from compliance_guard import full_compliance_check
from prepare_data import DataHub
from core.submission import SubmissionBuilder


def _contract_product(contract):
    from core.futures_alpha import product_from_contract
    return product_from_contract(contract)


def compute_formula(formula_text, data_hub):
    """
    Compute a factor from a DSL formula on real 15m data.
    Returns a pandas Series with the same index as pv_15m.
    """
    from factors.operators import (
        lag, delta, ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_zscore,
        ts_rank, ts_decay_linear, ts_cov, ts_corr, cs_rank, cs_demean, cs_zscore, safe_div, signed_power,
        ts_median, ts_quantile, ts_skew, ts_kurt, ts_ema, ts_argmax, ts_argmin,
        ts_pct_change, ts_minmax_norm, cs_scale, cs_winsorize, cs_quantile,
        cs_neutralize, signed_log, safe_log, safe_sqrt, sigmoid, clamp,
        min_of, max_of, ifelse, gt, ge, lt, le, eq, and_op, or_op, not_op,
        mean_of, weighted_sum, combine_rank,
    )

    pv = data_hub.pv_15m

    # Build evaluation namespace with allowed fields and operators
    ns = {}
    for col in pv.columns:
        ns[col] = pv[col]

    # Operator mappings
    ns.update({
        'rank': cs_rank, 'cs_rank': cs_rank,
        'zscore': cs_zscore, 'cs_zscore': cs_zscore,
        'demean': cs_demean, 'cs_demean': cs_demean,
        'delay': lag, 'lag': lag,
        'delta': delta,
        'ts_mean': ts_mean, 'ts_std': ts_std, 'ts_sum': ts_sum,
        'ts_max': ts_max, 'ts_min': ts_min,
        'ts_median': ts_median, 'ts_quantile': ts_quantile,
        'ts_skew': ts_skew, 'ts_kurt': ts_kurt, 'ts_ema': ts_ema,
        'ts_argmax': ts_argmax, 'ts_argmin': ts_argmin,
        'ts_pct_change': ts_pct_change, 'ts_minmax_norm': ts_minmax_norm,
        'ts_cov': ts_cov, 'ts_corr': ts_corr,
        'ts_zscore': ts_zscore, 'ts_rank': ts_rank,
        'ts_decay_linear': ts_decay_linear, 'decay_linear': ts_decay_linear,
        'cs_scale': cs_scale, 'scale': cs_scale,
        'cs_winsorize': cs_winsorize, 'winsorize': cs_winsorize,
        'cs_quantile': cs_quantile, 'cs_neutralize': cs_neutralize,
        'safe_div': safe_div, 'div': safe_div,
        'signed_power': signed_power, 'pow': signed_power,
        'neg': lambda x: -x,
        'abs': lambda x: x.abs() if hasattr(x, 'abs') else np.abs(x),
        'log': safe_log,
        'signed_log': signed_log,
        'sqrt': safe_sqrt,
        'sub': lambda a, b: a - b,
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'clip': clamp, 'clamp': clamp,
        'min_of': min_of, 'max_of': max_of,
        'sigmoid': sigmoid, 'tanh': np.tanh,
        'gt': gt, 'ge': ge, 'lt': lt, 'le': le, 'eq': eq,
        'and_op': and_op, 'or_op': or_op, 'not_op': not_op,
        'ifelse': ifelse,
        'sign': lambda x: np.sign(x) if not hasattr(x, 'apply') else x.apply(np.sign),
        'mean_of': mean_of,
        'weighted_sum': weighted_sum,
        'combine_rank': combine_rank,
    })

    # Register derived convenience fields after operators are available.
    derived_formulas = {
        "ret_1bar": "close_trade_px / delay(close_trade_px, 1) - 1",
        "vwap_dev": "close_trade_px / vwap - 1",
        "hl_range": "high_trade_px - low_trade_px",
        "hl_range_pct": "div(sub(high_trade_px, low_trade_px), close_trade_px)",
        "volume_ratio": "div(volume, ts_mean(volume, 20))",
        "dollar_volume_ratio": "div(dvolume, ts_mean(dvolume, 20))",
        "mid_spread": "sub(close_trade_px, close_mid_px)",
        "order_flow_imbalance": "div(sub(buy_volume, sell_volume), add(buy_volume, sell_volume))",
        "oi_pressure": "div(delta_oi, ts_mean(volume, 20))",
        "book_pressure": "mean_of(book_imbalance, div(book_ofi, ts_mean(volume, 20)))",
    }
    for name, expr in derived_formulas.items():
        try:
            ns[name] = eval(expr, {"__builtins__": {}}, ns)
        except Exception:
            pass

    # Evaluate using restricted eval with namespace
    try:
        result = eval(formula_text, {"__builtins__": {}}, ns)
    except Exception as e:
        raise RuntimeError(f"Formula computation failed: {e}")

    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            result = result.iloc[:, 0]
        else:
            raise RuntimeError("Formula returned multiple columns")

    return result


def evaluate_factor(alpha_series, data_hub, factor_name='test'):
    """Evaluate a computed factor against resp."""
    from core.evaluator import Evaluator

    resp = data_hub.resp
    restriction = data_hub.trading_restriction

    # Broadcast daily resp to 15m alpha
    a_df = alpha_series.to_frame("alpha").reset_index()
    r_df = resp.reset_index()[["date", "security_id", "resp"]]
    merged = pd.merge(a_df, r_df, on=["date", "security_id"], how="inner")
    if merged.empty: return {"error": "No overlap"}
    merged = merged.set_index(["date", "datetime", "security_id"]).sort_index()
    alpha_aligned, resp_aligned = merged["alpha"], merged["resp"]
    if restriction is not None and not restriction.empty:
        rest_df = restriction.reset_index()
        m_rest = pd.merge(merged.reset_index(), rest_df, on=["date", "security_id"], how="left")
        restriction_aligned = m_rest.set_index(["date", "datetime", "security_id"]).sort_index().get("trading_restriction", 0).fillna(0)
    else: restriction_aligned = pd.Series(0.0, index=merged.index)

    # Run official evaluator
    try:
        metrics = Evaluator.run(alpha_aligned, resp_aligned, restriction_aligned)
        submission_like = Evaluator.run_submission_like(alpha_aligned, resp_aligned, restriction_aligned)
    except Exception as e:
        return {'error': f'Evaluator failed: {e}'}

    market_metrics = evaluate_factor_by_market(alpha_aligned, resp_aligned, restriction_aligned)

    # Flatten and map to expected format for frontend/leaderboard
    overall = metrics.get('overall', {})
    cloud = submission_like or {}
    daily_ic = metrics.get('daily_ic', pd.Series(dtype=float))
    missing_days = len(set(data_hub.get_trading_days_list()) - set(alpha_aligned.index.get_level_values('date').unique().astype(str)))

    daily_ic_list = []
    if isinstance(daily_ic, pd.Series) and not daily_ic.empty:
        # Convert index to string dates if they aren't already
        daily_ic_list = [{'date': str(d), 'IC': float(v)}
                        for d, v in daily_ic.items() if np.isfinite(v)]

    # Monthly heatmap
    monthly_heatmap = {}
    if not daily_ic.empty:
        daily_ic_df = daily_ic.to_frame('IC')
        daily_ic_df.index = pd.to_datetime(daily_ic_df.index)
        monthly = daily_ic_df.groupby(daily_ic_df.index.to_period('M')).mean()
        for period, row in monthly.iterrows():
            monthly_heatmap[str(period)] = float(row['IC'])

    return {
        'factor_name': factor_name,
        'IC': cloud.get('IC', 0) / 100.0,
        'rank_ic': metrics.get('rank_ic', 0) / 100.0,
        'IR': cloud.get('IR', overall.get('IR', 0)),
        'Turnover': cloud.get('Turnover', overall.get('Turnover', 0)),
        'TurnoverLocal': cloud.get('TurnoverLocal', overall.get('Turnover', 0)),
        'Score': cloud.get('Score', overall.get('Score', 0)),
        'score_raw': cloud.get('Score', overall.get('Score', 0)),
        'ic_minus_tvr': ((cloud.get('IC', 0) - 0.0005 * cloud.get('Turnover', 0)) / 100.0),
        'cover_all': 1 if missing_days == 0 else 0,
        'missing_days': missing_days,
        'maxx': cloud.get('maxx', overall.get('maxx', 0)),
        'minn': cloud.get('minn', overall.get('minn', 0)),
        'stability_score': metrics.get('stability_score', 0),
        'positive_ic_ratio': metrics.get('positive_ic_ratio', 0),
        'PassGates': cloud.get('PassGates', overall.get('PassGates', False)),
        'gates_detail': cloud.get('GatesDetail', overall.get('GatesDetail', {})),
        'classification': 'Submission Ready' if cloud.get('PassGates') else ('Research Candidate' if abs(cloud.get('IC', 0)) > 0.1 else 'Drop'),
        'yearly': metrics.get('yearly', {}),
        'monthly_heatmap': monthly_heatmap,
        'time_series': {
            'daily_ic': daily_ic_list,
        },
        'score_formula': 'score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100',
        'score_components': {
            'IC': cloud.get('IC', 0) / 100.0,
            'Turnover': cloud.get('Turnover', overall.get('Turnover', 0)),
            'TurnoverLocal': cloud.get('TurnoverLocal', overall.get('Turnover', 0)),
            'IR': cloud.get('IR', overall.get('IR', 0)),
        },
        'official_metrics': submission_like,
        'official_IC': submission_like.get('IC', 0),
        'official_IR': submission_like.get('IR', 0),
        'official_Turnover': submission_like.get('Turnover', 0),
        'official_Score': submission_like.get('Score', 0),
        'market_metrics': market_metrics,
    }


def evaluate_factor_by_market(alpha_aligned, resp_aligned, restriction_aligned):
    """Evaluate C/LH/M separately so futures factors can be kept per effective market."""
    from core.evaluator import Evaluator

    out = {}
    securities = pd.Index(alpha_aligned.index.get_level_values("security_id")).astype(str)
    products = pd.Series(securities.map(_contract_product), index=alpha_aligned.index)
    for product in ("C", "LH", "M"):
        mask = products.eq(product).to_numpy()
        if not mask.any():
            out[product] = {"available": False, "effective": False, "reason": "no contracts"}
            continue
        a = alpha_aligned.iloc[mask]
        r = resp_aligned.reindex(a.index)
        rest = restriction_aligned.reindex(a.index).fillna(0)
        try:
            metrics = Evaluator.run(a, r, rest)
            cloud = Evaluator.run_submission_like(a, r, rest)
        except Exception as exc:
            out[product] = {"available": True, "effective": False, "reason": str(exc)}
            continue
        daily_ic = metrics.get("daily_ic", pd.Series(dtype=float))
        rank_ic = float(metrics.get("rank_ic", 0.0) / 100.0)
        ic = float(cloud.get("IC", 0.0) / 100.0)
        ir = float(cloud.get("IR", 0.0))
        dates = sorted(pd.to_datetime(a.index.get_level_values("date")).strftime("%Y-%m-%d").unique())
        oos_ic = 0.0
        if len(dates) >= 4 and isinstance(daily_ic, pd.Series) and not daily_ic.empty:
            split = max(1, int(len(dates) * 0.7))
            oos_dates = set(dates[split:])
            oos = daily_ic[pd.to_datetime(daily_ic.index).strftime("%Y-%m-%d").isin(oos_dates)]
            if not oos.empty:
                oos_ic = float(oos.mean())
        effective = bool(
            np.isfinite(ic)
            and np.isfinite(rank_ic)
            and abs(rank_ic) >= 0.015
            and (abs(ic) >= 0.0005 or abs(oos_ic) >= 0.0005)
        )
        out[product] = {
            "available": True,
            "effective": effective,
            "IC": ic,
            "RankIC": rank_ic,
            "IR": ir,
            "Turnover": float(cloud.get("Turnover", 0.0)),
            "OOS_IC": oos_ic,
            "days": len(dates),
            "contracts": int(pd.Index(a.index.get_level_values("security_id")).nunique()),
        }
    return out


def quick_test(formula_text, factor_name='quick_test', postprocess=None, hypothesis=None, data_hub=None):
    """
    Full quick test pipeline: parse → validate → compute → evaluate → classify.
    Returns structured JSON result.
    hypothesis: optional AI rationale / research hypothesis (shown in Feishu metadata).
    data_hub: optional pre-loaded DataHub instance to avoid redundant loading.
    """
    result = {'factor_name': factor_name, 'formula': formula_text, 'status': 'pending'}

    # Stage 1: Validate
    validation = validate_formula(formula_text)
    result['validation'] = validation.to_dict()
    if not validation.valid:
        result['status'] = 'validation_failed'
        result['classification'] = 'Drop'
        result['reason'] = '; '.join(validation.errors)
        return result

    # Stage 2: Compliance
    from compliance_guard import check_formula_compliance
    compliance = check_formula_compliance(formula_text)
    result['compliance'] = compliance.to_dict()
    if not compliance.passed:
        result['status'] = 'compliance_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(compliance)
        return result

    # Stage 3: Load data & compute
    try:
        t0 = time.time()
        if data_hub is None:
            hub = DataHub()
        else:
            hub = data_hub
        result['data_load_time'] = time.time() - t0

        t1 = time.time()
        alpha = compute_formula(formula_text, hub)
        result['compute_time'] = time.time() - t1

        # Optional postprocess
        if postprocess == 'rank':
            from factors.operators import cs_rank
            alpha = cs_rank(alpha)
        elif postprocess == 'zscore':
            from factors.operators import cs_zscore
            alpha = cs_zscore(alpha)

    except Exception as e:
        result['status'] = 'computation_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(e)
        return result

    # Stage 4: Evaluate
    try:
        t2 = time.time()
        metrics = evaluate_factor(alpha, hub, factor_name)
        result.update(metrics)
        result['eval_time'] = time.time() - t2
        result['status'] = 'success'

        from paths import SUBMISSIONS_ROOT, FUTURE_ALPHA_ROOT
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in factor_name)
        out_dir = os.path.join(SUBMISSIONS_ROOT, safe_name)
        os.makedirs(out_dir, exist_ok=True)

        from core.futures_alpha import (
            compute_existing_alpha_correlations,
            export_future_alpha_format,
            futures_research_score,
        )
        corr_report = compute_existing_alpha_correlations(
            alpha,
            factor_name,
            out_dir=out_dir,
        )
        result['existing_alpha_correlation'] = corr_report
        futures_score = futures_research_score(result, corr_report, result.get("market_metrics", {}))
        result.update(futures_score)

        trading_days = hub.get_trading_days_list()
        sanity_report = SubmissionBuilder.pre_submit_sanity_check(
            alpha,
            trading_days[0],
            trading_days[-1],
        )
        result['sanity_report'] = sanity_report
        result['coverage_pass'] = bool(sanity_report.get('cover_all') == 1)
        result['submission_ready_flag'] = bool(
            result.get('PassGates', False) and sanity_report.get('submission_ready', False)
        )

        gate_detail = dict(result.get('gates_detail', {}))
        gate_detail['Coverage'] = result['coverage_pass']
        gate_detail['SubmissionFormat'] = bool(sanity_report.get('submission_ready', False))
        result['gates_detail'] = gate_detail

        if result['submission_ready_flag']:
            result['classification'] = 'Submission Ready'
        elif result.get('PassGates'):
            result['classification'] = 'Research Candidate'
            result['reason'] = 'Quality gates passed, but submission profile still needs fixing'
        elif (
            result.get("futures_score", 0.0) >= 20.0
            or len([m for m in (result.get("market_metrics") or {}).values() if m.get("effective")]) >= 2
        ) and corr_report.get("max_abs_corr", 1.0) < 0.75:
            result['classification'] = 'Futures Research Candidate'
            result['reason'] = 'Futures score and novelty thresholds passed; requires longer OOS confirmation'

        if result['submission_ready_flag']:
            result['recommendation'] = '[PASS] Passed gates — consider adding to research queue'
        elif result['classification'] == 'Futures Research Candidate':
            result['recommendation'] = '[FUTURES] Novel/effective in at least one futures market — queue for longer OOS validation'
        elif result['classification'] == 'Research Candidate':
            result['recommendation'] = '[CANDIDATE] Research candidate — worth further exploration'
        else:
            result['recommendation'] = '[DROP] Metrics too poor for submission'
            
        desc = hypothesis or f"Quick test result for {factor_name}: {formula_text}"
        out_path = os.path.join(out_dir, f"{safe_name}.parquet")
        SubmissionBuilder.build(alpha.to_frame("alpha"), out_path)
        effective_products = [
            product
            for product, item in (result.get("market_metrics") or {}).items()
            if item.get("effective")
        ]
        should_export_future = bool(effective_products) and corr_report.get("max_abs_corr", 1.0) < 0.85
        if os.environ.get("AUTOALPHA_FUTURES_EXPORT_EFFECTIVE_ONLY", "1") != "1":
            effective_products = ["C", "LH", "M"]
            should_export_future = True
        future_alpha_paths = []
        if should_export_future:
            future_alpha_paths = export_future_alpha_format(
                alpha,
                safe_name,
                out_root=FUTURE_ALPHA_ROOT,
                products=effective_products,
            )
        result["effective_products"] = effective_products
        result["future_alpha_export_paths"] = future_alpha_paths
        meta_path = os.path.join(out_dir, f"{safe_name}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "factor_name": factor_name,
                    "formula": formula_text,
                    "description": desc,
                    "hypothesis": hypothesis or desc,
                    "metrics": {k: v for k, v in result.items() if k not in {"time_series", "validation", "compliance"}},
                    "sanity_report": sanity_report,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        result['submission_path'] = out_path
        result['submission_dir'] = os.path.dirname(out_path)
        result['metadata_path'] = meta_path

    except Exception as e:
        result['status'] = 'evaluation_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(e)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test a factor formula on real data')
    parser.add_argument('formula', help='DSL formula to test')
    parser.add_argument('--name', default='quick_test', help='Factor name')
    parser.add_argument('--postprocess', choices=['rank', 'zscore'], help='Postprocessing')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🧪 Quick Test: {args.formula}")
    print(f"{'='*60}\n")

    result = quick_test(args.formula, args.name, args.postprocess)

    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"IC:     {result['IC']:.4f}")
        print(f"RankIC: {result['rank_ic']:.4f}")
        print(f"IR:     {result['IR']:.2f}")
        print(f"TVR:    {result['Turnover']:.1f}")
        print(f"Score:  {result['Score']:.2f}")
        print(f"Gates:  {'PASS' if result['PassGates'] else 'FAIL'}")
        print(f"Class:  {result['classification']}")
        print(f"\n{result.get('recommendation', '')}")
    else:
        print(f"Reason: {result.get('reason', 'Unknown error')}")
