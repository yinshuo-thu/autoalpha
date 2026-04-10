"""
compliance_guard.py — Central Competition Rule Enforcement

Checks every factor against competition constraints before evaluation.
This is the single checkpoint that guarantees compliance.
"""
import numpy as np
import pandas as pd
from formula_validator import validate_formula, ValidationResult


class ComplianceResult:
    def __init__(self):
        self.passed = True
        self.checks = {}

    def add_check(self, name, passed, detail=''):
        self.checks[name] = {'passed': passed, 'detail': detail}
        if not passed:
            self.passed = False

    def to_dict(self):
        return {'passed': self.passed, 'checks': self.checks}

    def __repr__(self):
        status = '✅ PASS' if self.passed else '❌ FAIL'
        details = ', '.join(f"{k}:{'✓' if v['passed'] else '✗'}" for k, v in self.checks.items())
        return f"Compliance[{status}] {details}"


def check_formula_compliance(formula_text, registered_assets=None):
    """Check that a formula only uses allowed data and operators."""
    result = ComplianceResult()

    val = validate_formula(formula_text, registered_assets)
    result.add_check('syntax', val.valid,
                     '; '.join(val.errors) if val.errors else 'OK')
    result.add_check('no_leakage', not any('LEAKAGE' in e for e in val.errors),
                     'No forbidden fields used' if val.valid else '; '.join(val.errors))
    result.add_check('operators_allowed', not any('Unknown operator' in e for e in val.errors),
                     'All operators whitelisted' if val.valid else '; '.join(val.errors))

    return result


def check_alpha_compliance(alpha_series, universe=None, trading_days=None):
    """
    Check that computed alpha values meet submission requirements.
    
    Args:
        alpha_series: pd.Series with MultiIndex (date, datetime, security_id)
        universe: pd.DataFrame with (date, security_id) index and eq_univ column
        trading_days: list of expected trading day strings
    """
    result = ComplianceResult()

    if alpha_series is None or (hasattr(alpha_series, 'empty') and alpha_series.empty):
        result.add_check('non_empty', False, 'Alpha is empty')
        return result

    result.add_check('non_empty', True, f'{len(alpha_series):,} values')

    # 1. 15-minute alignment check
    if 'datetime' in alpha_series.index.names:
        dt_level = alpha_series.index.get_level_values('datetime')
        if hasattr(dt_level, 'minute'):
            minutes = dt_level.minute.unique()
            valid_mins = {0, 15, 30, 45}
            bad_mins = set(minutes) - valid_mins
            result.add_check('15min_aligned', len(bad_mins) == 0,
                           f"All bars on 15min grid" if len(bad_mins) == 0
                           else f"Found non-15min minutes: {bad_mins}")

    # 2. Bounds check: alpha should be in [-1, 1] or reasonable range
    vals = alpha_series.values if hasattr(alpha_series, 'values') else alpha_series
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) == 0:
        result.add_check('has_finite_values', False, 'All values are NaN/inf')
        return result

    result.add_check('has_finite_values', True, f'{len(finite_vals):,} finite values')

    max_val = float(np.nanmax(finite_vals))
    min_val = float(np.nanmin(finite_vals))
    result.add_check('bounded', max_val <= 100 and min_val >= -100,
                     f"range=[{min_val:.4f}, {max_val:.4f}]")

    # 3. Not constant
    nunique = len(np.unique(finite_vals[~np.isnan(finite_vals)]))
    result.add_check('not_constant', nunique > 1, f'{nunique} unique values')

    # 4. Coverage check
    if trading_days is not None:
        alpha_dates = set(alpha_series.index.get_level_values('date').unique().astype(str))
        expected = set(trading_days)
        missing = expected - alpha_dates
        coverage = 1 - len(missing) / max(len(expected), 1)
        result.add_check('coverage', coverage > 0.95,
                        f'{coverage:.1%} ({len(missing)} missing days)')

    # 5. NaN ratio
    nan_ratio = 1 - len(finite_vals) / max(len(vals), 1)
    result.add_check('nan_ratio', nan_ratio < 0.5,
                     f'{nan_ratio:.1%} NaN')

    # 6. Concentration
    if len(finite_vals) > 0:
        abs_vals = np.abs(finite_vals)
        total = np.sum(abs_vals)
        if total > 0:
            max_pct = float(np.max(abs_vals) / total * 100)
            result.add_check('concentration', max_pct < 10.0,
                           f'max weight = {max_pct:.2f}%')

    return result


def full_compliance_check(formula_text, alpha_series=None, universe=None, trading_days=None):
    """Run all compliance checks: formula + alpha values."""
    formula_result = check_formula_compliance(formula_text)
    
    combined = ComplianceResult()
    combined.checks.update(formula_result.checks)
    combined.passed = formula_result.passed

    if alpha_series is not None:
        alpha_result = check_alpha_compliance(alpha_series, universe, trading_days)
        combined.checks.update(alpha_result.checks)
        if not alpha_result.passed:
            combined.passed = False

    return combined
