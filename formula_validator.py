"""
formula_validator.py — Three-stage formula validation

Stage 1: Syntax check (via parser)
Stage 2: Whitelist check (fields + operators)
Stage 3: Leakage / future-function risk check
"""
from formula_parser import parse_formula, collect_fields, collect_operators, ParseError
from data_catalog import RAW_FIELDS, FORBIDDEN_FIELDS, DERIVED_FIELD_TEMPLATES
from operator_catalog import ALLOWED_OPERATORS


class ValidationResult:
    def __init__(self):
        self.valid = True
        self.errors = []
        self.warnings = []
        self.fields_used = set()
        self.operators_used = set()
        self.ast = None

    def add_error(self, msg):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg):
        self.warnings.append(msg)

    def to_dict(self):
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'fields_used': sorted(self.fields_used),
            'operators_used': sorted(self.operators_used),
        }


def validate_formula(formula_text, registered_assets=None):
    """
    Three-stage validation of a formula string.
    
    Args:
        formula_text: The DSL formula string
        registered_assets: Optional set of additional registered asset names
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    registered = registered_assets or set()

    # ── Stage 1: Syntax Check ──
    try:
        ast = parse_formula(formula_text)
        result.ast = ast
    except ParseError as e:
        result.add_error(f"Syntax error: {e}")
        return result

    # ── Stage 2: Whitelist Check ──
    # Check fields
    fields = collect_fields(ast)
    result.fields_used = fields
    allowed_fields = set(RAW_FIELDS.keys()) | set(DERIVED_FIELD_TEMPLATES.keys()) | registered

    for f in fields:
        if f in FORBIDDEN_FIELDS:
            result.add_error(
                f"⛔ FORBIDDEN FIELD: '{f}' cannot be used in factor construction. "
                f"It is only allowed for evaluation purposes."
            )
        elif f not in allowed_fields:
            # Check if it looks like a numeric constant that was parsed as ident
            try:
                float(f)
                continue  # It's actually a number
            except ValueError:
                pass
            result.add_error(
                f"Unknown field: '{f}'. Not in allowed raw fields, derived fields, or registered assets. "
                f"Available raw: {sorted(RAW_FIELDS.keys())}"
            )

    # Check operators
    operators = collect_operators(ast)
    result.operators_used = operators

    # Map common aliases
    ALIASES = {
        'cs_rank': 'rank', 'cs_zscore': 'zscore', 'cs_demean': 'demean',
        'safe_div': 'div', 'signed_power': 'pow', 'lag': 'delay',
        'ts_decay_linear': 'decay_linear',
    }

    for op in operators:
        canonical = ALIASES.get(op, op)
        if canonical not in ALLOWED_OPERATORS:
            result.add_error(
                f"Unknown operator: '{op}'. Not in allowed operator list. "
                f"See operator_catalog.py for all allowed operators."
            )

    # ── Stage 3: Leakage & Future Function Check ──
    # Check for known future-looking patterns
    FUTURE_RISK_PATTERNS = {
        'resp': 'Uses response variable — definite data leakage',
        'trading_restriction': 'Uses trading restriction — not allowed in factor',
    }
    for f in fields:
        if f in FUTURE_RISK_PATTERNS:
            result.add_error(f"⛔ LEAKAGE: {FUTURE_RISK_PATTERNS[f]}")

    # Warn about potential issues
    if 'ts_corr' in operators or 'ts_cov' in operators:
        result.add_warning(
            "ts_corr/ts_cov with small windows may be noisy. "
            "Consider window >= 20 for stability."
        )

    if any(f.startswith('open_') for f in fields):
        result.add_warning(
            "Using opening prices — ensure no look-ahead bias "
            "(open of current bar is known at bar time)."
        )

    return result
