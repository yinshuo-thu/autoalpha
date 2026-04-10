import pandas as pd
import numpy as np
import py_compile
from factors import operators

# Operator Registry - Explicitly defined safe functions for DSL
OPS_REGISTRY = {
    'lag': operators.lag,
    'delta': operators.delta,
    'ts_mean': operators.ts_mean,
    'ts_std': operators.ts_std,
    'ts_sum': operators.ts_sum,
    'ts_max': operators.ts_max,
    'ts_min': operators.ts_min,
    'ts_zscore': operators.ts_zscore,
    'ts_rank': operators.ts_rank,
    'ts_decay_linear': operators.ts_decay_linear,
    'cs_rank': operators.cs_rank,
    'cs_demean': operators.cs_demean,
    'cs_zscore': operators.cs_zscore,
    'safe_div': operators.safe_div,
    'signed_power': operators.signed_power,
    'abs': np.abs,
    'sign': np.sign,
    'log': np.log,
    'sqrt': np.sqrt,
    'np': np
}

class FormulaEngine:
    @staticmethod
    def evaluate(formula_str, df):
        """
        Evaluate a string DSL formula using the restricted operator registry.
        Expects a DataFrame with MultiIndex [date, datetime, security_id].
        """
        # Expose dataframe columns as variables in the formula
        local_env = {col: df[col] for col in df.columns}
        
        # Add basic Python operators just in case
        builtins_safe = {'__builtins__': {}}
        
        try:
            # Parse and evaluate
            code = compile(formula_str, "<string>", "eval")
            result = eval(code, OPS_REGISTRY, local_env)
            return result
        except SyntaxError as e:
            raise ValueError(f"Syntax error in formula '{formula_str}': {e}")
        except Exception as e:
            raise RuntimeError(f"Runtime error evaluating formula '{formula_str}': {e}")

    @staticmethod
    def evaluate_callable(func, df):
        """Evaluate a pure Python callable."""
        try:
            return func(df)
        except Exception as e:
            raise RuntimeError(f"Error evaluating python callable: {e}")
