import pandas as pd
import numpy as np
from core.postprocess import cs_rank, cs_demean

class Combiner:
    @staticmethod
    def align_and_combine(alpha_list, method='equal', weights=None):
        if not alpha_list:
            return pd.Series(dtype=float)
            
        dfs = [a.to_frame(name=f'a_{i}') for i, a in enumerate(alpha_list)]
        df_join = dfs[0]
        for df in dfs[1:]:
            df_join = df_join.join(df, how='outer')
            
        if method == 'equal':
            weights = np.ones(len(dfs)) / len(dfs)
        elif method == 'weighted':
            if weights is None:
                weights = np.ones(len(dfs)) / len(dfs)
            weights = np.array(weights)
            weights = weights / np.sum(np.abs(weights))
        else:
            raise ValueError(f"Unknown blend method {method}")
            
        # Masked Average to correctly handle NaNs rather than naively mapping to 0
        valid_mask = ~df_join.isna()
        weighted_sum = (df_join.fillna(0) * weights).sum(axis=1)
        weight_sum = (valid_mask * weights).sum(axis=1)
        combined = weighted_sum / weight_sum.clip(lower=1e-8)
            
        # Re-rank combined to ensure reasonable distribution
        res = cs_rank(combined)
        return cs_demean(res)
