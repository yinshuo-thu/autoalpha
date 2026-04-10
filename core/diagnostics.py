import os
import pandas as pd
import json

class Diagnostics:
    @staticmethod
    def export(eval_res, export_dir):
        """
        Exports the metrics and raw artifacts from evaluate_research into a directory.
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)
            
        overall = eval_res.get('overall', {})
        yearly = eval_res.get('yearly', {})
        
        # 1. Summary JSON
        summary = {
            'overall': overall,
            'yearly': yearly
        }
        with open(os.path.join(export_dir, 'metrics.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # 2. Daily IC CSV
        if 'daily_ic' in eval_res and not eval_res['daily_ic'].empty:
            df_ic = eval_res['daily_ic'].to_frame('IC')
            df_ic.to_csv(os.path.join(export_dir, 'daily_ic.csv'))
            
        # 3. Bar IC CSV
        if 'bar_ic' in eval_res and not eval_res['bar_ic'].empty:
            df_bar_ic = eval_res['bar_ic'].to_frame('IC')
            df_bar_ic.to_csv(os.path.join(export_dir, 'bar_ic.csv'))
            
        # 4. Daily Turnover CSV
        if 'daily_tvr' in eval_res and not eval_res['daily_tvr'].empty:
            df_tvr = eval_res['daily_tvr'].to_frame('Turnover')
            df_tvr.to_csv(os.path.join(export_dir, 'daily_turnover.csv'))
            
        # 5. Yearly summary CSV
        if yearly:
            df_y = pd.DataFrame.from_dict(yearly, orient='index')
            df_y.index.name = 'Year'
            df_y.to_csv(os.path.join(export_dir, 'yearly_summary.csv'))
            
        return export_dir
