import os
import pandas as pd
import json
from datetime import datetime

def export_to_parquet(alpha_series, factor_name, output_dir='outputs', metrics=None, description="Auto-generated factor via pipeline"):
    """
    Exports the alpha factor to a parquet file following Scientech competition rules.
    Also saves a JSON file with metadata, evaluation data, pass/fail status, and score.
    Saves in structured folders: output_dir/submissions/{factor_name}_{timestamp}_{y|n}/
    """
    if metrics is None:
        metrics = {}
        
    overall_metrics = metrics.get('overall', metrics)
    pass_gates = overall_metrics.get('PassGates', metrics.get('PassGates', False))
    suffix_flag = 'y' if pass_gates else 'n'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{factor_name}_{timestamp}_{suffix_flag}"
    
    # Submissions always go inside the 'submissions' folder internally
    if not output_dir.endswith('submissions'):
        base_dir = os.path.join(output_dir, 'submissions')
    else:
        base_dir = output_dir
        
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Reset index to extract date, datetime, security_id
    df = alpha_series.to_frame('alpha').reset_index()
    
    # Bound alpha between -1.0 and 1.0
    df['alpha'] = df['alpha'].clip(-1.0, 1.0)
    
    # Ensure correct data types matching the Data Specification
    df['date'] = df['date'].astype(str)
    
    # Formatting datetime to UTC string if it's a timestamp
    if pd.api.types.is_datetime64_any_dtype(df['datetime']):
        # If it's already timezone-aware, we convert to UTC, then strip TZ
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
    df['security_id'] = df['security_id'].astype(int)
    
    # Keep only the required columns
    df = df[['date', 'datetime', 'security_id', 'alpha']]
    
    # Drop rows with NaN alpha just in case
    df = df.dropna(subset=['alpha'])
    
    # Save to Parquet
    out_path = os.path.join(target_dir, f"{factor_name}_submission.pq")
    df.to_parquet(out_path, engine='pyarrow')
    
    # Save JSON metadata
    metadata = {
        "factor_name": factor_name,
        "description": description,
        "PassGates": pass_gates,
        "Score": overall_metrics.get("Score", metrics.get("Score", 0.0)),
        "IC": overall_metrics.get("IC", metrics.get("IC", 0.0)),
        "rank_ic": overall_metrics.get("rank_ic", metrics.get("rank_ic", 0.0)),
        "IR": overall_metrics.get("IR", metrics.get("IR", 0.0)),
        "Turnover": overall_metrics.get("Turnover", metrics.get("Turnover", 0.0)),
        "classification": metrics.get("classification", "Unknown"),
        "formula": metrics.get("formula", ""),
        "timestamp": timestamp
    }
    
    json_path = os.path.join(target_dir, f"{factor_name}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Exported valid submission parquet to: {target_dir} ({len(df)} rows)")
    
    return out_path
