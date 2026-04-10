import os
import pandas as pd
import pyarrow.parquet as pq
import glob
from paths import DATA_ROOT

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def run_stats():
    base_dir = DATA_ROOT
    
    # 1. Total size of data
    data_dirs = ['eq_data_stage1', 'eq_resp_stage1', 'eq_trading_restriction_stage1', 'resp']
    stats = []
    
    for d in data_dirs:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            size_gb = get_dir_size(path) / (1024 ** 3)
            num_files = sum([len(files) for r, d, files in os.walk(path)])
            stats.append(f"- **{d}**: {num_files} files, {size_gb:.2f} GB")
            
    # 2. Check basic_pv days
    basic_pv_path = os.path.join(base_dir, 'eq_data_stage1', 'basic_pv')
    years = []
    total_days = 0
    start_date = "9999-99-99"
    end_date = "0000-00-00"
    
    if os.path.exists(basic_pv_path):
        for year in os.listdir(basic_pv_path):
            year_path = os.path.join(basic_pv_path, year)
            if os.path.isdir(year_path):
                years.append(year)
                for month in os.listdir(year_path):
                    month_path = os.path.join(year_path, month)
                    if os.path.isdir(month_path):
                        for day in os.listdir(month_path):
                            if day.isdigit():
                                total_days += 1
                                date_str = f"{year}-{month}-{day}"
                                if date_str < start_date: start_date = date_str
                                if date_str > end_date: end_date = date_str
                                
    # 3. Read one universe file
    univ_path = os.path.join(base_dir, 'eq_data_stage1', 'universe')
    tot_univ_days = 0
    univ_security_count_avg = 0
    if os.path.exists(univ_path):
        univ_files = glob.glob(os.path.join(univ_path, "*", "data.pq"))
        if univ_files:
            try:
                df_univ = pd.read_parquet(univ_files[0])
                tot_univ_days = df_univ.index.get_level_values('date').nunique()
                univ_security_count_avg = len(df_univ) / tot_univ_days
            except Exception as e:
                pass
                
    # Format the markdown result
    print("### Data Contents Statistics\n")
    print("#### Directory Footprint")
    for s in stats:
        print(s)
    print("\n#### Date Range & Coverage")
    if total_days > 0:
        print(f"- **Total Trading Days Covered (basic_pv)**: {total_days}")
        print(f"- **Date Range**: {start_date} to {end_date}")
        
    if tot_univ_days > 0:
        print("\n#### Universe Metrics")
        print(f"- **Sample Universe Days**: {tot_univ_days} (from one file)")
        print(f"- **Average Securities per Day**: {univ_security_count_avg:.0f}")

if __name__ == '__main__':
    run_stats()
