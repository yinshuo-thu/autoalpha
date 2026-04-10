import os
import pandas as pd
import numpy as np


ALLOWED_UTC_TIMES = [
    "01:45:00",
    "02:00:00",
    "02:15:00",
    "02:30:00",
    "02:45:00",
    "03:00:00",
    "03:15:00",
    "03:30:00",
    "05:15:00",
    "05:30:00",
    "05:45:00",
    "06:00:00",
    "06:15:00",
    "06:30:00",
    "06:45:00",
    "07:00:00",
]


class SubmissionBuilder:
    @staticmethod
    def _ensure_frame(alpha_df):
        if isinstance(alpha_df, pd.Series):
            return alpha_df.to_frame("alpha")
        return alpha_df.copy()

    @staticmethod
    def _alpha_col(alpha_df):
        return "alpha" if "alpha" in alpha_df.columns else alpha_df.columns[0]

    @staticmethod
    def pre_submit_sanity_check(alpha_df, start_date, end_date):
        alpha_df = SubmissionBuilder._ensure_frame(alpha_df)
        if alpha_df.empty:
            return {
                "status": "FAIL (Empty Dataframe)",
                "submission_ready": False,
                "cover_all": 0,
                "missing_days_count": 0,
                "row_count": 0,
            }

        if not isinstance(alpha_df.index, pd.MultiIndex):
            raise ValueError("Submission alpha must use MultiIndex(date, datetime, security_id)")

        index_names = list(alpha_df.index.names)
        required_index = ["date", "datetime", "security_id"]
        for name in required_index:
            if name not in index_names:
                raise ValueError(f"Missing required index level: {name}")

        alpha_col = SubmissionBuilder._alpha_col(alpha_df)
        alpha_values = alpha_df[alpha_col]
        datetimes = pd.to_datetime(alpha_df.index.get_level_values("datetime"))
        observed_times = sorted({dt.strftime("%H:%M:%S") for dt in datetimes})
        invalid_times = sorted(set(observed_times) - set(ALLOWED_UTC_TIMES))
        exact_grid = len(invalid_times) == 0

        duplicate_keys = int(alpha_df.index.duplicated().sum())
        in_bounds = not ((alpha_values > 1.0).any() or (alpha_values < -1.0).any())

        nunique = alpha_df[alpha_col].nunique()
        is_constant = nunique <= 1

        obs_dates = set(
            pd.to_datetime(alpha_df.index.get_level_values("date")).strftime("%Y-%m-%d")
        )
        from core.datahub import get_trading_days

        req_dates = set(get_trading_days(start=start_date, end=end_date))
        missing = req_dates - obs_dates
        cov_all = 1 if len(missing) == 0 else 0

        per_day_bar_counts = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(alpha_df.index.get_level_values("date")).strftime("%Y-%m-%d"),
                    "time": datetimes.strftime("%H:%M:%S"),
                }
            )
            .drop_duplicates()
            .groupby("date")["time"]
            .nunique()
        )

        submission_ready = all(
            [
                exact_grid,
                not per_day_bar_counts.empty,
                int(per_day_bar_counts.min()) == len(ALLOWED_UTC_TIMES),
                int(per_day_bar_counts.max()) == len(ALLOWED_UTC_TIMES),
                duplicate_keys == 0,
                in_bounds,
                not is_constant,
                cov_all == 1,
            ]
        )

        if submission_ready:
            status = "PASS"
        elif cov_all == 0:
            status = "WARNING (Missing Trading Days)"
        else:
            status = "FAIL (Submission Profile Invalid)"

        report = {
            "status": status,
            "submission_ready": submission_ready,
            "cover_all": cov_all,
            "missing_days_count": len(missing),
            "missing_days_sample": sorted(list(missing))[:10],
            "max": float(alpha_values.max(skipna=True)) if alpha_values.notna().any() else 0.0,
            "min": float(alpha_values.min(skipna=True)) if alpha_values.notna().any() else 0.0,
            "row_count": int(len(alpha_df)),
            "unique_days": int(len(obs_dates)),
            "duplicate_keys": duplicate_keys,
            "alpha_in_bounds": in_bounds,
            "alpha_is_constant": is_constant,
            "allowed_utc_times": ALLOWED_UTC_TIMES,
            "observed_utc_times": observed_times,
            "invalid_utc_times": invalid_times,
            "exact_15m_grid": exact_grid,
            "bars_per_day_min": int(per_day_bar_counts.min()) if not per_day_bar_counts.empty else 0,
            "bars_per_day_max": int(per_day_bar_counts.max()) if not per_day_bar_counts.empty else 0,
            "bars_per_day_expected": len(ALLOWED_UTC_TIMES),
            "full_intraday_grid": bool(
                not per_day_bar_counts.empty
                and int(per_day_bar_counts.min()) == len(ALLOWED_UTC_TIMES)
                and int(per_day_bar_counts.max()) == len(ALLOWED_UTC_TIMES)
            ),
        }
        return report

    @staticmethod
    def expand_to_full_grid(alpha_df, start_date, end_date, chunk_days=20):
        alpha_df = SubmissionBuilder._ensure_frame(alpha_df)
        if alpha_df.empty:
            return alpha_df

        alpha_col = SubmissionBuilder._alpha_col(alpha_df)
        alpha_frame = alpha_df[[alpha_col]].sort_index()

        from core.datahub import get_trading_days, load_universe

        trading_days = get_trading_days(start=start_date, end=end_date)
        if not trading_days:
            return alpha_frame

        universe = load_universe(trading_days)
        if universe.empty:
            return alpha_frame

        universe = universe.reset_index()
        if "eq_univ" in universe.columns:
            universe = universe[universe["eq_univ"] == True]
        universe = universe[["date", "security_id"]]
        universe["date"] = pd.to_datetime(universe["date"])

        chunks = []
        for i in range(0, len(trading_days), chunk_days):
            date_slice = trading_days[i : i + chunk_days]
            date_ts = [pd.to_datetime(d) for d in date_slice]
            alpha_chunk = alpha_frame[
                alpha_frame.index.get_level_values("date").isin(date_ts)
            ]
            univ_chunk = universe[universe["date"].isin(date_ts)]

            day_frames = []
            for day in date_ts:
                securities = (
                    univ_chunk.loc[univ_chunk["date"] == day, "security_id"]
                    .drop_duplicates()
                    .astype(int)
                    .tolist()
                )
                if not securities:
                    continue
                datetimes = pd.to_datetime(
                    [f"{day.strftime('%Y-%m-%d')} {time_str}" for time_str in ALLOWED_UTC_TIMES]
                )
                skeleton = pd.MultiIndex.from_product(
                    [[day], datetimes, securities],
                    names=["date", "datetime", "security_id"],
                )
                day_frames.append(pd.DataFrame(index=skeleton))

            if not day_frames:
                continue

            skeleton_df = pd.concat(day_frames)
            expanded = skeleton_df.join(alpha_chunk, how="left")
            chunks.append(expanded)

        if not chunks:
            return alpha_frame

        return pd.concat(chunks).sort_index()

    @staticmethod
    def build(alpha_df, out_path):
        """
        Exports alpha prediction dataframe to parquet.
        """
        df_export = SubmissionBuilder._ensure_frame(alpha_df)
            
        df_export = df_export.reset_index()
        req_cols = ['date', 'datetime', 'security_id']
        for c in req_cols:
            if c not in df_export.columns:
                raise ValueError(f"Missing required index column: {c}")
                
        # Make sure alpha column exists
        if 'alpha' not in df_export.columns:
            # Maybe it's the 4th column
            avail = [c for c in df_export.columns if c not in req_cols]
            if avail:
                df_export = df_export.rename(columns={avail[0]: 'alpha'})

        df_export["date"] = pd.to_datetime(df_export["date"]).dt.strftime("%Y-%m-%d")
        if pd.api.types.is_datetime64_any_dtype(df_export["datetime"]):
            df_export["datetime"] = df_export["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            df_export["datetime"] = pd.to_datetime(df_export["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_export["security_id"] = df_export["security_id"].astype(int)
        df_export["alpha"] = pd.to_numeric(df_export["alpha"], errors="coerce").clip(-1.0, 1.0)
        df_export = df_export[['date', 'datetime', 'security_id', 'alpha']]
        
        # Ensure path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_export.to_parquet(out_path, engine='pyarrow')
        return out_path
