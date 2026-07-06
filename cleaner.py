import polars as pl
import numpy as np
from sklearn.impute import KNNImputer

def evaluate_dataset(df: pl.DataFrame) -> dict:
    total_rows = df.height
    unique_rows = df.unique().height
    duplicate_rows = max(0, total_rows - unique_rows) 
    
    summary = {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0
    }
    
    columns_profile = {}
    for col in df.columns:
        null_count = df[col].null_count()
        missing_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        unique_vals = df[col].n_unique()
        dtype_str = str(df[col].dtype)
        
        outlier_count = 0
        # IQR Outlier checking for numeric variables
        if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            arr = df[col].drop_nulls().to_numpy()
            if len(arr) > 4:
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = int(((arr < lower_bound) | (arr > upper_bound)).sum())

        columns_profile[col] = {
            "data_type": dtype_str,
            "unique_values": unique_vals,
            "missing": {"count": null_count, "percentage": round(missing_pct, 2)},
            "outliers": {"count": outlier_count}
        }
        
    return {"summary": summary, "columns": columns_profile}

def execute_flexible_clean(df: pl.DataFrame, configurations: dict) -> tuple[pl.DataFrame, list[str]]:
    """
    configurations shape:
    {
        "deduplicate": true,
        "columns": {
            "Age": {"clean_missing": true, "missing_strategy": "median", "handle_outliers": true},
            "Salary": {"clean_missing": true, "missing_strategy": "mean", "handle_outliers": false}
        }
    }
    """
    ledger = []
    working_df = df.clone()
    
    # Keep track of original column organization order
    original_column_order = list(working_df.columns)
    
    # 1. Selective Deduplication
    if configurations.get("deduplicate", True):
        initial_count = working_df.height
        working_df = working_df.unique()
        dropped = initial_count - working_df.height
        if dropped > 0:
            ledger.append(f"Deduplication: Safely purged {dropped} exact mirror duplicate rows.")

    col_configs = configurations.get("columns", {})
    
    for col in original_column_order:
        cfg = col_configs.get(col, {"clean_missing": True, "missing_strategy": "auto", "handle_outliers": True})
        original_dtype = working_df[col].dtype
        is_numeric = original_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        
        # 2. Outlier Capping (Winsorization Strategy)
        if cfg.get("handle_outliers", False) and is_numeric:
            non_nulls = working_df[col].drop_nulls()
            
            if non_nulls.len() > 4:
                p1 = np.percentile(non_nulls, 1)
                p99 = np.percentile(non_nulls, 99)
                
                # Clip values outside [p1, p99] while preserving metadata names
                working_df = working_df.with_columns(
                    pl.when(pl.col(col) < p1).then(p1)
                    .when(pl.col(col) > p99).then(p99)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                ledger.append(f"Outliers: Handled distribution spikes in '{col}' by capping at 1st ({p1:.2f}) and 99th ({p99:.2f}) percentiles.")

        # 3. Missing Data Treatment
        if cfg.get("clean_missing", True) and working_df[col].null_count() > 0:
            missing_pct = (working_df[col].null_count() / working_df.height) * 100
            strategy = cfg.get("missing_strategy", "auto")
            
            if strategy == "auto":
                strategy = "knn" if missing_pct >= 5.0 else "median"
            
            if not is_numeric and strategy in ["mean", "median", "knn"]:
                strategy = "mode" # Fallback for categorical labels
                
            if strategy == "mean":
                fill_val = working_df[col].mean()
                working_df = working_df.with_columns(pl.col(col).fill_null(fill_val))
                ledger.append(f"Imputation: Filled missing blocks in '{col}' via calculated average mean value ({fill_val:.2f}).")
                
            elif strategy == "median":
                fill_val = working_df[col].median()
                working_df = working_df.with_columns(pl.col(col).fill_null(fill_val))
                ledger.append(f"Imputation: Filled missing blocks in '{col}' via statistical center median value ({fill_val:.2f}).")
                
            elif strategy == "mode":
                mode_series = working_df[col].mode()
                fill_val = mode_series[0] if mode_series.len() > 0 else None
                if fill_val is not None:
                    working_df = working_df.with_columns(pl.col(col).fill_null(fill_val))
                    ledger.append(f"Imputation: Categorical patch applied to '{col}' using calculated Mode ('{fill_val}').")
                    
            elif strategy == "knn":
                ledger.append(f"Imputation: High missing variance ({missing_pct:.2f}%) on '{col}' triggered KNN structural synthesis.")
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    raw_arr = working_df[col].to_numpy().reshape(-1, 1)
                    imputed_arr = imputer.fit_transform(raw_arr)
                    
                    # FIXED: Cast array series explicitly back to original type to avoid floating variations
                    clean_series = pl.Series(col, imputed_arr.flatten()).cast(original_dtype)
                    working_df = working_df.with_columns(clean_series)
                except Exception:
                    # Defensive fallback if fitting matrix fails
                    fallback = working_df[col].median() if is_numeric else working_df[col].mode()[0]
                    working_df = working_df.with_columns(pl.col(col).fill_null(fallback))

    # FINAL ENFORCEMENT: Explicitly enforce the original column structural layout before export
    working_df = working_df.select(original_column_order)

    return working_df, ledger