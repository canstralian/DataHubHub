import pandas as pd
import numpy as np

def get_dataset_info(df):
    """
    Get basic information about a dataset.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        'column_types': df.dtypes.astype(str).value_counts().to_dict(),
        'column_info': []
    }
    
    # Get info for each column
    for col in df.columns:
        col_info = {
            'name': col,
            'type': str(df[col].dtype),
            'missing': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100,
            'unique_values': df[col].nunique()
        }
        
        # Add additional info for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            })
        
        # Add additional info for categorical/text columns
        elif pd.api.types.is_object_dtype(df[col]):
            # Get top values
            value_counts = df[col].value_counts().head(5).to_dict()
            col_info['top_values'] = value_counts
            
            # Estimate if it's a categorical column
            if df[col].nunique() / len(df) < 0.1:  # If less than 10% of rows have unique values
                col_info['likely_categorical'] = True
            else:
                col_info['likely_categorical'] = False
        
        info['column_info'].append(col_info)
    
    return info

def detect_dataset_format(df):
    """
    Try to detect the format/type of the dataset based on its structure.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        String indicating the likely format
    """
    # Check for text data
    text_cols = 0
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].str.len().mean() > 100:
            text_cols += 1
    
    if text_cols / len(df.columns) > 0.5:
        return "text"
    
    # Check for time series data
    date_cols = 0
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            date_cols += 1
    
    if date_cols > 0:
        return "time_series"
    
    # Check if it looks like tabular data
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
    
    if numeric_cols > 0 and categorical_cols > 0:
        return "mixed"
    elif numeric_cols > 0:
        return "numeric"
    elif categorical_cols > 0:
        return "categorical"
    
    # Default
    return "generic"

def check_column_completeness(df, threshold=0.8):
    """
    Check if columns have good completeness (less than 20% missing values by default).
    
    Args:
        df: Pandas DataFrame
        threshold: Completeness threshold (0.8 = 80% complete)
        
    Returns:
        List of columns with poor completeness
    """
    results = []
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        completeness = 1 - missing_ratio
        
        if completeness < threshold:
            results.append({
                'Column': col,
                'Completeness': f"{completeness:.2%}",
                'Missing': f"{missing_ratio:.2%}",
                'Recommendation': 'Consider imputing or removing this column'
            })
    
    return results

def detect_outliers(series, method='iqr', factor=1.5):
    """
    Detect outliers in a pandas Series using IQR or Z-score method.
    
    Args:
        series: Pandas Series with numeric values
        method: 'iqr' or 'zscore'
        factor: Multiplier for IQR or Z-score threshold
        
    Returns:
        Tuple of (outlier_indices, lower_bound, upper_bound)
    """
    if method == 'iqr':
        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
    else:  # zscore
        # Z-score method
        from scipy import stats
        z_scores = stats.zscore(series.dropna())
        abs_z_scores = abs(z_scores)
        
        # Filter for Z-scores above threshold
        outlier_indices = np.where(abs_z_scores > factor)[0]
        outliers = series.dropna().iloc[outlier_indices].index.tolist()
        
        # Compute equivalent bounds for consistency
        mean = series.mean()
        std = series.std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    
    return outliers, lower_bound, upper_bound
