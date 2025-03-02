import streamlit as st
import pandas as pd
import numpy as np

def process_with_smolagents(dataset, operation, custom_code=None):
    """
    Process dataset using SmolaAgents for various operations.
    
    Args:
        dataset: Pandas DataFrame to process
        operation: Type of processing operation
        custom_code: Custom code to execute (for custom processing)
        
    Returns:
        Processed pandas DataFrame
    """
    if dataset is None:
        raise ValueError("No dataset provided")
    
    # Create a copy to avoid modifying the original
    processed_df = dataset.copy()
    
    try:
        if operation == "Data Cleaning":
            processed_df = clean_dataset(processed_df)
        elif operation == "Feature Engineering":
            processed_df = engineer_features(processed_df)
        elif operation == "Data Transformation":
            processed_df = transform_dataset(processed_df)
        elif operation == "Custom Processing" and custom_code:
            # Execute custom code
            # Note: This is a security risk in a real application
            # Should be replaced with a safer approach
            local_vars = {"df": processed_df}
            exec(custom_code, {"pd": pd, "np": np}, local_vars)
            processed_df = local_vars["df"]
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return processed_df
    
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        raise

def clean_dataset(df):
    """
    Clean the dataset by handling missing values, duplicates, and outliers.
    
    Args:
        df: Pandas DataFrame to clean
        
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    for col in cleaned_df.columns:
        # For numeric columns
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            # If more than 20% missing, leave as is
            if cleaned_df[col].isna().mean() > 0.2:
                continue
            
            # Otherwise impute with median
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # For categorical columns
        elif pd.api.types.is_object_dtype(cleaned_df[col]):
            # If more than 20% missing, leave as is
            if cleaned_df[col].isna().mean() > 0.2:
                continue
            
            # Otherwise impute with mode
            mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
    
    # Handle outliers in numeric columns
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        # Skip if too many missing values
        if cleaned_df[col].isna().mean() > 0.1:
            continue
        
        # Calculate IQR
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Cap outliers instead of removing
        cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
    
    return cleaned_df

def engineer_features(df):
    """
    Perform basic feature engineering on the dataset.
    
    Args:
        df: Pandas DataFrame to process
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    engineered_df = df.copy()
    
    # Get numeric columns
    numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
    
    # Skip if less than 2 numeric columns
    if len(numeric_cols) >= 2:
        # Create interaction features for pairs of numeric columns
        # Limit to first 5 columns to avoid feature explosion
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:5]:
                # Product interaction
                engineered_df[f"{col1}_{col2}_product"] = engineered_df[col1] * engineered_df[col2]
                
                # Ratio interaction (avoid division by zero)
                denominator = engineered_df[col2].replace(0, np.nan)
                engineered_df[f"{col1}_{col2}_ratio"] = engineered_df[col1] / denominator
    
    # Create binary features from categorical columns
    cat_cols = engineered_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        # Skip if too many unique values (>10)
        if engineered_df[col].nunique() > 10:
            continue
            
        # One-hot encode
        dummies = pd.get_dummies(engineered_df[col], prefix=col, drop_first=True)
        engineered_df = pd.concat([engineered_df, dummies], axis=1)
    
    # Create aggregated features
    if len(numeric_cols) >= 3:
        # Sum of all numeric features
        engineered_df['sum_numeric'] = engineered_df[numeric_cols].sum(axis=1)
        
        # Mean of all numeric features
        engineered_df['mean_numeric'] = engineered_df[numeric_cols].mean(axis=1)
        
        # Standard deviation of numeric features
        engineered_df['std_numeric'] = engineered_df[numeric_cols].std(axis=1)
    
    return engineered_df

def transform_dataset(df):
    """
    Perform data transformations on the dataset.
    
    Args:
        df: Pandas DataFrame to transform
        
    Returns:
        Transformed pandas DataFrame
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Create a copy to avoid modifying the original
    transformed_df = df.copy()
    
    # Get numeric columns
    numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Create scaled versions of numeric columns
        
        # Standard scaling (z-score)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(transformed_df[numeric_cols])
        scaled_df = pd.DataFrame(
            scaled_data, 
            columns=[f"{col}_scaled" for col in numeric_cols],
            index=transformed_df.index
        )
        
        # Min-max scaling (0-1 range)
        minmax_scaler = MinMaxScaler()
        minmax_data = minmax_scaler.fit_transform(transformed_df[numeric_cols])
        minmax_df = pd.DataFrame(
            minmax_data,
            columns=[f"{col}_normalized" for col in numeric_cols],
            index=transformed_df.index
        )
        
        # Log transform (for positive columns only)
        log_cols = []
        for col in numeric_cols:
            if (transformed_df[col] > 0).all():
                transformed_df[f"{col}_log"] = np.log(transformed_df[col])
                log_cols.append(f"{col}_log")
        
        # Combine all transformations
        transformed_df = pd.concat([transformed_df, scaled_df, minmax_df], axis=1)
    
    # One-hot encode categorical columns
    cat_cols = transformed_df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        # One-hot encode all categorical columns
        transformed_df = pd.get_dummies(transformed_df, columns=cat_cols, drop_first=False)
    
    return transformed_df
