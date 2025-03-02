import streamlit as st
import pandas as pd
import numpy as np
import json
from utils.dataset_utils import check_column_completeness, detect_outliers

def render_dataset_validation(dataset, dataset_type):
    """
    Renders validation checks for the dataset.
    
    Args:
        dataset: The dataset to validate (pandas DataFrame)
        dataset_type: The type of dataset (csv, json, etc.)
    """
    if dataset is None:
        st.warning("No dataset to validate.")
        return
    
    st.markdown("<h3>Dataset Validation</h3>", unsafe_allow_html=True)
    
    # Data quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate data quality metrics
    total_cells = dataset.shape[0] * dataset.shape[1]
    missing_cells = dataset.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = dataset.duplicated().sum()
    duplicate_percentage = (duplicate_rows / dataset.shape[0]) * 100 if dataset.shape[0] > 0 else 0
    
    with col1:
        st.metric("Completeness", f"{100 - missing_percentage:.2f}%")
    with col2:
        st.metric("Missing Values", f"{missing_cells:,} ({missing_percentage:.2f}%)")
    with col3:
        st.metric("Duplicate Rows", f"{duplicate_rows:,} ({duplicate_percentage:.2f}%)")
    with col4:
        # Quality score is a simple metric between 0-100 based on completeness and duplicates
        quality_score = 100 - (missing_percentage + duplicate_percentage)
        quality_score = max(0, min(100, quality_score))  # Clamp between 0 and 100
        st.metric("Quality Score", f"{quality_score:.2f}/100")
    
    # Tabs for different validation aspects
    tab1, tab2 = st.tabs(["Data Quality Issues", "Anomaly Detection"])
    
    with tab1:
        st.markdown("### Data Quality Issues")
        
        # Check for missing values by column
        missing_by_col = dataset.isna().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]
        
        if not missing_by_col.empty:
            st.markdown("#### Missing Values by Column")
            missing_df = pd.DataFrame({
                'Column': missing_by_col.index,
                'Missing Count': missing_by_col.values,
                'Percentage': (missing_by_col.values / dataset.shape[0] * 100).round(2)
            })
            missing_df['Status'] = missing_df['Percentage'].apply(
                lambda x: "🟢 Good" if x < 5 else ("🟠 Warning" if x < 20 else "🔴 Critical")
            )
            
            st.dataframe(
                missing_df.style.format({
                    'Percentage': '{:.2f}%'
                }).background_gradient(subset=['Percentage'], cmap='Reds'),
                use_container_width=True
            )
        else:
            st.success("No missing values found in the dataset!")
        
        # Check for duplicate rows
        if duplicate_rows > 0:
            st.markdown("#### Duplicate Rows")
            st.warning(f"Found {duplicate_rows} duplicate rows ({duplicate_percentage:.2f}% of the dataset)")
            
            # Option to show duplicates
            if st.checkbox("Show duplicates"):
                st.dataframe(dataset[dataset.duplicated(keep='first')], use_container_width=True)
        else:
            st.success("No duplicate rows found in the dataset!")
        
        # Check column data types
        st.markdown("#### Column Data Types")
        type_issues = []
        
        for col in dataset.columns:
            dtype = dataset[col].dtype
            if dtype == 'object':
                # Check if it could be numeric
                try:
                    # Try to convert a sample to numeric
                    sample = dataset[col].dropna().head(100)
                    if len(sample) > 0:
                        numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                        if numeric_count / len(sample) > 0.8:  # If more than 80% can be converted
                            type_issues.append({
                                'Column': col,
                                'Current Type': 'object',
                                'Suggested Type': 'numeric',
                                'Issue': 'Column contains mostly numeric values but is stored as text'
                            })
                            continue
                except:
                    pass
                
                # Check if it could be datetime
                try:
                    sample = dataset[col].dropna().head(100)
                    if len(sample) > 0:
                        datetime_count = pd.to_datetime(sample, errors='coerce').notna().sum()
                        if datetime_count / len(sample) > 0.8:  # If more than 80% can be converted
                            type_issues.append({
                                'Column': col,
                                'Current Type': 'object',
                                'Suggested Type': 'datetime',
                                'Issue': 'Column contains mostly dates but is stored as text'
                            })
                except:
                    pass
        
        if type_issues:
            st.dataframe(pd.DataFrame(type_issues), use_container_width=True)
        else:
            st.success("No data type issues detected!")
        
        # Check for column completeness
        st.markdown("#### Column Completeness Check")
        completeness_results = check_column_completeness(dataset)
        if completeness_results:
            st.dataframe(pd.DataFrame(completeness_results), use_container_width=True)
        else:
            st.success("All columns have good completeness!")
    
    with tab2:
        st.markdown("### Anomaly Detection")
        
        # Detect outliers in numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_num_col = st.selectbox("Select column to check for outliers", numeric_cols)
            
            outliers, lower_bound, upper_bound = detect_outliers(dataset[selected_num_col])
            outlier_percentage = (len(outliers) / len(dataset)) * 100
            
            st.markdown(f"#### Outliers in column: {selected_num_col}")
            st.metric("Outliers Detected", f"{len(outliers)} ({outlier_percentage:.2f}%)")
            
            st.markdown(f"""
            **Bounds for outlier detection:**
            - Lower bound: {lower_bound:.4f}
            - Upper bound: {upper_bound:.4f}
            """)
            
            if len(outliers) > 0:
                # Plot with outliers highlighted
                import plotly.express as px
                
                # Create a new column for coloring
                temp_df = dataset.copy()
                temp_df['is_outlier'] = temp_df.index.isin(outliers)
                
                fig = px.box(
                    temp_df, 
                    y=selected_num_col,
                    color='is_outlier',
                    color_discrete_map={True: "#FF5757", False: "#2563EB"},
                    title=f"Outliers in {selected_num_col}",
                    labels={"is_outlier": "Is Outlier"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to show outliers in table
                if st.checkbox("Show outlier data"):
                    st.dataframe(dataset.loc[outliers], use_container_width=True)
            else:
                st.success(f"No outliers detected in {selected_num_col}!")
        else:
            st.warning("No numeric columns found for outlier detection.")
