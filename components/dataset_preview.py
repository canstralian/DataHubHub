import streamlit as st
import pandas as pd
import json

def render_dataset_preview(dataset, dataset_type):
    """
    Renders a preview of the dataset with pagination options.
    
    Args:
        dataset: The dataset to preview (pandas DataFrame)
        dataset_type: The type of dataset (csv, json, etc.)
    """
    if dataset is None:
        st.warning("No dataset to preview.")
        return
    
    st.markdown(f"<h3>Dataset Preview: {st.session_state.dataset_name}</h3>", unsafe_allow_html=True)
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{dataset.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{dataset.shape[1]:,}")
    with col3:
        st.metric("Type", dataset_type.upper())
    
    # Preview options
    col1, col2 = st.columns([1, 3])
    with col1:
        num_rows = st.number_input("Rows to display", min_value=5, max_value=100, value=10, step=5)
    with col2:
        preview_mode = st.radio("Preview mode", ["Head", "Tail", "Sample"], horizontal=True)
    
    # Display dataset preview
    st.markdown("<div class='dataset-preview'>", unsafe_allow_html=True)
    
    if preview_mode == "Head":
        st.dataframe(dataset.head(num_rows), use_container_width=True)
    elif preview_mode == "Tail":
        st.dataframe(dataset.tail(num_rows), use_container_width=True)
    else:  # Sample
        st.dataframe(dataset.sample(min(num_rows, len(dataset))), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show dataset schema
    with st.expander("Dataset Schema"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Column Types**")
            type_df = pd.DataFrame({
                'Column': dataset.dtypes.index,
                'Type': dataset.dtypes.values.astype(str)
            })
            st.dataframe(type_df, use_container_width=True)
        
        with col2:
            st.markdown("**Missing Values**")
            missing_df = pd.DataFrame({
                'Column': dataset.columns,
                'Missing': dataset.isna().sum().values,
                'Percentage': dataset.isna().sum().values / len(dataset) * 100
            })
            st.dataframe(missing_df.style.format({
                'Percentage': '{:.2f}%'
            }), use_container_width=True)
    
    # Raw data
    with st.expander("Raw Data (First 5 records)"):
        if dataset_type == 'csv':
            st.code(dataset.head(5).to_csv(index=False), language="text")
        else:  # json or jsonl
            st.code(dataset.head(5).to_json(orient='records', indent=2), language="json")
