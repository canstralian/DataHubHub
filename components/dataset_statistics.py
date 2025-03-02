import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_dataset_statistics(dataset, dataset_type):
    """
    Renders statistical analysis of the dataset.
    
    Args:
        dataset: The dataset to analyze (pandas DataFrame)
        dataset_type: The type of dataset (csv, json, etc.)
    """
    if dataset is None:
        st.warning("No dataset to analyze.")
        return
    
    st.markdown("<h3>Dataset Statistics</h3>", unsafe_allow_html=True)
    
    # Tabs for different kinds of statistics
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Distribution Analysis", "Correlation Analysis"])
    
    with tab1:
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        # Filter only numeric columns for statistics
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Display summary statistics
            st.dataframe(dataset[numeric_cols].describe().T.style.highlight_max(axis=1, color='#FFD21E'), use_container_width=True)
            
            # Top values for categorical columns
            categorical_cols = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
            if categorical_cols:
                st.markdown("### Category Value Counts")
                selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
                
                # Show top values and their counts
                value_counts = dataset[selected_cat_col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Top 10 values in {selected_cat_col}",
                    labels={"x": selected_cat_col, "y": "Count"},
                    color_discrete_sequence=["#2563EB"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found in the dataset.")
    
    with tab2:
        # Distribution analysis
        st.markdown("### Distribution Analysis")
        
        if numeric_cols:
            selected_num_col = st.selectbox("Select numeric column", numeric_cols)
            
            # Create distribution plot
            fig = px.histogram(
                dataset, 
                x=selected_num_col,
                title=f"Distribution of {selected_num_col}",
                marginal="box",
                color_discrete_sequence=["#FFD21E"],
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic distribution stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{dataset[selected_num_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{dataset[selected_num_col].median():.2f}")
            with col3:
                st.metric("Min", f"{dataset[selected_num_col].min():.2f}")
            with col4:
                st.metric("Max", f"{dataset[selected_num_col].max():.2f}")
        else:
            st.warning("No numeric columns found in the dataset.")
    
    with tab3:
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        
        if len(numeric_cols) > 1:
            # Compute correlation matrix
            corr_matrix = dataset[numeric_cols].corr()
            
            # Plot heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=["#84919A", "#FFFFFF", "#FFD21E"],
                title="Correlation Matrix",
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlated features
            st.markdown("### Top Correlated Features")
            
            # Convert correlation matrix to a long format
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Display top 10 correlated pairs
            if corr_pairs:
                top_pairs = pd.DataFrame(corr_pairs[:10], columns=["Feature 1", "Feature 2", "Correlation"])
                st.dataframe(
                    top_pairs.style.format({
                        "Correlation": "{:.4f}"
                    }).background_gradient(subset=["Correlation"], cmap="coolwarm"),
                    use_container_width=True
                )
                
                # Scatter plot for the top correlated pair
                if corr_pairs:
                    top_pair = corr_pairs[0]
                    fig = px.scatter(
                        dataset, 
                        x=top_pair[0], 
                        y=top_pair[1],
                        title=f"Scatter plot: {top_pair[0]} vs {top_pair[1]} (Corr: {top_pair[2]:.4f})",
                        color_discrete_sequence=["#2563EB"],
                        template="simple_white"
                    )
                    fig.add_traces(
                        go.Scatter(
                            x=[None], 
                            y=[None],
                            mode='lines',
                            line=dict(color="#FFD21E", width=3),
                            name='Best Fit'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least two numeric columns for correlation analysis.")
