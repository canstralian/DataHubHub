import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_dataset_visualization(dataset, dataset_type):
    """
    Renders visualizations for the dataset.
    
    Args:
        dataset: The dataset to visualize (pandas DataFrame)
        dataset_type: The type of dataset (csv, json, etc.)
    """
    if dataset is None:
        st.warning("No dataset to visualize.")
        return
    
    st.markdown("<h3>Dataset Visualization</h3>", unsafe_allow_html=True)
    
    # Get column types
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in dataset.columns if dataset[col].dtype == 'datetime64[ns]']
    
    # Add visualization options based on column types
    viz_type = st.selectbox(
        "Select visualization type",
        ["Distribution", "Correlation", "Categories", "Time Series", "Custom"],
        help="Choose the type of visualization to create"
    )
    
    if viz_type == "Distribution":
        if numeric_cols:
            # Select columns for distribution visualization
            selected_cols = st.multiselect(
                "Select columns to visualize", 
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if not selected_cols:
                st.warning("Please select at least one column to visualize.")
                return
            
            # Distribution plots
            if len(selected_cols) == 1:
                # Single column histogram with density curve
                col = selected_cols[0]
                fig = px.histogram(
                    dataset, 
                    x=col,
                    histnorm='probability density',
                    title=f"Distribution of {col}",
                    color_discrete_sequence=["#FFD21E"],
                    template="simple_white"
                )
                fig.add_traces(
                    go.Scatter(
                        x=dataset[col].sort_values(),
                        y=dataset[col].sort_values().reset_index(drop=True).rolling(
                            window=int(len(dataset[col])/10) if len(dataset[col]) > 10 else len(dataset[col]),
                            min_periods=1,
                            center=True
                        ).mean(),
                        mode='lines',
                        line=dict(color="#2563EB", width=3),
                        name='Smoothed'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Multiple histograms in a grid
                num_cols = min(len(selected_cols), 2)
                num_rows = (len(selected_cols) + num_cols - 1) // num_cols
                
                fig = make_subplots(
                    rows=num_rows, 
                    cols=num_cols,
                    subplot_titles=[f"Distribution of {col}" for col in selected_cols]
                )
                
                for i, col in enumerate(selected_cols):
                    row = i // num_cols + 1
                    col_pos = i % num_cols + 1
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=dataset[col],
                            name=col,
                            marker_color="#FFD21E"
                        ),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(
                    title="Distribution of Selected Features",
                    showlegend=False,
                    template="simple_white",
                    height=300 * num_rows
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show distribution statistics
            st.markdown("### Distribution Statistics")
            stats_df = dataset[selected_cols].describe().T
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("No numeric columns found for distribution visualization.")
    
    elif viz_type == "Correlation":
        if len(numeric_cols) >= 2:
            # Correlation matrix
            st.markdown("### Correlation Matrix")
            
            # Select columns for correlation
            selected_cols = st.multiselect(
                "Select columns for correlation analysis", 
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_cols) < 2:
                st.warning("Please select at least two columns for correlation analysis.")
                return
            
            # Compute correlation
            corr = dataset[selected_cols].corr()
            
            # Heatmap
            fig = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
                template="simple_white",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot matrix for selected columns
            if len(selected_cols) > 2 and len(selected_cols) <= 5:  # Limit to 5 columns for readability
                st.markdown("### Scatter Plot Matrix")
                fig = px.scatter_matrix(
                    dataset,
                    dimensions=selected_cols,
                    color_discrete_sequence=["#2563EB"],
                    title="Scatter Plot Matrix",
                    template="simple_white"
                )
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation pairs as bar chart
            st.markdown("### Top Correlation Pairs")
            
            # Get correlation pairs
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({
                        'Feature 1': corr.columns[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': corr.iloc[i, j]
                    })
            
            # Sort by absolute correlation
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)
            
            # Create bar chart
            if corr_pairs:
                # Convert to DataFrame
                corr_df = pd.DataFrame(corr_pairs)
                pair_labels = [f"{row['Feature 1']} & {row['Feature 2']}" for _, row in corr_df.iterrows()]
                
                # Bar chart
                fig = px.bar(
                    x=pair_labels,
                    y=[abs(c) for c in corr_df['Correlation']],
                    color=corr_df['Correlation'],
                    color_continuous_scale="RdBu_r",
                    labels={'x': 'Feature Pairs', 'y': 'Absolute Correlation'},
                    title="Top Feature Correlations"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least two numeric columns for correlation analysis.")
    
    elif viz_type == "Categories":
        if categorical_cols:
            # Select categorical column
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            
            # Category counts
            value_counts = dataset[selected_cat].value_counts()
            
            # Limit to top N categories if there are too many
            if len(value_counts) > 20:
                st.info(f"Showing top 20 categories out of {len(value_counts)}")
                value_counts = value_counts.head(20)
            
            # Bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Category Counts for {selected_cat}",
                labels={'x': selected_cat, 'y': 'Count'},
                color_discrete_sequence=["#FFD21E"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # If there are numeric columns, show relationship with categorical
            if numeric_cols:
                st.markdown(f"### {selected_cat} vs Numeric Features")
                selected_num = st.selectbox("Select numeric column", numeric_cols)
                
                # Box plot
                fig = px.box(
                    dataset,
                    x=selected_cat,
                    y=selected_num,
                    title=f"{selected_cat} vs {selected_num}",
                    color_discrete_sequence=["#2563EB"],
                    template="simple_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics by category
                st.markdown(f"### Statistics of {selected_num} by {selected_cat}")
                stats_by_cat = dataset.groupby(selected_cat)[selected_num].describe()
                st.dataframe(stats_by_cat, use_container_width=True)
        else:
            st.warning("No categorical columns found for category visualization.")
    
    elif viz_type == "Time Series":
        # Check if there are potential date columns
        potential_date_cols = date_cols.copy()
        
        # Also check for object columns that might be dates
        for col in categorical_cols:
            # Sample the column to check if it contains date-like strings
            sample = dataset[col].dropna().head(5).tolist()
            if sample and all('/' in str(x) or '-' in str(x) for x in sample):
                potential_date_cols.append(col)
        
        if potential_date_cols:
            date_col = st.selectbox("Select date column", potential_date_cols)
            
            # Convert to datetime if it's not already
            if dataset[date_col].dtype != 'datetime64[ns]':
                try:
                    temp_df = dataset.copy()
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                except:
                    st.error(f"Could not convert {date_col} to datetime.")
                    return
            else:
                temp_df = dataset.copy()
            
            # Select numeric column for time series
            if numeric_cols:
                value_col = st.selectbox("Select value column", numeric_cols)
                
                # Aggregate by time period
                time_period = st.selectbox(
                    "Aggregate by",
                    ["Day", "Week", "Month", "Quarter", "Year"]
                )
                
                # Set up time grouping
                if time_period == "Day":
                    temp_df['period'] = temp_df[date_col].dt.date
                elif time_period == "Week":
                    temp_df['period'] = temp_df[date_col].dt.to_period('W').dt.start_time
                elif time_period == "Month":
                    temp_df['period'] = temp_df[date_col].dt.to_period('M').dt.start_time
                elif time_period == "Quarter":
                    temp_df['period'] = temp_df[date_col].dt.to_period('Q').dt.start_time
                else:  # Year
                    temp_df['period'] = temp_df[date_col].dt.year
                
                # Aggregate data
                agg_method = st.selectbox("Aggregation method", ["Mean", "Sum", "Min", "Max", "Count"])
                agg_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Min": "min",
                    "Max": "max",
                    "Count": "count"
                }
                
                time_series = temp_df.groupby('period')[value_col].agg(agg_map[agg_method]).reset_index()
                
                # Line chart
                fig = px.line(
                    time_series,
                    x='period',
                    y=value_col,
                    title=f"{agg_method} of {value_col} by {time_period}",
                    markers=True,
                    color_discrete_sequence=["#2563EB"],
                    template="simple_white"
                )
                fig.update_layout(
                    xaxis_title=time_period,
                    yaxis_title=f"{agg_method} of {value_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show trendline option
                if st.checkbox("Show trendline"):
                    fig = px.scatter(
                        time_series,
                        x='period',
                        y=value_col,
                        trendline="ols",
                        title=f"{agg_method} of {value_col} by {time_period} with Trendline",
                        color_discrete_sequence=["#2563EB"],
                        template="simple_white"
                    )
                    fig.update_layout(
                        xaxis_title=time_period,
                        yaxis_title=f"{agg_method} of {value_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Table view of time series data
                st.dataframe(time_series, use_container_width=True)
            else:
                st.warning("No numeric columns found for time series values.")
        else:
            st.warning("No date columns found for time series visualization.")
    
    elif viz_type == "Custom":
        st.markdown("### Custom Visualization")
        st.info("Create a custom plot by selecting axes and plot type")
        
        # Select plot type
        plot_type = st.selectbox(
            "Select plot type",
            ["Scatter", "Line", "Bar", "Box", "Violin", "Histogram", "Pie", "3D Scatter"]
        )
        
        # Depending on the plot type, get required axes
        if plot_type in ["Scatter", "Line", "Bar", "3D Scatter"]:
            # For scatter/line/bar, we need x and y
            x_col = st.selectbox("X-axis", dataset.columns.tolist())
            y_col = st.selectbox("Y-axis", numeric_cols if numeric_cols else dataset.columns.tolist())
            
            # For 3D scatter, we need a z-axis
            if plot_type == "3D Scatter":
                z_col = st.selectbox("Z-axis", numeric_cols if numeric_cols else dataset.columns.tolist())
            
            # Optional color dimension
            use_color = st.checkbox("Add color dimension")
            color_col = None
            if use_color:
                color_col = st.selectbox("Color by", dataset.columns.tolist())
            
            # Create plot
            if plot_type == "Scatter":
                fig = px.scatter(
                    dataset,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col}",
                    template="simple_white"
                )
            elif plot_type == "Line":
                fig = px.line(
                    dataset.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col}",
                    template="simple_white"
                )
            elif plot_type == "Bar":
                fig = px.bar(
                    dataset,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} by {x_col}",
                    template="simple_white"
                )
            elif plot_type == "3D Scatter":
                fig = px.scatter_3d(
                    dataset,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    color=color_col,
                    title=f"3D Scatter: {x_col}, {y_col}, {z_col}",
                    template="simple_white"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type in ["Box", "Violin"]:
            # For box/violin, we need x (categorical) and y (numeric)
            x_col = st.selectbox("X-axis (categories)", categorical_cols if categorical_cols else dataset.columns.tolist())
            y_col = st.selectbox("Y-axis (values)", numeric_cols if numeric_cols else dataset.columns.tolist())
            
            # Optional color dimension
            use_color = st.checkbox("Add color dimension")
            color_col = None
            if use_color:
                color_col = st.selectbox("Color by", dataset.columns.tolist())
            
            # Create plot
            if plot_type == "Box":
                fig = px.box(
                    dataset,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"Box Plot: {y_col} by {x_col}",
                    template="simple_white"
                )
            else:  # Violin
                fig = px.violin(
                    dataset,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"Violin Plot: {y_col} by {x_col}",
                    template="simple_white"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Histogram":
            # For histogram, we need just one column
            value_col = st.selectbox("Value column", dataset.columns.tolist())
            
            # Bins option
            n_bins = st.slider("Number of bins", 5, 100, 20)
            
            # Optional color dimension
            use_color = st.checkbox("Add color dimension")
            color_col = None
            if use_color:
                color_col = st.selectbox("Color by", dataset.columns.tolist())
            
            # Create plot
            fig = px.histogram(
                dataset,
                x=value_col,
                color=color_col,
                nbins=n_bins,
                title=f"Histogram of {value_col}",
                template="simple_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Pie":
            # For pie, we need a categorical column
            cat_col = st.selectbox("Category column", categorical_cols if categorical_cols else dataset.columns.tolist())
            
            # Optional value column
            use_values = st.checkbox("Use custom values")
            value_col = None
            if use_values and numeric_cols:
                value_col = st.selectbox("Value column", numeric_cols)
            
            # Limit to top N categories if there are too many
            top_n = st.slider("Limit to top N categories", 0, 20, 10, 
                help="Set to 0 to show all categories. Recommended to limit to top 10-15 categories for readability.")
            
            # Process data for pie chart
            if top_n > 0:
                if use_values and value_col:
                    pie_data = dataset.groupby(cat_col)[value_col].sum().reset_index()
                    pie_data = pie_data.sort_values(value_col, ascending=False).head(top_n)
                else:
                    value_counts = dataset[cat_col].value_counts().reset_index()
                    value_counts.columns = [cat_col, 'count']
                    pie_data = value_counts.head(top_n)
                    value_col = 'count'
            else:
                if use_values and value_col:
                    pie_data = dataset.groupby(cat_col)[value_col].sum().reset_index()
                else:
                    value_counts = dataset[cat_col].value_counts().reset_index()
                    value_counts.columns = [cat_col, 'count']
                    pie_data = value_counts
                    value_col = 'count'
            
            # Create plot
            fig = px.pie(
                pie_data,
                names=cat_col,
                values=value_col,
                title=f"Pie Chart of {cat_col}",
                template="simple_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
