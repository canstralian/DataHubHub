import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
from datetime import datetime
from components.dataset_uploader import render_dataset_uploader
from components.dataset_preview import render_dataset_preview
from components.dataset_statistics import render_dataset_statistics
from components.dataset_validation import render_dataset_validation
from components.dataset_visualization import render_dataset_visualization
from components.fine_tuning import render_finetune_ui
from components.code_quality import render_code_quality_tools
from utils.huggingface_integration import search_huggingface_datasets, load_huggingface_dataset
from utils.dataset_utils import get_dataset_info, detect_dataset_format
from utils.smolagents_integration import process_with_smolagents
from database import (
    DatasetOperations,
    TrainingOperations,
    CodeQualityOperations
)

# Set page configuration
st.set_page_config(
    page_title="ML Dataset & Code Generation Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    with open("assets/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except:
    st.write("CSS file not found. Using default styling.")

# App title and description
st.markdown("""
<div class="header">
    <h1>ML Dataset & Code Generation Manager</h1>
    <p>A platform for managing ML datasets and fine-tuning code generation models with Hugging Face integration</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'database_id' not in st.session_state:
    st.session_state.database_id = None

# Sidebar navigation
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select a page",
    ["Upload Dataset", "Explore & Analyze", "Hugging Face Integration", 
     "Process with SmolaAgents", "Fine-Tune Code Models", "Code Quality Tools", "Database Management"]
)

st.sidebar.markdown("""
<div class="sidebar-footer">
    <p>Built with Streamlit & Hugging Face</p>
</div>
""", unsafe_allow_html=True)

# Main content
if page == "Upload Dataset":
    st.markdown("<h2>Upload Dataset</h2>", unsafe_allow_html=True)
    render_dataset_uploader()
    
    if st.session_state.dataset is not None:
        st.success(f"Dataset '{st.session_state.dataset_name}' loaded successfully!")
        
        # Basic dataset info and preview
        st.markdown("### Dataset Preview")
        render_dataset_preview(st.session_state.dataset, st.session_state.dataset_type)
        
        # Validation
        st.markdown("### Dataset Validation")
        render_dataset_validation(st.session_state.dataset, st.session_state.dataset_type)

elif page == "Explore & Analyze":
    st.markdown("<h2>Explore & Analyze</h2>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        # Dataset statistics
        st.markdown("### Dataset Statistics")
        render_dataset_statistics(st.session_state.dataset, st.session_state.dataset_type)
        
        # Dataset visualization
        st.markdown("### Dataset Visualization")
        render_dataset_visualization(st.session_state.dataset, st.session_state.dataset_type)

elif page == "Hugging Face Integration":
    st.markdown("<h2>Hugging Face Integration</h2>", unsafe_allow_html=True)
    
    # Search for datasets
    st.subheader("Search Hugging Face Datasets")
    search_query = st.text_input("Search datasets", "")
    search_button = st.button("Search")
    
    if search_button and search_query:
        with st.spinner("Searching Hugging Face datasets..."):
            search_results = search_huggingface_datasets(search_query)
            if search_results:
                st.success(f"Found {len(search_results)} datasets matching '{search_query}'")
                
                # Display results
                for i, dataset in enumerate(search_results):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{dataset['name']}**")
                            st.markdown(f"*{dataset.get('description', 'No description available')}*")
                        with col2:
                            if st.button("Load", key=f"load_{i}"):
                                with st.spinner(f"Loading dataset {dataset['name']}..."):
                                    try:
                                        loaded_dataset = load_huggingface_dataset(dataset['id'])
                                        st.session_state.dataset = loaded_dataset
                                        st.session_state.dataset_name = dataset['name']
                                        st.session_state.dataset_type = detect_dataset_format(loaded_dataset)
                                        st.session_state.dataset_info = get_dataset_info(loaded_dataset)
                                        st.success(f"Dataset '{dataset['name']}' loaded successfully!")
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Error loading dataset: {str(e)}")
            else:
                st.warning(f"No datasets found matching '{search_query}'")
    
    # Dataset from Hugging Face
    st.subheader("Load Dataset Directly")
    dataset_id = st.text_input("Hugging Face Dataset ID", "")
    if st.button("Load Dataset"):
        if dataset_id:
            with st.spinner(f"Loading dataset {dataset_id}..."):
                try:
                    loaded_dataset = load_huggingface_dataset(dataset_id)
                    st.session_state.dataset = loaded_dataset
                    st.session_state.dataset_name = dataset_id
                    st.session_state.dataset_type = detect_dataset_format(loaded_dataset)
                    st.session_state.dataset_info = get_dataset_info(loaded_dataset)
                    st.success(f"Dataset '{dataset_id}' loaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
        else:
            st.warning("Please enter a dataset ID")

elif page == "Process with SmolaAgents":
    st.markdown("<h2>Process with SmolaAgents</h2>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        st.markdown("""
        SmolaAgents are lightweight agents that can help process and transform your dataset.
        """)
        
        operation = st.selectbox(
            "Select operation", 
            ["Data Cleaning", "Feature Engineering", "Data Transformation", "Custom Processing"]
        )
        
        if operation == "Custom Processing":
            custom_code = st.text_area("Enter custom processing code", height=200)
        
        if st.button("Process Dataset"):
            with st.spinner("Processing dataset with SmolaAgents..."):
                try:
                    processed_dataset = process_with_smolagents(
                        st.session_state.dataset, 
                        operation,
                        custom_code if operation == "Custom Processing" else None
                    )
                    
                    st.session_state.dataset = processed_dataset
                    st.success("Dataset processed successfully!")
                    
                    # Show preview of processed dataset
                    st.subheader("Processed Dataset Preview")
                    render_dataset_preview(processed_dataset, st.session_state.dataset_type)
                    
                except Exception as e:
                    st.error(f"Error processing dataset: {str(e)}")

elif page == "Fine-Tune Code Models":
    # Render the fine-tuning UI
    render_finetune_ui()

elif page == "Code Quality Tools":
    # Render the code quality tools
    render_code_quality_tools()

elif page == "Database Management":
    st.markdown("<h2>Database Management</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Datasets", "Training Jobs", "Code Quality Checks"])
    
    with tab1:
        st.subheader("Stored Datasets")
        datasets = DatasetOperations.get_all_datasets()
        
        if not datasets:
            st.info("No datasets stored in the database yet.")
        else:
            # Display datasets table
            datasets_data = []
            for ds in datasets:
                datasets_data.append({
                    "ID": ds.id,
                    "Name": ds.name,
                    "Format": ds.format,
                    "Rows": ds.rows,
                    "Columns": ds.columns,
                    "Source": ds.source or "Unknown",
                    "Created": ds.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            st.dataframe(pd.DataFrame(datasets_data), use_container_width=True)
            
            # Select dataset for detailed view
            selected_dataset_id = st.selectbox(
                "Select dataset for details",
                options=[ds.id for ds in datasets],
                format_func=lambda x: next((ds.name for ds in datasets if ds.id == x), "Unknown")
            )
            
            if selected_dataset_id:
                dataset = DatasetOperations.get_dataset_by_id(selected_dataset_id)
                columns = DatasetOperations.get_column_info(selected_dataset_id)
                
                st.subheader(f"Dataset: {dataset.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", dataset.rows)
                with col2:
                    st.metric("Columns", dataset.columns)
                with col3:
                    st.metric("Format", dataset.format)
                
                st.markdown("#### Description")
                st.write(dataset.description or "No description available")
                
                st.markdown("#### Columns")
                if columns:
                    columns_data = []
                    for col in columns:
                        col_data = {
                            "Name": col.name,
                            "Type": col.data_type,
                            "Missing (%)": f"{col.missing_percentage:.2f}%",
                            "Unique Values": col.unique_values or "N/A"
                        }
                        
                        # Add numeric stats if available
                        if col.min_value is not None:
                            col_data["Min"] = f"{col.min_value:.2f}"
                            col_data["Max"] = f"{col.max_value:.2f}"
                            col_data["Mean"] = f"{col.mean_value:.2f}"
                            col_data["Std"] = f"{col.std_value:.2f}"
                        
                        columns_data.append(col_data)
                    
                    st.dataframe(pd.DataFrame(columns_data), use_container_width=True)
                else:
                    st.info("No column information available")
                
                # Button to delete dataset
                if st.button("Delete Dataset", key="delete_dataset"):
                    DatasetOperations.delete_dataset(selected_dataset_id)
                    st.success(f"Dataset '{dataset.name}' deleted successfully")
                    st.experimental_rerun()
        
        # Store current dataset
        if st.session_state.dataset is not None:
            st.subheader("Store Current Dataset")
            
            description = st.text_area(
                "Dataset Description", 
                value=f"Dataset loaded on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                source = st.selectbox("Source", ["Local", "Hugging Face", "Generated", "Processed", "Other"])
            with col2:
                source_url = st.text_input("Source URL/Path", value=st.session_state.get("dataset_source_url", ""))
            
            if st.button("Store in Database"):
                try:
                    # Store dataset info in database
                    stored_dataset = DatasetOperations.store_dataframe_info(
                        df=st.session_state.dataset,
                        name=st.session_state.dataset_name,
                        description=description,
                        source=source,
                        source_url=source_url
                    )
                    
                    st.session_state.database_id = stored_dataset.id
                    st.success(f"Dataset '{st.session_state.dataset_name}' stored in database with ID: {stored_dataset.id}")
                except Exception as e:
                    st.error(f"Error storing dataset: {str(e)}")
    
    with tab2:
        st.subheader("Training Jobs")
        jobs = TrainingOperations.get_all_training_jobs()
        
        if not jobs:
            st.info("No training jobs in the database yet.")
        else:
            # Display jobs table
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "ID": job.id,
                    "Name": job.name,
                    "Model": job.model_type,
                    "Task": job.task_type,
                    "Status": job.status.capitalize(),
                    "Dataset": job.dataset_id,
                    "Created": job.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            st.dataframe(pd.DataFrame(jobs_data), use_container_width=True)
            
            # Select job for detailed view
            selected_job_id = st.selectbox(
                "Select job for details",
                options=[job.id for job in jobs],
                format_func=lambda x: next((job.name for job in jobs if job.id == x), "Unknown")
            )
            
            if selected_job_id:
                job = TrainingOperations.get_training_job_by_id(selected_job_id)
                logs = TrainingOperations.get_training_logs(selected_job_id)
                
                st.subheader(f"Training Job: {job.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", job.status.capitalize())
                with col2:
                    st.metric("Model", job.model_type)
                with col3:
                    st.metric("Task", job.task_type)
                
                st.markdown("#### Description")
                st.write(job.description or "No description available")
                
                # Display hyperparameters
                if job.hyperparameters:
                    st.markdown("#### Hyperparameters")
                    for param, value in job.hyperparameters.items():
                        st.code(f"{param}: {value}")
                
                # Display metrics
                if job.metrics:
                    st.markdown("#### Metrics")
                    metrics_df = pd.DataFrame([job.metrics])
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Display logs
                if logs:
                    st.markdown("#### Training Logs")
                    logs_data = []
                    for log in logs:
                        logs_data.append({
                            "Time": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "Level": log.level,
                            "Message": log.message
                        })
                    
                    st.dataframe(pd.DataFrame(logs_data), use_container_width=True)
                
                # Button to delete job
                if st.button("Delete Job", key="delete_job"):
                    TrainingOperations.delete_training_job(selected_job_id)
                    st.success(f"Training job '{job.name}' deleted successfully")
                    st.experimental_rerun()
    
    with tab3:
        st.subheader("Code Quality Checks")
        checks = CodeQualityOperations.get_all_code_quality_checks()
        
        if not checks:
            st.info("No code quality checks in the database yet.")
        else:
            # Display checks table
            checks_data = []
            for check in checks:
                checks_data.append({
                    "ID": check.id,
                    "Filename": check.filename,
                    "Tool": check.tool,
                    "Score": check.score if check.score is not None else "N/A",
                    "Issues": check.issues_count,
                    "Created": check.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            st.dataframe(pd.DataFrame(checks_data), use_container_width=True)
            
            # Select check for detailed view
            selected_check_id = st.selectbox(
                "Select check for details",
                options=[check.id for check in checks],
                format_func=lambda x: next((f"{check.filename} ({check.tool})" for check in checks if check.id == x), "Unknown")
            )
            
            if selected_check_id:
                check = next((c for c in checks if c.id == selected_check_id), None)
                
                if check:
                    st.subheader(f"Code Quality Check: {check.filename}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tool", check.tool)
                    with col2:
                        st.metric("Score", f"{check.score:.2f}" if check.score is not None else "N/A")
                    with col3:
                        st.metric("Issues", check.issues_count)
                    
                    if check.report:
                        st.markdown("#### Report")
                        st.code(check.report)
