import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
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

# Sidebar navigation
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select a page",
    ["Upload Dataset", "Explore & Analyze", "Hugging Face Integration", 
     "Process with SmolaAgents", "Fine-Tune Code Models", "Code Quality Tools"]
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
