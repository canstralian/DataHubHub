import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Ensure necessary directories exist
os.makedirs('database/data', exist_ok=True)
os.makedirs('assets', exist_ok=True)
os.makedirs('fine_tuned_models', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="ML Dataset & Code Generation Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path("assets/custom.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Main title
st.title("📊 ML Dataset & Code Generation Manager")
st.write("A comprehensive platform for ML dataset management and code generation.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Dataset Management", "Fine-tuning", "Code Quality"])

# Display content based on the selected page
if page == "Home":
    st.header("Welcome to ML Dataset & Code Generation Manager")
    st.write("This application helps you manage ML datasets and fine-tune code generation models.")
    
    # Create two columns for the dashboard
    col1, col2 = st.columns(2)
    
    # Sample data for demonstration
    with col1:
        st.subheader("Recent Datasets")
        data = {
            "Name": ["Code Comments", "Function Docs", "API Usage"],
            "Rows": [1200, 850, 500],
            "Format": ["CSV", "JSON", "CSV"]
        }
        st.dataframe(pd.DataFrame(data))
    
    with col2:
        st.subheader("Training Jobs")
        # Create sample data with controlled values to avoid infinite extents
        training_data = {
            'Training Loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26, 0.25],
            'Validation Loss': [0.85, 0.7, 0.65, 0.6, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53],
            'Accuracy': [0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87]
        }
        chart_data = pd.DataFrame(training_data)
        st.line_chart(chart_data)

elif page == "Dataset Management":
    st.header("Dataset Management")
    st.write("Upload, explore, and manage your datasets here.")
    
    # Sample upload widget
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Display sample data
        st.subheader("Preview")
        sample_df = pd.DataFrame({
            "code": ["def hello():", "import numpy as np", "print('Hello')"],
            "comment": ["Function greeting", "Import numpy library", "Print hello message"]
        })
        st.dataframe(sample_df)
        
        # Show statistics
        st.subheader("Statistics")
        st.metric("Rows", "100")
        st.metric("Columns", "2")
        st.metric("Missing Values", "0%")

elif page == "Fine-tuning":
    st.header("Model Fine-tuning")
    st.write("Fine-tune code generation models using your datasets.")
    
    model_type = st.selectbox(
        "Select model type",
        ["CodeT5", "CodeBERT", "CodeGPT"]
    )
    
    epochs = st.slider("Number of epochs", 1, 20, 5)
    batch_size = st.slider("Batch size", 8, 128, 32)
    
    if st.button("Start Fine-tuning"):
        st.info("Fine-tuning started! This might take some time...")
        progress_bar = st.progress(0)
        
        # Simulate progress
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
        
        st.success("Fine-tuning completed!")

elif page == "Code Quality":
    st.header("Code Quality Tools")
    st.write("Tools for improving code quality and maintaining standards.")
    
    code = st.text_area("Paste your code here", "def example():\n    print('Hello, world!')")
    
    tool = st.radio("Select tool", ["Formatter", "Linter", "Type Checker"])
    
    if st.button("Run"):
        st.code(code, language="python")
        st.success(f"{tool} executed successfully!")