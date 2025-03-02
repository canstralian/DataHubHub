import streamlit as st
import pandas as pd
import json
import io
from utils.dataset_utils import get_dataset_info, detect_dataset_format

def render_dataset_uploader():
    """
    Renders the dataset upload component that supports CSV and JSON formats.
    """
    st.markdown("""
    <div class="upload-container">
        <p>Upload your dataset in CSV or JSON format</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "json"], 
        help="Upload a CSV or JSON file containing your dataset"
    )
    
    # Sample dataset option
    st.markdown("Or use a sample dataset:")
    sample_dataset = st.selectbox(
        "Select a sample dataset",
        ["None", "Iris Dataset", "Titanic Dataset", "Boston Housing Dataset"]
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Check file extension
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
                dataset_type = "csv"
            elif file_extension == "json":
                # Try different JSON formats
                try:
                    # First try parsing as a regular JSON with records orientation
                    df = pd.read_json(uploaded_file)
                    dataset_type = "json"
                except:
                    # If that fails, try to parse as JSON Lines
                    try:
                        df = pd.read_json(uploaded_file, lines=True)
                        dataset_type = "jsonl"
                    except:
                        # If that also fails, load raw JSON and convert
                        content = json.loads(uploaded_file.getvalue().decode("utf-8"))
                        if isinstance(content, list):
                            df = pd.DataFrame(content)
                        elif isinstance(content, dict):
                            # Handle nested dict structures
                            if any(isinstance(v, list) for v in content.values()):
                                # Find the list field and use it
                                for key, value in content.items():
                                    if isinstance(value, list):
                                        df = pd.DataFrame(value)
                                        break
                            else:
                                # Flat dict or dict of dicts
                                df = pd.DataFrame([content])
                        dataset_type = "json"
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return
            
            # Store dataset and its info in session state
            st.session_state.dataset = df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.dataset_type = dataset_type
            st.session_state.dataset_info = get_dataset_info(df)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Process sample dataset
    elif sample_dataset != "None":
        try:
            if sample_dataset == "Iris Dataset":
                # Load Iris dataset
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                df['target'] = iris.target
                dataset_type = "csv"
                
            elif sample_dataset == "Titanic Dataset":
                # URL for Titanic dataset
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                df = pd.read_csv(url)
                dataset_type = "csv"
                
            elif sample_dataset == "Boston Housing Dataset":
                # Load Boston Housing dataset
                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()
                df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
                df['target'] = housing.target
                dataset_type = "csv"
            
            # Store dataset and its info in session state
            st.session_state.dataset = df
            st.session_state.dataset_name = sample_dataset
            st.session_state.dataset_type = dataset_type
            st.session_state.dataset_info = get_dataset_info(df)
            
        except Exception as e:
            st.error(f"Error loading sample dataset: {str(e)}")
