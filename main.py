import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path

# Make sure necessary directories exist
os.makedirs('assets', exist_ok=True)
os.makedirs('database/data', exist_ok=True)
os.makedirs('fine_tuned_models', exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="ML Dataset & Code Generation Manager",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    """Load custom CSS styles"""
    css_dir = Path("assets")
    css_path = css_dir / "custom.css"
    
    if not css_path.exists():
        # Create assets directory if it doesn't exist
        css_dir.mkdir(exist_ok=True)
        
        # Create a basic CSS file if it doesn't exist
        with open(css_path, "w") as f:
            f.write("""
            /* Custom styles for ML Dataset & Code Generation Manager */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
                color: #1A1C1F;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                color: #1A1C1F;
                background-color: #F8F9FA;
            }
            
            .stButton button {
                background-color: #2563EB;
                color: white;
                border-radius: 4px;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 600;
            }
            
            .stButton button:hover {
                background-color: #1D4ED8;
            }
            
            /* Card styling */
            .card {
                background-color: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            
            /* Accent colors */
            .accent-primary {
                color: #2563EB;
            }
            
            .accent-secondary {
                color: #84919A;
            }
            
            .accent-success {
                color: #10B981;
            }
            
            .accent-warning {
                color: #F59E0B;
            }
            
            .accent-danger {
                color: #EF4444;
            }
            """)
    
    # Load custom CSS
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_finetune_ui():
    """
    Renders the fine-tuning UI for code generation models.
    """
    try:
        from components.fine_tuning.finetune_ui import render_finetune_ui as ft_ui
        ft_ui()
    except ImportError as e:
        st.error(f"Could not load fine-tuning UI: {e}")
        
        # Create default fine-tuning UI component if not exists
        os.makedirs("components/fine_tuning", exist_ok=True)
        if not os.path.exists("components/fine_tuning/__init__.py"):
            with open("components/fine_tuning/__init__.py", "w") as f:
                f.write('"""\nFine-tuning package for code generation models.\n"""\n')
        
        if not os.path.exists("components/fine_tuning/finetune_ui.py"):
            with open("components/fine_tuning/finetune_ui.py", "w") as f:
                f.write('''"""
Streamlit UI for fine-tuning code generation models.
"""
import streamlit as st
import pandas as pd
import os

def render_dataset_preparation():
    """
    Render the dataset preparation interface.
    """
    st.subheader("Dataset Preparation")
    st.write("Prepare your dataset for fine-tuning code generation models.")
    
    # Dataset upload
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
                
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            
            # Example of data columns mapping
            st.subheader("Column Mapping")
            
            input_col = st.selectbox("Select input column (e.g., code)", df.columns)
            target_col = st.selectbox("Select target column (e.g., comment)", df.columns)
            
            # Sample transformation
            if st.button("Apply Transformation"):
                if input_col and target_col:
                    # Example transformation: simple trim/clean
                    df[input_col] = df[input_col].astype(str).str.strip()
                    df[target_col] = df[target_col].astype(str).str.strip()
                    
                    st.write("Transformed Dataset:")
                    st.dataframe(df.head())
                    
                    # Option to save processed dataset
                    if st.button("Save Processed Dataset"):
                        processed_path = os.path.join("datasets", "processed_dataset.csv")
                        os.makedirs("datasets", exist_ok=True)
                        df.to_csv(processed_path, index=False)
                        st.success(f"Dataset saved to {processed_path}")
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

def render_model_training():
    """
    Render the model training interface.
    """
    st.subheader("Model Training")
    st.write("Configure and start training your model.")
    
    # Model selection
    model_options = [
        "Salesforce/codet5-small",
        "Salesforce/codet5-base",
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base"
    ]
    
    selected_model = st.selectbox("Select base model", model_options)
    
    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=8)
        epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=3)
        learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=0.1, value=0.0001, format="%.5f")
    
    with col2:
        max_input_length = st.number_input("Max input length", min_value=32, max_value=512, value=128)
        max_target_length = st.number_input("Max target length", min_value=32, max_value=512, value=128)
        task_type = st.selectbox("Task type", ["Code to Comment", "Comment to Code"])
    
    # Training button (placeholder)
    if st.button("Start Training"):
        st.info("Training would start here. This is a placeholder.")
        # In a real implementation, this would call the training function
        # and display a progress bar or redirect to a training monitoring page

def render_model_testing():
    """
    Render the model testing interface.
    """
    st.subheader("Model Testing")
    st.write("Test your fine-tuned model with custom inputs.")
    
    # Model selection
    st.selectbox("Select fine-tuned model", ["No models available yet"])
    
    # Test input
    if st.selectbox("Task type", ["Code to Comment", "Comment to Code"]) == "Code to Comment":
        test_input = st.text_area("Enter code to generate a comment", 
                                  value="def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    else:\\n        return fibonacci(n-1) + fibonacci(n-2)")
        placeholder = "# This function implements the Fibonacci sequence recursively..."
    else:
        test_input = st.text_area("Enter comment to generate code", 
                                 value="# A function that calculates the factorial of a number recursively")
        placeholder = "def factorial(n):\\n    if n == 0:\\n        return 1\\n    else:\\n        return n * factorial(n-1)"
    
    # Generate button (placeholder)
    if st.button("Generate"):
        st.code(placeholder, language="python")
        # In a real implementation, this would call the model inference function

def render_finetune_ui():
    """
    Render the fine-tuning UI for code generation models.
    """
    st.title("Fine-Tune Code Generation Models")
    
    tabs = st.tabs(["Dataset Preparation", "Model Training", "Model Testing"])
    
    with tabs[0]:
        render_dataset_preparation()
    
    with tabs[1]:
        render_model_training()
    
    with tabs[2]:
        render_model_testing()
''')
        
        # Try again after creating the files
        try:
            from components.fine_tuning.finetune_ui import render_finetune_ui as ft_ui
            ft_ui()
        except ImportError as e:
            st.error(f"Still could not load fine-tuning UI after creating files: {e}")
            st.info("Please restart the app to initialize the components.")

def render_code_quality_ui():
    """
    Renders the code quality tools UI.
    """
    try:
        from components.code_quality import render_code_quality_tools
        render_code_quality_tools()
    except ImportError:
        st.error("Code quality tools not found. Implementing basic version.")
        st.title("Code Quality Tools")
        st.write("This section will provide tools for code linting, formatting, and testing.")
        
        # Tabs for different code quality tools
        tabs = st.tabs(["Linting", "Formatting", "Type Checking", "Testing"])
        
        with tabs[0]:
            st.subheader("Code Linting")
            st.write("Tools for checking code quality and style.")
            st.code("# Coming soon: PyLint and Flake8 integration")
        
        with tabs[1]:
            st.subheader("Code Formatting")
            st.write("Tools for formatting code according to style guides.")
            st.code("# Coming soon: Black and isort integration")
        
        with tabs[2]:
            st.subheader("Type Checking")
            st.write("Tools for checking type annotations.")
            st.code("# Coming soon: MyPy integration")
        
        with tabs[3]:
            st.subheader("Testing")
            st.write("Tools for running tests and checking code coverage.")
            st.code("# Coming soon: PyTest integration")

def render_dataset_management_ui():
    """
    Renders the dataset management UI.
    """
    st.title("Dataset Management")
    
    # Tabs for different dataset operations
    tabs = st.tabs(["Upload", "Preview", "Statistics", "Visualization", "Validation"])
    
    with tabs[0]:
        try:
            from components.dataset_uploader import render_dataset_uploader
            render_dataset_uploader()
        except ImportError:
            st.subheader("Dataset Upload")
            st.write("Upload your datasets in CSV or JSON format.")
            
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "json"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        dataset_type = "csv"
                    else:
                        df = pd.read_json(uploaded_file)
                        dataset_type = "json"
                        
                    st.session_state["dataset"] = df
                    st.session_state["dataset_type"] = dataset_type
                    st.success(f"Successfully loaded {dataset_type.upper()} file with {df.shape[0]} rows and {df.shape[1]} columns.")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tabs[1]:
        if "dataset" in st.session_state:
            try:
                from components.dataset_preview import render_dataset_preview
                render_dataset_preview(st.session_state["dataset"], st.session_state["dataset_type"])
            except ImportError:
                st.subheader("Dataset Preview")
                st.dataframe(st.session_state["dataset"].head(10))
        else:
            st.info("Please upload a dataset first.")
    
    with tabs[2]:
        if "dataset" in st.session_state:
            try:
                from components.dataset_statistics import render_dataset_statistics
                render_dataset_statistics(st.session_state["dataset"], st.session_state["dataset_type"])
            except ImportError:
                st.subheader("Dataset Statistics")
                st.write("Basic statistics:")
                st.write(st.session_state["dataset"].describe())
                
                # Missing values
                missing_data = st.session_state["dataset"].isnull().sum()
                st.write("Missing values per column:")
                st.write(missing_data[missing_data > 0])
        else:
            st.info("Please upload a dataset first.")
    
    with tabs[3]:
        if "dataset" in st.session_state:
            try:
                from components.dataset_visualization import render_dataset_visualization
                render_dataset_visualization(st.session_state["dataset"], st.session_state["dataset_type"])
            except ImportError:
                st.subheader("Dataset Visualization")
                
                # Only show for numerical columns
                numeric_cols = st.session_state["dataset"].select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_axis = st.selectbox("X-axis", numeric_cols)
                    
                    with col2:
                        y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    fig = px.scatter(st.session_state["dataset"], x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No numerical columns available for visualization.")
        else:
            st.info("Please upload a dataset first.")
    
    with tabs[4]:
        if "dataset" in st.session_state:
            try:
                from components.dataset_validation import render_dataset_validation
                render_dataset_validation(st.session_state["dataset"], st.session_state["dataset_type"])
            except ImportError:
                st.subheader("Dataset Validation")
                
                # Simple validation checks
                st.write("Dataset Shape:", st.session_state["dataset"].shape)
                st.write("Duplicate Rows:", st.session_state["dataset"].duplicated().sum())
                
                # Missing values percentage
                missing_percent = (st.session_state["dataset"].isnull().sum() / len(st.session_state["dataset"])) * 100
                st.write("Missing Values Percentage:")
                st.write(missing_percent[missing_percent > 0])
        else:
            st.info("Please upload a dataset first.")

def main():
    """
    Main function to run the application.
    """
    # Load custom CSS
    load_css()
    
    # Sidebar for navigation
    st.sidebar.title("ML Dataset & Code Gen Manager")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Dataset Management", "Fine-Tuning", "Code Quality Tools"])
    
    # Display selected page
    if page == "Home":
        st.title("ML Dataset & Code Generation Manager")
        st.write("Welcome to the ML Dataset & Code Generation Manager. This platform helps you manage ML datasets and fine-tune code generation models.")
        
        # Main features in cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Dataset Management</h3>
                <p>Upload, analyze, visualize, and validate your ML datasets.</p>
                <ul>
                    <li>Support for CSV and JSON formats</li>
                    <li>Statistical analysis and visualization</li>
                    <li>Data validation and quality checks</li>
                    <li>Hugging Face Hub integration</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Code Quality Tools</h3>
                <p>Tools for ensuring high-quality code.</p>
                <ul>
                    <li>Code linting with PyLint</li>
                    <li>Code formatting with Black and isort</li>
                    <li>Type checking with MyPy</li>
                    <li>Testing with PyTest</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Fine-Tuning</h3>
                <p>Fine-tune code generation models on your custom datasets.</p>
                <ul>
                    <li>Support for CodeT5, CodeBERT models</li>
                    <li>Code-to-comment and comment-to-code tasks</li>
                    <li>Custom dataset preparation</li>
                    <li>Model testing and evaluation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Hugging Face Integration</h3>
                <p>Seamless integration with Hugging Face Hub.</p>
                <ul>
                    <li>Search and load models and datasets</li>
                    <li>Deploy fine-tuned models to Hugging Face Spaces</li>
                    <li>Share and collaborate on models and datasets</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Get started section
        st.subheader("Get Started")
        st.write("To get started, navigate to the Dataset Management page to upload your data, or explore the Fine-Tuning page to train code generation models.")
        
    elif page == "Dataset Management":
        render_dataset_management_ui()
        
    elif page == "Fine-Tuning":
        render_finetune_ui()
        
    elif page == "Code Quality Tools":
        render_code_quality_ui()

if __name__ == "__main__":
    main()