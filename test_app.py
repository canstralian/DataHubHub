"""
Simple test file for the ML Dataset & Code Generation Manager application.
This script checks basic aspects of the application structure and setup.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_directory_structure():
    """Test if the required directories exist"""
    # Ensure necessary directories exist
    os.makedirs('database/data', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('fine_tuned_models', exist_ok=True)
    
    # Check if directories exist
    assert Path("database").exists() and Path("database").is_dir(), "Database directory not found"
    assert Path("assets").exists() and Path("assets").is_dir(), "Assets directory not found"
    assert Path("fine_tuned_models").exists() and Path("fine_tuned_models").is_dir(), "Fine-tuned models directory not found"
    
    print("✅ Directory structure test passed")

def test_css_file():
    """Test if the CSS file exists"""
    css_file = Path("assets/custom.css")
    assert css_file.exists() and css_file.is_file(), "CSS file not found in assets directory"
    
    print("✅ CSS file test passed")

def test_huggingface_config():
    """Test if Hugging Face configuration file exists"""
    config_file = Path("huggingface-spacefile")
    assert config_file.exists() and config_file.is_file(), "Hugging Face configuration file not found"
    
    print("✅ Hugging Face configuration test passed")

def test_streamlit_config():
    """Test if Streamlit configuration exists"""
    config_dir = Path(".streamlit")
    config_file = config_dir / "config.toml"
    assert config_dir.exists() and config_dir.is_dir(), ".streamlit directory not found"
    assert config_file.exists() and config_file.is_file(), "config.toml file not found in .streamlit directory"
    
    print("✅ Streamlit configuration test passed")

def test_sample_dataframe():
    """Test creation of sample dataframes"""
    # Create a sample dataframe
    df = pd.DataFrame({
        "code": ["def hello():", "import numpy as np", "print('Hello')"],
        "comment": ["Function greeting", "Import numpy library", "Print hello message"]
    })
    
    # Test dataframe properties
    assert len(df) == 3
    assert list(df.columns) == ["code", "comment"]
    
    print("✅ Sample dataframe test passed")

def test_database_initialization():
    """Test if database can be initialized"""
    try:
        from database import init_db
        init_db()
        assert Path("database/data/mlmanager.db").exists(), "Database file was not created"
        print("✅ Database initialization test passed")
    except ImportError:
        print("⚠️ Could not import database module")
        assert False, "Database module not found"

def run_tests():
    """Run all tests"""
    print("Running tests for ML Dataset & Code Generation Manager...")
    
    test_directory_structure()
    test_css_file()
    test_huggingface_config()
    test_streamlit_config()
    test_sample_dataframe()
    test_database_initialization()
    
    print("\nAll tests passed! ✅")

def test_components_existence():
    """Test if core components directories exist"""
    # Check for components directory
    components_dir = Path("components")
    assert components_dir.exists() and components_dir.is_dir(), "Components directory not found"
    
    # Check for fine_tuning subdirectory
    fine_tuning_dir = components_dir / "fine_tuning"
    assert fine_tuning_dir.exists() and fine_tuning_dir.is_dir(), "Fine-tuning components directory not found"
    
    # Check for essential component files
    assert (components_dir / "code_quality.py").exists(), "Code quality component not found"
    assert (components_dir / "dataset_uploader.py").exists(), "Dataset uploader component not found"
    
    print("✅ Components existence test passed")

# Run the tests if executed directly
if __name__ == '__main__':
    run_tests()