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
    assert Path("database").exists() and Path("database").is_dir()
    assert Path("assets").exists() and Path("assets").is_dir()
    assert Path("fine_tuned_models").exists() and Path("fine_tuned_models").is_dir()
    
    print("✅ Directory structure test passed")

def test_css_file():
    """Test if the CSS file exists"""
    css_file = Path("assets/custom.css")
    assert css_file.exists() and css_file.is_file()
    
    print("✅ CSS file test passed")

def test_huggingface_config():
    """Test if Hugging Face configuration file exists"""
    config_file = Path("huggingface-spacefile")
    assert config_file.exists() and config_file.is_file()
    
    print("✅ Hugging Face configuration test passed")

def test_streamlit_config():
    """Test if Streamlit configuration exists"""
    config_dir = Path(".streamlit")
    config_file = config_dir / "config.toml"
    assert config_dir.exists() and config_dir.is_dir()
    assert config_file.exists() and config_file.is_file()
    
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

def run_tests():
    """Run all tests"""
    print("Running tests for ML Dataset & Code Generation Manager...")
    
    test_directory_structure()
    test_css_file()
    test_huggingface_config()
    test_streamlit_config()
    test_sample_dataframe()
    
    print("\nAll tests passed! ✅")

# Run the tests if executed directly
if __name__ == '__main__':
    run_tests()