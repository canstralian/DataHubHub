"""
Code quality tools and configuration for the application.
"""
import streamlit as st
import subprocess
import os
from pathlib import Path
import tempfile
import json

def render_code_quality_tools():
    """
    Render the code quality tools interface.
    """
    st.markdown("<h2>Code Quality Tools</h2>", unsafe_allow_html=True)
    
    # Tabs for different tools
    tab1, tab2, tab3, tab4 = st.tabs(["Linting", "Formatting", "Type Checking", "Testing"])
    
    with tab1:
        render_linting_tools()
    
    with tab2:
        render_formatting_tools()
    
    with tab3:
        render_type_checking_tools()
    
    with tab4:
        render_testing_tools()

def render_linting_tools():
    """
    Render linting tools interface.
    """
    st.markdown("### Linting with Pylint/Flake8")
    st.markdown("""
    Linting tools help identify potential errors, enforce coding standards, and encourage best practices.
    
    **Available Tools:**
    - **Pylint**: Comprehensive linter that checks for errors and enforces a coding standard
    - **Flake8**: Wrapper around PyFlakes, pycodestyle, and McCabe complexity checker
    """)
    
    # File upload for linting
    uploaded_file = st.file_uploader("Upload Python file for linting", type=["py"])
    
    linter = st.radio("Select linter", ["Pylint", "Flake8"])
    
    if uploaded_file and st.button("Run Linter"):
        with st.spinner(f"Running {linter}..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if linter == "Pylint":
                    # Run pylint
                    result = subprocess.run(
                        ["pylint", tmp_path],
                        capture_output=True,
                        text=True
                    )
                else:
                    # Run flake8
                    result = subprocess.run(
                        ["flake8", tmp_path],
                        capture_output=True,
                        text=True
                    )
                
                # Display results
                st.subheader("Linting Results")
                if result.returncode == 0:
                    st.success("No issues found!")
                else:
                    st.error("Issues found:")
                    st.code(result.stdout or result.stderr, language="text")
            
            except Exception as e:
                st.error(f"Error running {linter}: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

def render_formatting_tools():
    """
    Render code formatting tools interface.
    """
    st.markdown("### Code Formatting with Black & isort")
    st.markdown("""
    Code formatters automatically reformat your code to follow a consistent style.
    
    **Available Tools:**
    - **Black**: The uncompromising Python code formatter
    - **isort**: A utility to sort imports alphabetically and automatically separate them into sections
    """)
    
    # File upload for formatting
    uploaded_file = st.file_uploader("Upload Python file for formatting", type=["py"])
    
    formatter = st.radio("Select formatter", ["Black", "isort", "Both"])
    
    if uploaded_file and st.button("Format Code"):
        with st.spinner(f"Running {formatter}..."):
            # Get original code
            original_code = uploaded_file.getvalue().decode("utf-8")
            
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                formatted_code = ""
                
                if formatter in ["Black", "Both"]:
                    # Run black
                    result = subprocess.run(
                        ["black", tmp_path],
                        capture_output=True,
                        text=True
                    )
                    
                    with open(tmp_path, "r") as f:
                        formatted_code = f.read()
                
                if formatter in ["isort", "Both"]:
                    # If both, use the code formatted by black
                    if formatter == "Both":
                        with open(tmp_path, "w") as f:
                            f.write(formatted_code)
                    
                    # Run isort
                    result = subprocess.run(
                        ["isort", tmp_path],
                        capture_output=True,
                        text=True
                    )
                    
                    with open(tmp_path, "r") as f:
                        formatted_code = f.read()
                
                # Display results side by side
                st.subheader("Formatting Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Code")
                    st.code(original_code, language="python")
                
                with col2:
                    st.markdown("#### Formatted Code")
                    st.code(formatted_code, language="python")
            
            except Exception as e:
                st.error(f"Error running {formatter}: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

def render_type_checking_tools():
    """
    Render type checking tools interface.
    """
    st.markdown("### Type Checking with mypy")
    st.markdown("""
    Static type checking helps catch type errors before runtime.
    
    **Available Tool:**
    - **mypy**: Optional static typing for Python
    """)
    
    # File upload for type checking
    uploaded_file = st.file_uploader("Upload Python file for type checking", type=["py"])
    
    if uploaded_file and st.button("Check Types"):
        with st.spinner("Running mypy..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Run mypy
                result = subprocess.run(
                    ["mypy", tmp_path],
                    capture_output=True,
                    text=True
                )
                
                # Display results
                st.subheader("Type Checking Results")
                if result.returncode == 0:
                    st.success("No type issues found!")
                else:
                    st.error("Type issues found:")
                    st.code(result.stdout or result.stderr, language="text")
            
            except Exception as e:
                st.error(f"Error running mypy: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

def render_testing_tools():
    """
    Render testing tools interface.
    """
    st.markdown("### Testing with pytest")
    st.markdown("""
    Testing frameworks help ensure your code works as expected.
    
    **Available Tool:**
    - **pytest**: Simple and powerful testing framework
    """)
    
    # Test file upload
    test_file = st.file_uploader("Upload test file", type=["py"])
    
    # Code file upload (optional)
    code_file = st.file_uploader("Upload code file to test (optional)", type=["py"])
    
    if test_file and st.button("Run Tests"):
        with st.spinner("Running tests..."):
            # Create temporary directory for test files
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save test file
                test_path = os.path.join(tmp_dir, "test_" + test_file.name)
                with open(test_path, "wb") as f:
                    f.write(test_file.getvalue())
                
                # Save code file if provided
                if code_file:
                    code_path = os.path.join(tmp_dir, code_file.name)
                    with open(code_path, "wb") as f:
                        f.write(code_file.getvalue())
                
                try:
                    # Run pytest
                    result = subprocess.run(
                        ["pytest", "-v", test_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Display results
                    st.subheader("Test Results")
                    st.code(result.stdout, language="text")
                    
                    if result.returncode == 0:
                        st.success("All tests passed!")
                    else:
                        st.error("Some tests failed.")
                
                except Exception as e:
                    st.error(f"Error running tests: {str(e)}")

def create_pylintrc():
    """
    Create a sample pylintrc configuration file.
    """
    pylintrc = """[MASTER]
# Python version
py-version = 3.8

# Parallel processing
jobs = 1

[MESSAGES CONTROL]
# Disable specific messages
disable=
    C0111, # missing-docstring
    C0103, # invalid-name
    R0903, # too-few-public-methods
    R0913, # too-many-arguments
    W0511, # fixme

[FORMAT]
# Maximum line length
max-line-length = 100

# Expected indentation
indent-string = '    '

[DESIGN]
# Maximum number of locals for function / method body
max-locals = 15

# Maximum number of arguments for function / method
max-args = 5

# Maximum number of attributes for a class
max-attributes = 7
"""
    return pylintrc

def create_flake8_config():
    """
    Create a sample flake8 configuration file.
    """
    flake8_config = """[flake8]
max-line-length = 100
exclude = .git,__pycache__,build,dist
ignore =
    E203, # whitespace before ':'
    E501, # line too long
    W503  # line break before binary operator
"""
    return flake8_config

def create_mypy_config():
    """
    Create a sample mypy configuration file.
    """
    mypy_config = """[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False

[mypy.plugins.numpy.*]
follow_imports = skip

[mypy.plugins.pandas.*]
follow_imports = skip
"""
    return mypy_config

def create_pytest_config():
    """
    Create a sample pytest configuration file.
    """
    pytest_config = """[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
"""
    return pytest_config