"""
Streamlit UI for fine-tuning code generation models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import uuid
import threading
from transformers import TrainingArguments
from datasets import Dataset

from components.fine_tuning.model_interface import (
    load_model_and_tokenizer,
    preprocess_code_dataset,
    setup_trainer,
    generate_code_comment,
    generate_code_from_comment,
    save_training_config,
    load_training_config
)

# Initialize training state
if 'training_run_id' not in st.session_state:
    st.session_state.training_run_id = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = "idle"  # idle, running, completed, failed
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0.0
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_tokenizer' not in st.session_state:
    st.session_state.trained_tokenizer = None
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'fine_tuning_dataset' not in st.session_state:
    st.session_state.fine_tuning_dataset = None

# Directory for saving models
MODELS_DIR = Path("./fine_tuned_models")
MODELS_DIR.mkdir(exist_ok=True)

# Set for background training thread
training_thread = None

def render_dataset_preparation():
    """
    Render the dataset preparation interface.
    """
    st.markdown("### Dataset Preparation")

    # Dataset input options
    dataset_source = st.radio(
        "Choose dataset source",
        ["Upload CSV", "Manual Input", "Use Current Dataset"],
        help="Select how you want to provide your fine-tuning dataset"
    )

    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload fine-tuning dataset (CSV)",
            type=["csv"],
            help="CSV file with 'input' and 'target' columns"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Check if required columns exist
                if "input" not in df.columns or "target" not in df.columns:
                    st.error("CSV must contain 'input' and 'target' columns.")
                    return

                # Preview dataset
                st.markdown("### Dataset Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Dataset statistics
                st.markdown("### Dataset Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of examples", len(df))
                with col2:
                    st.metric("Average input length", df["input"].astype(str).str.len().mean().round(1))

                # Save dataset
                if st.button("Use this dataset"):
                    st.session_state.fine_tuning_dataset = df
                    st.success(f"Dataset with {len(df)} examples loaded successfully!")

            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    elif dataset_source == "Manual Input":
        st.markdown("""
        Enter pairs of inputs and targets for fine-tuning. For code-to-comment tasks, the input is code and 
        the target is a comment. For comment-to-code tasks, the input is a comment and the target is code.
        """)

        # Container for input fields
        examples_container = st.container()

        # Default number of example fields
        if "num_examples" not in st.session_state:
            st.session_state.num_examples = 3

        # Add more examples button
        if st.button("Add another example"):
            st.session_state.num_examples += 1

        # Input fields for examples
        inputs = []
        targets = []

        with examples_container:
            for i in range(st.session_state.num_examples):
                st.markdown(f"### Example {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    input_text = st.text_area(f"Input {i+1}", key=f"input_{i}", height=150)
                    inputs.append(input_text)
                with col2:
                    target_text = st.text_area(f"Target {i+1}", key=f"target_{i}", height=150)
                    targets.append(target_text)

        # Create dataset from manual input
        if st.button("Create Dataset from Examples"):
            # Filter out empty examples
            valid_examples = [(inp, tgt) for inp, tgt in zip(inputs, targets) if inp.strip() and tgt.strip()]

            if valid_examples:
                df = pd.DataFrame(valid_examples, columns=["input", "target"])
                st.session_state.fine_tuning_dataset = df

                # Preview dataset
                st.markdown("### Dataset Preview")
                st.dataframe(df, use_container_width=True)
                st.success(f"Dataset with {len(df)} examples created successfully!")
            else:
                st.warning("No valid examples found. Please enter at least one input-target pair.")

    elif dataset_source == "Use Current Dataset":
        if st.session_state.dataset is None:
            st.warning("No dataset is currently loaded. Please upload or select a dataset first.")
        else:
            st.markdown("### Current Dataset")
            st.dataframe(st.session_state.dataset.head(), use_container_width=True)

            # Column selection
            col1, col2 = st.columns(2)
            with col1:
                input_col = st.selectbox("Select column for inputs", st.session_state.dataset.columns)
            with col2:
                target_col = st.selectbox("Select column for targets", st.session_state.dataset.columns)

            # Create fine-tuning dataset
            if st.button("Create Fine-Tuning Dataset"):
                df = st.session_state.dataset[[input_col, target_col]].copy()
                df.columns = ["input", "target"]

                # Verify data types and convert to string if necessary
                df["input"] = df["input"].astype(str)
                df["target"] = df["target"].astype(str)

                # Preview
                st.dataframe(df.head(), use_container_width=True)

                # Store dataset
                st.session_state.fine_tuning_dataset = df
                st.success(f"Fine-tuning dataset with {len(df)} examples created successfully!")

def render_model_training():
    """
    Render the model training interface.
    """
    st.markdown("### Model Training")

    # Check if dataset is available
    if st.session_state.fine_tuning_dataset is None:
        st.warning("Please prepare a dataset in the 'Dataset Preparation' tab first.")
        return

    # Model selection
    model_options = {
        "Salesforce/codet5-small": "CodeT5 Small (60M params)",
        "Salesforce/codet5-base": "CodeT5 Base (220M params)",
        "Salesforce/codet5-large": "CodeT5 Large (770M params)",
        "microsoft/codebert-base": "CodeBERT Base (125M params)",
        "facebook/bart-base": "BART Base (140M params)"
    }

    model_name = st.selectbox(
        "Select pre-trained model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        help="Select the base model for fine-tuning"
    )

    # Task type
    task_type = st.selectbox(
        "Select task type",
        ["Code to Comment", "Comment to Code"],
        help="Choose the direction of your task"
    )

    # Task prefix
    if task_type == "Code to Comment":
        task_prefix = "translate code to comment: "
    else:
        task_prefix = "translate comment to code: "

    # Hyperparameters
    st.markdown("### Training Hyperparameters")

    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            value=5e-5,
            help="Step size for optimizer updates"
        )
        epochs = st.slider(
            "Epochs",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of complete passes through the dataset"
        )
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[1, 2, 4, 8, 16, 32],
            value=8,
            help="Number of examples processed in each training step"
        )
        max_input_length = st.slider(
            "Max Input Length (tokens)",
            min_value=64,
            max_value=512,
            value=256,
            help="Maximum length of input sequences"
        )

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            weight_decay = st.select_slider(
                "Weight Decay",
                options=[0.0, 0.01, 0.05, 0.1],
                value=0.01,
                help="L2 regularization"
            )
            warmup_steps = st.slider(
                "Warmup Steps",
                min_value=0,
                max_value=1000,
                value=100,
                help="Steps for learning rate warmup"
            )
        with col2:
            max_target_length = st.slider(
                "Max Target Length (tokens)",
                min_value=64,
                max_value=512,
                value=256,
                help="Maximum length of target sequences"
            )
            gradient_accumulation = st.slider(
                "Gradient Accumulation Steps",
                min_value=1,
                max_value=16,
                value=1,
                help="Number of steps to accumulate gradients"
            )

    # Model output configuration
    st.markdown("### Model Output Configuration")
    model_name_custom = st.text_input(
        "Custom model name",
        value=f"{model_name.split('/')[-1]}-finetuned-{task_type.lower().replace(' ', '-')}",
        help="Name for your fine-tuned model"
    )

    # Training controls
    st.markdown("### Training Controls")

    # Check if training is in progress
    if st.session_state.training_status == "running":
        # Display progress
        st.progress(st.session_state.training_progress)

        # Show logs
        if st.session_state.training_logs:
            st.markdown("### Training Logs")
            log_text = "\n".join(st.session_state.training_logs[-10:])  # Show last 10 logs
            st.text_area("Latest logs", log_text, height=200, disabled=True)

        # Stop button
        if st.button("Stop Training"):
            # Logic to stop training thread
            st.session_state.training_status = "stopping"
            st.warning("Stopping training after current epoch completes...")

    elif st.session_state.training_status == "completed":
        st.success(f"Training completed! Model saved as: {model_name_custom}")

        # Show metrics if available
        if "training_metrics" in st.session_state:
            st.markdown("### Training Metrics")
            metrics_df = pd.DataFrame(st.session_state.training_metrics)
            st.line_chart(metrics_df)

        # Reset button
        if st.button("Start New Training"):
            st.session_state.training_status = "idle"
            st.session_state.training_progress = 0.0
            st.session_state.training_logs = []
            st.experimental_rerun()

    else:  # idle or failed
        # If previously failed, show error
        if st.session_state.training_status == "failed":
            st.error("Previous training failed. See logs for details.")
            if st.session_state.training_logs:
                st.text_area("Error logs", "\n".join(st.session_state.training_logs[-5:]), height=100, disabled=True)

        # Start training button
        if st.button("Start Training"):
            # Validate dataset
            if len(st.session_state.fine_tuning_dataset) < 5:
                st.warning("Dataset is very small. Consider adding more examples for better results.")

            # Set up training configuration
            training_config = {
                "model_name": model_name,
                "task_type": task_type,
                "task_prefix": task_prefix,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_input_length": max_input_length,
                "max_target_length": max_target_length,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "gradient_accumulation": gradient_accumulation,
                "output_model_name": model_name_custom,
                "dataset_size": len(st.session_state.fine_tuning_dataset)
            }

            # Update session state
            st.session_state.training_status = "running"
            st.session_state.training_progress = 0.0
            st.session_state.training_logs = ["Training initialized..."]
            st.session_state.training_run_id = str(uuid.uuid4())

            # TODO: Start actual training process using transformers
            st.info("Training would start here with the Hugging Face transformers library")

            # For now, just simulate training progress
            st.session_state.training_progress = 0.1
            st.session_state.training_logs.append("Loaded model and tokenizer")
            st.session_state.training_logs.append("Preprocessing dataset...")

            # Rerun to update UI with progress
            st.experimental_rerun()

def render_model_testing():
    """
    Render the model testing interface.
    """
    st.markdown("### Test & Use Model")

    # Check if a model is trained/available
    if st.session_state.trained_model is None and st.session_state.training_status != "completed":
        # Look for saved models
        saved_models = list(MODELS_DIR.glob("*/"))
        if not saved_models:
            st.warning("No trained models available. Please train a model first.")
            return

        # Let user select a saved model
        model_options = [model.name for model in saved_models]
        selected_model = st.selectbox("Select a saved model", model_options)

        if st.button("Load Selected Model"):
            st.info(f"Loading model {selected_model}...")
            # TODO: Load model logic
            st.session_state.trained_model = "loaded"  # Placeholder
            st.session_state.trained_tokenizer = "loaded"  # Placeholder
            st.success("Model loaded successfully!")

    else:
        # Model is available for testing
        model_type = "Code to Comment" if "code-to-comment" in st.session_state.get("model_name", "") else "Comment to Code"

        st.markdown(f"### Testing {model_type} Generation")

        if model_type == "Code to Comment":
            input_text = st.text_area(
                "Enter code snippet",
                height=200,
                help="Enter a code snippet to generate a comment"
            )

            if st.button("Generate Comment"):
                if input_text:
                    with st.spinner("Generating comment..."):
                        # TODO: Replace with actual model inference
                        result = f"/* This code {input_text.split()[0:3]} ... */"
                        st.markdown("### Generated Comment")
                        st.code(result)
                else:
                    st.warning("Please enter a code snippet.")

        else:  # Comment to Code
            input_text = st.text_area(
                "Enter comment/description",
                height=150,
                help="Enter a description to generate code"
            )

            language = st.selectbox(
                "Programming language",
                ["Python", "JavaScript", "Java", "C++", "Go"]
            )

            if st.button("Generate Code"):
                if input_text:
                    with st.spinner("Generating code..."):
                        # TODO: Replace with actual model inference
                        result = f"def example_function():\n    # {input_text}\n    pass"
                        st.markdown("### Generated Code")
                        st.code(result, language=language.lower())
                else:
                    st.warning("Please enter a comment or description.")

        # Batch testing
        with st.expander("Batch Testing"):
            st.markdown("Upload a CSV file with test cases to evaluate your model.")

            test_file = st.file_uploader(
                "Upload test cases (CSV)",
                type=["csv"],
                help="CSV file with 'input' and 'expected' columns"
            )

            if test_file is not None:
                try:
                    test_df = pd.read_csv(test_file)
                    st.dataframe(test_df.head(), use_container_width=True)

                    if st.button("Run Batch Test"):
                        with st.spinner("Running tests..."):
                            # TODO: Actual batch inference
                            st.success("Batch testing completed!")

                            # Dummy results
                            results = pd.DataFrame({
                                "input": test_df["input"],
                                "expected": test_df.get("expected", [""] * len(test_df)),
                                "generated": ["Sample output " + str(i) for i in range(len(test_df))],
                                "match_score": np.random.uniform(0.5, 1.0, len(test_df))
                            })

                            st.dataframe(results, use_container_width=True)

                            # Metrics
                            st.markdown("### Evaluation Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Average Match Score", f"{results['match_score'].mean():.2f}")
                            with col2:
                                st.metric("Tests Passed", f"{sum(results['match_score'] > 0.8)}/{len(results)}")

                except Exception as e:
                    st.error(f"Error loading test file: {str(e)}")

def render_finetune_ui():
    """
    Render the fine-tuning UI for code generation models.
    """
    st.markdown("<h2>Fine-Tune Code Generation Model</h2>", unsafe_allow_html=True)

    # Overview and instructions
    with st.expander("About Fine-Tuning", expanded=False):
        st.markdown("""
        ## Fine-Tuning a Code Generation Model

        This interface allows you to fine-tune pre-trained code generation models from Hugging Face
        on your custom dataset to adapt them to your specific coding style or task.

        ### How to use:
        1. **Prepare your dataset** - Upload a CSV file with 'input' and 'target' columns:
           - For code-to-comment: 'input' = code snippets, 'target' = corresponding comments
           - For comment-to-code: 'input' = comments, 'target' = corresponding code snippets

        2. **Configure training** - Set hyperparameters like learning rate, batch size, and epochs

        3. **Start fine-tuning** - Launch the training process and monitor progress

        4. **Test your model** - Once training is complete, test your model on new inputs

        ### Tips for better results:
        - Use a consistent format for your code snippets and comments
        - Start with a small dataset (50-100 examples) to verify the process
        - Try different hyperparameters to find the best configuration
        """)

    # Main UI with tabs
    tab1, tab2, tab3 = st.tabs(["Dataset Preparation", "Model Training", "Test & Use Model"])

    with tab1:
        render_dataset_preparation()

    with tab2:
        render_model_training()

    with tab3:
        render_model_testing()