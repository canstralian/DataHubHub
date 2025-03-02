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
                
                # Store dataset
                st.session_state.fine_tuning_dataset = df
                st.success("Dataset loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    elif dataset_source == "Manual Input":
        st.markdown("### Manual Input")
        st.info("Enter pairs of code/comment examples for fine-tuning")
        
        # Initialize empty dataframe
        if 'manual_pairs' not in st.session_state:
            st.session_state.manual_pairs = pd.DataFrame({"input": [""], "target": [""]})
        
        # Display existing pairs
        for i in range(len(st.session_state.manual_pairs)):
            col1, col2 = st.columns(2)
            with col1:
                st.text_area(
                    f"Input #{i+1}",
                    st.session_state.manual_pairs.loc[i, "input"],
                    key=f"input_{i}",
                    height=150
                )
            with col2:
                st.text_area(
                    f"Target #{i+1}",
                    st.session_state.manual_pairs.loc[i, "target"],
                    key=f"target_{i}",
                    height=150
                )
        
        # Buttons to add/remove pairs
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Example"):
                st.session_state.manual_pairs = pd.concat([
                    st.session_state.manual_pairs,
                    pd.DataFrame({"input": [""], "target": [""]})
                ]).reset_index(drop=True)
                st.experimental_rerun()
        
        with col2:
            if st.button("Remove Last Example") and len(st.session_state.manual_pairs) > 1:
                st.session_state.manual_pairs = st.session_state.manual_pairs.iloc[:-1].reset_index(drop=True)
                st.experimental_rerun()
        
        # Save button
        if st.button("Save Examples as Dataset"):
            # Update inputs and targets from session state
            for i in range(len(st.session_state.manual_pairs)):
                if f"input_{i}" in st.session_state and f"target_{i}" in st.session_state:
                    st.session_state.manual_pairs.loc[i, "input"] = st.session_state[f"input_{i}"]
                    st.session_state.manual_pairs.loc[i, "target"] = st.session_state[f"target_{i}"]
            
            # Filter out empty pairs
            filtered_df = st.session_state.manual_pairs[
                (st.session_state.manual_pairs["input"].str.strip() != "") &
                (st.session_state.manual_pairs["target"].str.strip() != "")
            ].reset_index(drop=True)
            
            if len(filtered_df) > 0:
                st.session_state.fine_tuning_dataset = filtered_df
                st.success(f"Dataset with {len(filtered_df)} examples created successfully!")
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
            value=2e-5,
            format_func=lambda x: f"{x:.1e}"
        )
        
        num_epochs = st.slider(
            "Number of Epochs",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of complete passes through the dataset"
        )
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[1, 2, 4, 8, 16, 32],
            value=4,
            help="Number of examples processed at once"
        )
        
        warmup_steps = st.slider(
            "Warmup Steps",
            min_value=0,
            max_value=1000,
            value=100,
            step=50,
            help="Steps for learning rate warmup"
        )
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        weight_decay = st.slider(
            "Weight Decay",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.01,
            help="L2 regularization factor"
        )
        
        gradient_accumulation_steps = st.slider(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=16,
            value=1,
            help="Number of steps to accumulate gradients before updating weights"
        )
        
        fp16 = st.checkbox(
            "Use Mixed Precision (fp16)",
            value=True,
            help="Use 16-bit floating-point precision to reduce memory usage"
        )
        
        save_steps = st.slider(
            "Save Checkpoint Steps",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            help="Steps between saving model checkpoints"
        )
    
    # Training progress
    st.markdown("### Training Status")
    
    # Progress bar
    progress_bar = st.progress(st.session_state.training_progress)
    
    # Status message
    status_container = st.empty()
    if st.session_state.training_status == "idle":
        status_container.info("Ready to start training")
    elif st.session_state.training_status == "running":
        status_container.warning("Training in progress...")
    elif st.session_state.training_status == "completed":
        status_container.success("Training completed successfully!")
    elif st.session_state.training_status == "failed":
        status_container.error("Training failed. See logs for details.")
    
    # Logs area
    if st.session_state.training_logs:
        with st.expander("Training Logs", expanded=True):
            for log in st.session_state.training_logs:
                st.text(log)
    
    # Start training button
    if st.session_state.training_status in ["idle", "completed", "failed"]:
        if st.button("Start Fine-Tuning"):
            # Create a unique run ID
            run_id = str(uuid.uuid4())
            st.session_state.training_run_id = run_id
            
            # Reset progress and status
            st.session_state.training_progress = 0.0
            st.session_state.training_status = "running"
            st.session_state.training_logs = ["Initializing fine-tuning..."]
            
            # Prepare output directory
            output_dir = MODELS_DIR / run_id
            output_dir.mkdir(exist_ok=True)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                fp16=fp16,
                logging_dir=str(output_dir / "logs"),
                logging_steps=10,
                save_steps=save_steps,
                evaluation_strategy="steps",
                eval_steps=100,
                load_best_model_at_end=True,
                save_total_limit=3,
                report_to="none"
            )
            
            # Save training configuration
            config = {
                "model_name": model_name,
                "task_type": task_type,
                "task_prefix": task_prefix,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "fp16": fp16,
                "dataset_size": len(st.session_state.fine_tuning_dataset),
                "timestamp": datetime.now().isoformat()
            }
            save_training_config(config, str(output_dir / "config.json"))
            
            # Convert dataset to Hugging Face Dataset format
            dataset = Dataset.from_pandas(st.session_state.fine_tuning_dataset)
            
            # Start training in a background thread
            def train_model():
                try:
                    # Add log
                    st.session_state.training_logs.append(f"Loading model {model_name}...")
                    
                    # Load model and tokenizer
                    tokenizer, model = load_model_and_tokenizer(model_name)
                    
                    # Add log
                    st.session_state.training_logs.append(f"Preprocessing dataset with {len(dataset)} examples...")
                    
                    # Preprocess dataset
                    def preprocess_function(examples):
                        return preprocess_code_dataset(examples, tokenizer, prefix=task_prefix)
                    
                    tokenized_dataset = dataset.map(preprocess_function, batched=True)
                    
                    # Add log
                    st.session_state.training_logs.append("Setting up trainer...")
                    
                    # Setup trainer
                    trainer = setup_trainer(model, tokenizer, tokenized_dataset, training_args)
                    
                    # Add log
                    st.session_state.training_logs.append("Starting training...")
                    
                    # Define callback class for progress tracking
                    class ProgressCallback(torch.utils.callbacks.ProgressCallback):
                        def on_log(self, args, state, control, logs=None, **kwargs):
                            if logs is not None and "loss" in logs:
                                log_text = f"Step {state.global_step}: Loss: {logs['loss']:.4f}"
                                if "eval_loss" in logs:
                                    log_text += f", Eval Loss: {logs['eval_loss']:.4f}"
                                st.session_state.training_logs.append(log_text)
                                
                                # Update progress
                                progress = min(state.global_step / (state.max_steps or 1), 1.0)
                                st.session_state.training_progress = progress
                    
                    # Add callback
                    trainer.add_callback(ProgressCallback())
                    
                    # Train the model
                    trainer.train()
                    
                    # Add log
                    st.session_state.training_logs.append("Training completed. Saving model...")
                    
                    # Save model and tokenizer
                    model.save_pretrained(str(output_dir / "final_model"))
                    tokenizer.save_pretrained(str(output_dir / "final_model"))
                    
                    # Store model and tokenizer in session state
                    st.session_state.trained_model = model
                    st.session_state.trained_tokenizer = tokenizer
                    
                    # Update status
                    st.session_state.training_status = "completed"
                    st.session_state.training_progress = 1.0
                    st.session_state.training_logs.append(f"Training completed successfully! Model saved at {output_dir}/final_model")
                
                except Exception as e:
                    # Log error
                    error_msg = f"Error during training: {str(e)}"
                    st.session_state.training_logs.append(error_msg)
                    st.session_state.training_logs.append(f"Stack trace: {import_traceback().format_exc()}")
                    
                    # Update status
                    st.session_state.training_status = "failed"
            
            # Import traceback for error handling
            import_traceback = lambda: __import__('traceback')
            
            # Start training thread
            threading.Thread(target=train_model).start()
            
            # Rerun to update UI
            st.experimental_rerun()
    
    # Stop training button (when training is in progress)
    elif st.session_state.training_status == "running":
        if st.button("Stop Training"):
            st.session_state.training_logs.append("Stopping training...")
            st.session_state.training_status = "failed"
            st.experimental_rerun()

def render_model_testing():
    """
    Render the model testing interface.
    """
    st.markdown("### Test & Use Model")
    
    # Check if a model is available
    if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
        st.warning("No fine-tuned model available. Please complete training first.")
        
        # Option to load from disk
        st.markdown("### Load Existing Model")
        
        # List available models
        model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "final_model").exists()]
        
        if not model_dirs:
            st.info("No saved models found.")
            return
        
        # Display available models
        model_options = {}
        for model_dir in model_dirs:
            config_path = model_dir / "config.json"
            if config_path.exists():
                try:
                    config = load_training_config(str(config_path))
                    timestamp = datetime.fromisoformat(config.get("timestamp", "2000-01-01T00:00:00"))
                    model_options[str(model_dir)] = f"{config.get('task_type', 'Unknown')} - {timestamp.strftime('%Y-%m-%d %H:%M')}"
                except:
                    model_options[str(model_dir)] = f"Model {model_dir.name}"
            else:
                model_options[str(model_dir)] = f"Model {model_dir.name}"
        
        selected_model_dir = st.selectbox(
            "Select a saved model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        if st.button("Load Selected Model"):
            try:
                # Load model and tokenizer
                model_path = Path(selected_model_dir) / "final_model"
                model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                
                # Store in session state
                st.session_state.trained_model = model
                st.session_state.trained_tokenizer = tokenizer
                
                # Load config
                config_path = Path(selected_model_dir) / "config.json"
                if config_path.exists():
                    config = load_training_config(str(config_path))
                    st.session_state.task_type = config.get("task_type", "Code to Comment")
                    st.session_state.task_prefix = config.get("task_prefix", "translate code to comment: ")
                else:
                    st.session_state.task_type = "Code to Comment"
                    st.session_state.task_prefix = "translate code to comment: "
                
                st.success("Model loaded successfully!")
                st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        
        return
    
    # Get task info
    task_type = getattr(st.session_state, 'task_type', "Code to Comment")
    task_prefix = getattr(st.session_state, 'task_prefix', "translate code to comment: ")
    
    # Input area
    st.markdown(f"### Test {task_type}")
    
    if task_type == "Code to Comment":
        test_input = st.text_area(
            "Enter code to generate a comment",
            height=200,
            help="Enter a code snippet to generate a comment"
        )
        input_label = "Code Input"
        output_label = "Generated Comment"
        generate_func = generate_code_comment
    else:  # Comment to Code
        test_input = st.text_area(
            "Enter comment to generate code",
            height=100,
            help="Enter a comment to generate code"
        )
        input_label = "Comment Input"
        output_label = "Generated Code"
        generate_func = generate_code_from_comment
    
    # Generation parameters
    with st.expander("Generation Parameters", expanded=False):
        max_length = st.slider(
            "Max Length",
            min_value=16,
            max_value=512,
            value=128,
            help="Maximum length of generated output"
        )
        
        num_beams = st.slider(
            "Beam Size",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of beams for beam search"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls randomness of generation (higher = more random)"
        )
    
    # Generate button
    if test_input and st.button("Generate"):
        with st.spinner("Generating..."):
            # Use the appropriate generation function
            model = st.session_state.trained_model
            tokenizer = st.session_state.trained_tokenizer
            
            if task_type == "Code to Comment":
                result = generate_code_comment(model, tokenizer, test_input, prefix=task_prefix, max_length=max_length)
            else:
                result = generate_code_from_comment(model, tokenizer, test_input, prefix=task_prefix, max_length=max_length)
            
            # Display results
            st.markdown(f"### {output_label}")
            st.code(result, language="text" if task_type == "Code to Comment" else "python")
            
            # Show input and output side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {input_label}")
                st.code(test_input, language="python" if task_type == "Code to Comment" else "text")
            with col2:
                st.markdown(f"#### {output_label}")
                st.code(result, language="text" if task_type == "Code to Comment" else "python")