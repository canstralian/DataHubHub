
"""
Hugging Face model interface for code generation fine-tuning.
"""
import streamlit as st
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import numpy as np
import time
import os
from pathlib import Path
import uuid
import json

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name):
    """
    Load a pre-trained model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the model on Hugging Face (e.g., 'Salesforce/codet5-base')
        
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def preprocess_code_dataset(dataset_df, tokenizer, max_input_length=256, max_target_length=256, task_prefix=""):
    """
    Preprocess the code dataset for fine-tuning.
    
    Args:
        dataset_df: Pandas DataFrame with 'input' and 'target' columns
        tokenizer: HuggingFace tokenizer
        max_input_length: Maximum length for input sequences
        max_target_length: Maximum length for target sequences
        task_prefix: Prefix to add to inputs (e.g., "translate code to comment: ")
        
    Returns:
        HuggingFace Dataset ready for training
    """
    def preprocess_function(examples):
        inputs = [task_prefix + text for text in examples["input"]]
        targets = examples["target"]
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert DataFrame to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(dataset_df)
    
    # Split dataset into train and validation
    splits = hf_dataset.train_test_split(test_size=0.1)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    
    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "target"]
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "target"]
    )
    
    return train_dataset, eval_dataset

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir, training_args):
    """
    Set up the Trainer for fine-tuning.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        train_dataset: Preprocessed training dataset
        eval_dataset: Preprocessed evaluation dataset
        output_dir: Directory to save model and checkpoints
        training_args: Dictionary of training arguments
        
    Returns:
        HuggingFace Trainer
    """
    # Define training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_args.get("batch_size", 8),
        per_device_eval_batch_size=training_args.get("batch_size", 8),
        learning_rate=training_args.get("learning_rate", 5e-5),
        num_train_epochs=training_args.get("epochs", 3),
        weight_decay=training_args.get("weight_decay", 0.01),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        gradient_accumulation_steps=training_args.get("gradient_accumulation", 1),
        warmup_steps=training_args.get("warmup_steps", 100),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer

def generate_code_comment(model, tokenizer, code, max_length=100, task_prefix="translate code to comment: "):
    """
    Generate a comment for a given code snippet.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        code: Input code snippet
        max_length: Maximum length of the generated comment
        task_prefix: Prefix to add to the input
        
    Returns:
        Generated comment as string
    """
    inputs = tokenizer(task_prefix + code, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    output_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    comment = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return comment

def generate_code_from_comment(model, tokenizer, comment, max_length=200, task_prefix="translate comment to code: "):
    """
    Generate code from a given comment/description.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        comment: Input comment or description
        max_length: Maximum length of the generated code
        task_prefix: Prefix to add to the input
        
    Returns:
        Generated code as string
    """
    inputs = tokenizer(task_prefix + comment, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    output_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return code

def save_training_config(output_dir, config):
    """
    Save training configuration to a JSON file.
    
    Args:
        output_dir: Directory to save the configuration
        config: Dictionary with training configuration
    """
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def load_training_config(model_dir):
    """
    Load training configuration from a JSON file.
    
    Args:
        model_dir: Directory with the saved model
        
    Returns:
        Dictionary with training configuration
    """
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
