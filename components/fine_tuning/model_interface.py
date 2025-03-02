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

def preprocess_code_dataset(examples, tokenizer, max_input_length=512, max_target_length=128, prefix=""):
    """
    Preprocess a code dataset for training.
    
    Args:
        examples: Dataset examples with 'input' and 'target' fields
        tokenizer: Hugging Face tokenizer
        max_input_length: Maximum length for inputs
        max_target_length: Maximum length for targets
        prefix: Optional prefix to add to inputs
        
    Returns:
        Preprocessed examples with tokenized inputs and labels
    """
    # Format inputs
    inputs = [f"{prefix}{ex}" for ex in examples["input"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"], 
            max_length=max_target_length, 
            padding="max_length", 
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id in labels with -100 so that loss ignores padding
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in model_inputs["labels"][i]
        ]
    
    return model_inputs

def compute_metrics(eval_preds):
    """
    Compute metrics for model evaluation.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    preds, labels = eval_preds
    
    # Decode predicted tokens to text
    # Implementation depends on specific metrics needed (BLEU, exact match, etc.)
    
    # For now, return simple loss-based metric
    return {"sequence_accuracy": 0.0}  # Placeholder

def save_training_config(config, path):
    """
    Save training configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_training_config(path):
    """
    Load training configuration from a JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)

def setup_trainer(model, tokenizer, dataset, training_args):
    """
    Set up a Hugging Face Trainer for fine-tuning.
    
    Args:
        model: Pre-trained model
        tokenizer: Tokenizer for the model
        dataset: Dataset for fine-tuning
        training_args: Training arguments
        
    Returns:
        Trainer object
    """
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length"
    )
    
    # Split dataset into train and validation
    split_dataset = dataset.train_test_split(test_size=0.1)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    return trainer

def generate_code_comment(model, tokenizer, code_input, prefix="translate code to comment: ", max_length=64):
    """
    Generate a comment for a code snippet using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer for the model
        code_input: Input code snippet
        prefix: Prefix to add to the input
        max_length: Maximum length of the generated comment
        
    Returns:
        Generated comment
    """
    inputs = tokenizer(f"{prefix}{code_input}", return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    generated_comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_comment

def generate_code_from_comment(model, tokenizer, comment_input, prefix="translate comment to code: ", max_length=128):
    """
    Generate code from a comment using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer for the model
        comment_input: Input comment
        prefix: Prefix to add to the input
        max_length: Maximum length of the generated code
        
    Returns:
        Generated code
    """
    inputs = tokenizer(f"{prefix}{comment_input}", return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code