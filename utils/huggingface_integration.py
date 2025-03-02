import streamlit as st
import pandas as pd
import os
from huggingface_hub import HfApi, list_datasets
from datasets import load_dataset

@st.cache_data(ttl=3600)
def search_huggingface_datasets(query, limit=20):
    """
    Search for datasets on Hugging Face Hub.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of dataset metadata
    """
    try:
        api = HfApi()
        datasets = list_datasets(
            filter=query,
            limit=limit
        )
        
        # Convert to list of dicts with relevant info
        results = []
        for dataset in datasets:
            results.append({
                'id': dataset.id,
                'name': dataset.id.split('/')[-1],
                'description': dataset.description or "No description available",
                'author': dataset.author or "Unknown",
                'tags': dataset.tags,
                'downloads': dataset.downloads
            })
        
        return results
    except Exception as e:
        st.error(f"Error searching Hugging Face Hub: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def load_huggingface_dataset(dataset_id, split='train'):
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_id: ID of the dataset on HF Hub (e.g., 'mnist', 'glue', etc.)
        split: Dataset split to load (e.g., 'train', 'test', 'validation')
        
    Returns:
        Pandas DataFrame containing the dataset
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_id, split=split)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset '{dataset_id}': {str(e)}")
        raise

def upload_to_huggingface(dataset, dataset_name, token=None):
    """
    Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset: Pandas DataFrame to upload
        dataset_name: Name for the dataset
        token: Hugging Face API token (optional, will use environment variable if not provided)
        
    Returns:
        URL to the uploaded dataset
    """
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("No Hugging Face token provided. Set the HF_TOKEN environment variable or pass a token.")
    
    try:
        # Convert to HF dataset
        from datasets import Dataset
        hf_dataset = Dataset.from_pandas(dataset)
        
        # Upload to HF Hub
        push_result = hf_dataset.push_to_hub(
            dataset_name,
            token=token
        )
        
        return f"https://huggingface.co/datasets/{push_result.repo_id}"
    except Exception as e:
        st.error(f"Error uploading to Hugging Face Hub: {str(e)}")
        raise
