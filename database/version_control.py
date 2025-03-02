"""
Version control system for datasets in the ML Dataset & Code Generation Manager.
This module provides functionality to:
- Create dataset versions
- List version history
- Restore previous versions
- Compare versions
"""

import os
import json
import shutil
import hashlib
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Setup version control directory
VERSION_CONTROL_DIR = Path("database/version_control")
VERSION_CONTROL_DIR.mkdir(exist_ok=True, parents=True)

class DatasetVersion:
    """Class representing a dataset version"""
    
    def __init__(
        self, 
        dataset_id: int, 
        version_id: str, 
        timestamp: datetime.datetime, 
        metadata: Dict[str, Any],
        file_path: Path,
        parent_version: Optional[str] = None,
        description: str = ""
    ):
        self.dataset_id = dataset_id
        self.version_id = version_id
        self.timestamp = timestamp
        self.metadata = metadata
        self.file_path = file_path
        self.parent_version = parent_version
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary for serialization"""
        return {
            "dataset_id": self.dataset_id,
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "file_path": str(self.file_path),
            "parent_version": self.parent_version,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create version from dictionary"""
        return cls(
            dataset_id=data["dataset_id"],
            version_id=data["version_id"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"],
            file_path=Path(data["file_path"]),
            parent_version=data.get("parent_version"),
            description=data.get("description", "")
        )

def get_dataset_versions_path(dataset_id: int) -> Path:
    """Get path to versions file for a dataset"""
    return VERSION_CONTROL_DIR / f"dataset_{dataset_id}_versions.json"

def get_dataset_data_dir(dataset_id: int) -> Path:
    """Get directory for storing dataset version files"""
    dataset_dir = VERSION_CONTROL_DIR / f"dataset_{dataset_id}"
    dataset_dir.mkdir(exist_ok=True)
    return dataset_dir

def create_version(
    dataset_id: int, 
    df: pd.DataFrame, 
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> DatasetVersion:
    """
    Create a new version of a dataset
    
    Args:
        dataset_id: ID of the dataset
        df: Pandas DataFrame with dataset content
        description: Optional description of the changes
        metadata: Optional metadata about the version
        
    Returns:
        DatasetVersion object for the new version
    """
    if metadata is None:
        metadata = {}
    
    # Get versions file path
    versions_path = get_dataset_versions_path(dataset_id)
    
    # Load existing versions if available
    versions = []
    parent_version = None
    if versions_path.exists():
        with open(versions_path, 'r') as f:
            versions_data = json.load(f)
            versions = [DatasetVersion.from_dict(v) for v in versions_data]
            if versions:
                parent_version = versions[-1].version_id
    
    # Create version ID based on content hash and timestamp
    content_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:8]
    timestamp = datetime.datetime.now()
    version_id = f"{timestamp.strftime('%Y%m%d%H%M%S')}_{content_hash}"
    
    # Add basic metadata
    metadata.update({
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "content_hash": content_hash
    })
    
    # Save dataset content to versioned file
    dataset_dir = get_dataset_data_dir(dataset_id)
    file_path = dataset_dir / f"{version_id}.parquet"
    df.to_parquet(file_path)
    
    # Create version object
    version = DatasetVersion(
        dataset_id=dataset_id,
        version_id=version_id,
        timestamp=timestamp,
        metadata=metadata,
        file_path=file_path,
        parent_version=parent_version,
        description=description
    )
    
    # Add to versions list and save
    versions.append(version)
    
    with open(versions_path, 'w') as f:
        json.dump([v.to_dict() for v in versions], f, indent=2)
    
    return version

def get_versions(dataset_id: int) -> List[DatasetVersion]:
    """
    Get all versions of a dataset
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        List of DatasetVersion objects
    """
    versions_path = get_dataset_versions_path(dataset_id)
    
    if not versions_path.exists():
        return []
    
    with open(versions_path, 'r') as f:
        versions_data = json.load(f)
        return [DatasetVersion.from_dict(v) for v in versions_data]

def get_version(dataset_id: int, version_id: str) -> Optional[DatasetVersion]:
    """
    Get a specific version of a dataset
    
    Args:
        dataset_id: ID of the dataset
        version_id: ID of the version
        
    Returns:
        DatasetVersion object or None if not found
    """
    versions = get_versions(dataset_id)
    for version in versions:
        if version.version_id == version_id:
            return version
    return None

def get_latest_version(dataset_id: int) -> Optional[DatasetVersion]:
    """
    Get the latest version of a dataset
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        DatasetVersion object or None if no versions
    """
    versions = get_versions(dataset_id)
    if versions:
        return versions[-1]
    return None

def load_version_data(version: DatasetVersion) -> pd.DataFrame:
    """
    Load dataset data for a version
    
    Args:
        version: DatasetVersion object
        
    Returns:
        Pandas DataFrame with dataset content
    """
    if not version.file_path.exists():
        raise FileNotFoundError(f"Dataset file for version {version.version_id} not found")
    
    return pd.read_parquet(version.file_path)

def restore_version(dataset_id: int, version_id: str, description: str = "Restored version") -> DatasetVersion:
    """
    Restore a previous version of a dataset as a new version
    
    Args:
        dataset_id: ID of the dataset
        version_id: ID of the version to restore
        description: Description for the new version
        
    Returns:
        DatasetVersion object for the new version
    """
    # Get version to restore
    version_to_restore = get_version(dataset_id, version_id)
    if not version_to_restore:
        raise ValueError(f"Version {version_id} not found for dataset {dataset_id}")
    
    # Load data from the version
    df = load_version_data(version_to_restore)
    
    # Create new version with restored data
    metadata = {
        "restored_from": version_id,
        **version_to_restore.metadata
    }
    
    return create_version(
        dataset_id=dataset_id,
        df=df,
        description=description,
        metadata=metadata
    )

def compare_versions(dataset_id: int, version_id1: str, version_id2: str) -> Dict[str, Any]:
    """
    Compare two versions of a dataset
    
    Args:
        dataset_id: ID of the dataset
        version_id1: ID of the first version
        version_id2: ID of the second version
        
    Returns:
        Dictionary with comparison results
    """
    # Get versions
    version1 = get_version(dataset_id, version_id1)
    version2 = get_version(dataset_id, version_id2)
    
    if not version1 or not version2:
        missing = []
        if not version1:
            missing.append(version_id1)
        if not version2:
            missing.append(version_id2)
        raise ValueError(f"Versions not found: {', '.join(missing)}")
    
    # Load data
    df1 = load_version_data(version1)
    df2 = load_version_data(version2)
    
    # Basic comparison
    comparison = {
        "version1": version_id1,
        "version2": version_id2,
        "version1_timestamp": version1.timestamp,
        "version2_timestamp": version2.timestamp,
        "rows_diff": len(df2) - len(df1),
        "columns_diff": {},
        "columns_added": [],
        "columns_removed": []
    }
    
    # Check for added/removed columns
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    comparison["columns_added"] = list(columns2 - columns1)
    comparison["columns_removed"] = list(columns1 - columns2)
    
    # Compare common columns
    common_columns = columns1.intersection(columns2)
    for col in common_columns:
        if df1[col].dtype != df2[col].dtype:
            comparison["columns_diff"][col] = {
                "type_changed": True,
                "type1": str(df1[col].dtype),
                "type2": str(df2[col].dtype)
            }
        elif df1[col].equals(df2[col]):
            # Columns are identical
            pass
        else:
            # Columns have different values
            comparison["columns_diff"][col] = {
                "type_changed": False,
                "values_changed": True
            }
    
    return comparison

def delete_version(dataset_id: int, version_id: str) -> bool:
    """
    Delete a specific version of a dataset
    
    Args:
        dataset_id: ID of the dataset
        version_id: ID of the version to delete
        
    Returns:
        True if deleted, False if not found
    """
    versions_path = get_dataset_versions_path(dataset_id)
    
    if not versions_path.exists():
        return False
    
    with open(versions_path, 'r') as f:
        versions_data = json.load(f)
    
    # Find version to delete
    version_index = None
    file_path = None
    parent_version = None
    for i, v in enumerate(versions_data):
        if v["version_id"] == version_id:
            version_index = i
            file_path = v["file_path"]
            if i < len(versions_data) - 1:
                # Update parent version for next version
                parent_version = v.get("parent_version")
            break
    
    if version_index is None:
        return False
    
    # Remove from versions list
    del versions_data[version_index]
    
    # Update parent version for next version if needed
    if version_index < len(versions_data) and parent_version is not None:
        versions_data[version_index]["parent_version"] = parent_version
    
    # Save updated versions
    with open(versions_path, 'w') as f:
        json.dump(versions_data, f, indent=2)
    
    # Delete file
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    
    return True