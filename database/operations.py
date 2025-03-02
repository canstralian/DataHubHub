"""
Database operations for the ML Dataset & Code Generation Manager.
"""
from sqlalchemy.orm import Session
from database.models import (
    Dataset,
    DatasetColumn,
    DatasetVersion,
    TrainingJob,
    TrainingLog,
    CodeQualityCheck,
    get_session,
    init_db
)
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Initialize the database
init_db()

class DatasetOperations:
    """
    Operations for working with datasets in the database.
    """
    
    @staticmethod
    def create_dataset(name, description, format, rows, columns, source=None, source_url=None, additional_data=None):
        """
        Create a new dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            format: Dataset format (csv, json, etc.)
            rows: Number of rows
            columns: Number of columns
            source: Source of the dataset (local, huggingface, etc.)
            source_url: URL or path to the dataset source
            additional_data: Additional metadata
            
        Returns:
            Newly created Dataset object
        """
        with get_session() as session:
            dataset = Dataset(
                name=name,
                description=description,
                format=format,
                rows=rows,
                columns=columns,
                source=source,
                source_url=source_url,
                additional_data=additional_data
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset
    
    @staticmethod
    def get_all_datasets():
        """
        Get all datasets.
        
        Returns:
            List of Dataset objects
        """
        with get_session() as session:
            return session.query(Dataset).all()
    
    @staticmethod
    def get_dataset_by_id(dataset_id):
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset object
        """
        with get_session() as session:
            return session.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    @staticmethod
    def get_dataset_by_name(name):
        """
        Get a dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset object
        """
        with get_session() as session:
            return session.query(Dataset).filter(Dataset.name == name).first()
    
    @staticmethod
    def update_dataset(dataset_id, **kwargs):
        """
        Update a dataset.
        
        Args:
            dataset_id: Dataset ID
            **kwargs: Fields to update
            
        Returns:
            Updated Dataset object
        """
        with get_session() as session:
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                for key, value in kwargs.items():
                    if hasattr(dataset, key):
                        setattr(dataset, key, value)
                session.commit()
                session.refresh(dataset)
            return dataset
    
    @staticmethod
    def delete_dataset(dataset_id):
        """
        Delete a dataset.
        
        Args:
            dataset_id: Dataset ID
        """
        with get_session() as session:
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                session.delete(dataset)
                session.commit()
    
    @staticmethod
    def add_column_info(dataset_id, name, data_type, **kwargs):
        """
        Add column information for a dataset.
        
        Args:
            dataset_id: Dataset ID
            name: Column name
            data_type: Column data type
            **kwargs: Additional fields
            
        Returns:
            Newly created DatasetColumn object
        """
        with get_session() as session:
            column = DatasetColumn(
                dataset_id=dataset_id,
                name=name,
                data_type=data_type,
                **kwargs
            )
            session.add(column)
            session.commit()
            session.refresh(column)
            return column
    
    @staticmethod
    def get_column_info(dataset_id):
        """
        Get column information for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of DatasetColumn objects
        """
        with get_session() as session:
            return session.query(DatasetColumn).filter(DatasetColumn.dataset_id == dataset_id).all()
    
    @staticmethod
    def store_dataframe_info(df, name, description=None, source=None, source_url=None):
        """
        Store information about a pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            name: Dataset name
            description: Dataset description
            source: Source of the dataset
            source_url: URL or path to the dataset source
            
        Returns:
            Newly created Dataset object
        """
        # Create dataset
        dataset = DatasetOperations.create_dataset(
            name=name,
            description=description or f"Dataset {name}",
            format="pandas",
            rows=len(df),
            columns=len(df.columns),
            source=source,
            source_url=source_url,
            additional_data={
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            }
        )
        
        # Add column information
        for col in df.columns:
            # Get column stats
            col_stats = {}
            
            # Basic stats for all columns
            col_stats["missing_values"] = df[col].isna().sum()
            col_stats["missing_percentage"] = (df[col].isna().sum() / len(df)) * 100
            col_stats["unique_values"] = df[col].nunique()
            
            # For numeric columns, add additional stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats["min_value"] = float(df[col].min()) if not df[col].isna().all() else None
                col_stats["max_value"] = float(df[col].max()) if not df[col].isna().all() else None
                col_stats["mean_value"] = float(df[col].mean()) if not df[col].isna().all() else None
                col_stats["median_value"] = float(df[col].median()) if not df[col].isna().all() else None
                col_stats["std_value"] = float(df[col].std()) if not df[col].isna().all() else None
            
            # Add column info
            DatasetOperations.add_column_info(
                dataset_id=dataset.id,
                name=col,
                data_type=str(df[col].dtype),
                **col_stats
            )
        
        return dataset

class TrainingOperations:
    """
    Operations for working with training jobs in the database.
    """
    
    @staticmethod
    def create_training_job(name, dataset_id, model_type, task_type, description=None, hyperparameters=None):
        """
        Create a new training job.
        
        Args:
            name: Job name
            dataset_id: Dataset ID
            model_type: Model type (CodeT5, CodeBERT, etc.)
            task_type: Task type (Code to Comment, Comment to Code, etc.)
            description: Job description
            hyperparameters: Hyperparameters for training
            
        Returns:
            Newly created TrainingJob object
        """
        with get_session() as session:
            job = TrainingJob(
                name=name,
                dataset_id=dataset_id,
                model_type=model_type,
                task_type=task_type,
                description=description,
                hyperparameters=hyperparameters
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
            
    @staticmethod
    def delete_training_job(job_id):
        """
        Delete a training job.
        
        Args:
            job_id: Job ID
        """
        with get_session() as session:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                session.delete(job)
                session.commit()
    
    @staticmethod
    def get_all_training_jobs():
        """
        Get all training jobs.
        
        Returns:
            List of TrainingJob objects
        """
        with get_session() as session:
            return session.query(TrainingJob).all()
    
    @staticmethod
    def get_training_jobs_by_dataset(dataset_id):
        """
        Get training jobs for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of TrainingJob objects
        """
        with get_session() as session:
            return session.query(TrainingJob).filter(TrainingJob.dataset_id == dataset_id).all()
    
    @staticmethod
    def get_training_job_by_id(job_id):
        """
        Get a training job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            TrainingJob object
        """
        with get_session() as session:
            return session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    @staticmethod
    def update_training_job(job_id, **kwargs):
        """
        Update a training job.
        
        Args:
            job_id: Job ID
            **kwargs: Fields to update
            
        Returns:
            Updated TrainingJob object
        """
        with get_session() as session:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                session.commit()
                session.refresh(job)
            return job
    
    @staticmethod
    def start_training_job(job_id):
        """
        Mark a training job as started.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated TrainingJob object
        """
        return TrainingOperations.update_training_job(
            job_id=job_id,
            status="running",
            started_at=datetime.now()
        )
    
    @staticmethod
    def complete_training_job(job_id, metrics=None, model_path=None):
        """
        Mark a training job as completed.
        
        Args:
            job_id: Job ID
            metrics: Training metrics
            model_path: Path to the trained model
            
        Returns:
            Updated TrainingJob object
        """
        return TrainingOperations.update_training_job(
            job_id=job_id,
            status="completed",
            completed_at=datetime.now(),
            metrics=metrics,
            model_path=model_path
        )
    
    @staticmethod
    def fail_training_job(job_id):
        """
        Mark a training job as failed.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated TrainingJob object
        """
        return TrainingOperations.update_training_job(
            job_id=job_id,
            status="failed",
            completed_at=datetime.now()
        )
    
    @staticmethod
    def add_training_log(job_id, message, level="INFO", metrics=None):
        """
        Add a training log.
        
        Args:
            job_id: Job ID
            message: Log message
            level: Log level (INFO, WARNING, ERROR, etc.)
            metrics: Training metrics at this log point
            
        Returns:
            Newly created TrainingLog object
        """
        with get_session() as session:
            log = TrainingLog(
                training_job_id=job_id,
                level=level,
                message=message,
                metrics=metrics
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log
    
    @staticmethod
    def get_training_logs(job_id):
        """
        Get training logs for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of TrainingLog objects
        """
        with get_session() as session:
            return session.query(TrainingLog).filter(TrainingLog.training_job_id == job_id).order_by(TrainingLog.timestamp).all()

class DatasetVersionOperations:
    """
    Operations for working with dataset versions in the database.
    """
    
    @staticmethod
    def create_version(dataset_id, version_id, file_path, description=None, parent_version_id=None, metadata=None):
        """
        Create a new dataset version.
        
        Args:
            dataset_id: Dataset ID
            version_id: Version ID (unique identifier)
            file_path: Path to the stored version file
            description: Version description
            parent_version_id: Parent version ID
            metadata: Version metadata
            
        Returns:
            Newly created DatasetVersion object
        """
        with get_session() as session:
            version = DatasetVersion(
                dataset_id=dataset_id,
                version_id=version_id,
                file_path=file_path,
                description=description,
                parent_version_id=parent_version_id,
                metadata=metadata
            )
            session.add(version)
            
            # Update the dataset's current version ID
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                dataset.current_version_id = version_id
            
            session.commit()
            session.refresh(version)
            return version
    
    @staticmethod
    def get_versions(dataset_id):
        """
        Get all versions for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of DatasetVersion objects
        """
        with get_session() as session:
            return session.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset_id
            ).order_by(DatasetVersion.created_at).all()
    
    @staticmethod
    def get_version(dataset_id, version_id):
        """
        Get a specific version of a dataset.
        
        Args:
            dataset_id: Dataset ID
            version_id: Version ID
            
        Returns:
            DatasetVersion object
        """
        with get_session() as session:
            return session.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_id == version_id
            ).first()
    
    @staticmethod
    def get_latest_version(dataset_id):
        """
        Get the latest version of a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            DatasetVersion object
        """
        with get_session() as session:
            return session.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset_id
            ).order_by(DatasetVersion.created_at.desc()).first()
    
    @staticmethod
    def create_version_from_dataframe(dataset_id, df, description=None, parent_version_id=None):
        """
        Create a new version from a pandas DataFrame.
        
        Args:
            dataset_id: Dataset ID
            df: Pandas DataFrame with dataset content
            description: Version description
            parent_version_id: Parent version ID
            
        Returns:
            Newly created DatasetVersion object
        """
        import hashlib
        
        # Create version ID based on content hash and timestamp
        content_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:8]
        timestamp = datetime.now()
        version_id = f"{timestamp.strftime('%Y%m%d%H%M%S')}_{content_hash}"
        
        # Create metadata
        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "content_hash": content_hash,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
        
        # Ensure version directory exists
        version_dir = Path("database/data/versions")
        version_dir.mkdir(exist_ok=True, parents=True)
        
        # Save dataset to parquet file
        file_path = str(version_dir / f"dataset_{dataset_id}_version_{version_id}.parquet")
        df.to_parquet(file_path)
        
        # Create version in database
        return DatasetVersionOperations.create_version(
            dataset_id=dataset_id,
            version_id=version_id,
            file_path=file_path,
            description=description,
            parent_version_id=parent_version_id,
            metadata=metadata
        )
    
    @staticmethod
    def load_version_data(version):
        """
        Load dataset data for a version.
        
        Args:
            version: DatasetVersion object
            
        Returns:
            Pandas DataFrame with dataset content
        """
        if not os.path.exists(version.file_path):
            raise FileNotFoundError(f"Dataset file for version {version.version_id} not found at {version.file_path}")
        
        return pd.read_parquet(version.file_path)
    
    @staticmethod
    def compare_versions(dataset_id, version_id1, version_id2):
        """
        Compare two versions of a dataset.
        
        Args:
            dataset_id: Dataset ID
            version_id1: ID of the first version
            version_id2: ID of the second version
            
        Returns:
            Dictionary with comparison results
        """
        # Get versions
        version1 = DatasetVersionOperations.get_version(dataset_id, version_id1)
        version2 = DatasetVersionOperations.get_version(dataset_id, version_id2)
        
        if not version1 or not version2:
            missing = []
            if not version1:
                missing.append(version_id1)
            if not version2:
                missing.append(version_id2)
            raise ValueError(f"Versions not found: {', '.join(missing)}")
        
        # Load data
        df1 = DatasetVersionOperations.load_version_data(version1)
        df2 = DatasetVersionOperations.load_version_data(version2)
        
        # Basic comparison
        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "version1_timestamp": version1.created_at,
            "version2_timestamp": version2.created_at,
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
    
    @staticmethod
    def restore_version(dataset_id, version_id, description=None):
        """
        Restore a previous version of a dataset as a new version.
        
        Args:
            dataset_id: Dataset ID
            version_id: ID of the version to restore
            description: Description for the new version
            
        Returns:
            Newly created DatasetVersion object with restored data
        """
        # Get version to restore
        version_to_restore = DatasetVersionOperations.get_version(dataset_id, version_id)
        if not version_to_restore:
            raise ValueError(f"Version {version_id} not found for dataset {dataset_id}")
        
        # Load data from the version
        df = DatasetVersionOperations.load_version_data(version_to_restore)
        
        # Create new version with restored data
        if description is None:
            description = f"Restored from version {version_id}"
        
        return DatasetVersionOperations.create_version_from_dataframe(
            dataset_id=dataset_id,
            df=df,
            description=description,
            parent_version_id=version_id
        )
    
    @staticmethod
    def delete_version(dataset_id, version_id):
        """
        Delete a specific version of a dataset.
        
        Args:
            dataset_id: Dataset ID
            version_id: ID of the version to delete
            
        Returns:
            True if deleted, False if not found
        """
        with get_session() as session:
            version = session.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_id == version_id
            ).first()
            
            if not version:
                return False
            
            # Check if this is the current version for the dataset
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.current_version_id == version_id:
                # Find the most recent version that isn't this one
                new_current = session.query(DatasetVersion).filter(
                    DatasetVersion.dataset_id == dataset_id,
                    DatasetVersion.version_id != version_id
                ).order_by(DatasetVersion.created_at.desc()).first()
                
                if new_current:
                    dataset.current_version_id = new_current.version_id
                else:
                    dataset.current_version_id = None
            
            # Delete the file if it exists
            if version.file_path and os.path.exists(version.file_path):
                os.remove(version.file_path)
            
            # Delete from database
            session.delete(version)
            session.commit()
            
            return True

class CodeQualityOperations:
    """
    Operations for working with code quality checks in the database.
    """
    
    @staticmethod
    def create_code_quality_check(filename, tool, score=None, issues_count=0, report=None):
        """
        Create a new code quality check.
        
        Args:
            filename: File name
            tool: Tool used for check (pylint, flake8, mypy, etc.)
            score: Quality score
            issues_count: Number of issues found
            report: Full report
            
        Returns:
            Newly created CodeQualityCheck object
        """
        with get_session() as session:
            check = CodeQualityCheck(
                filename=filename,
                tool=tool,
                score=score,
                issues_count=issues_count,
                report=report
            )
            session.add(check)
            session.commit()
            session.refresh(check)
            return check
    
    @staticmethod
    def get_all_code_quality_checks():
        """
        Get all code quality checks.
        
        Returns:
            List of CodeQualityCheck objects
        """
        with get_session() as session:
            return session.query(CodeQualityCheck).all()
    
    @staticmethod
    def get_code_quality_checks_by_filename(filename):
        """
        Get code quality checks for a file.
        
        Args:
            filename: File name
            
        Returns:
            List of CodeQualityCheck objects
        """
        with get_session() as session:
            return session.query(CodeQualityCheck).filter(CodeQualityCheck.filename == filename).all()
    
    @staticmethod
    def get_code_quality_checks_by_tool(tool):
        """
        Get code quality checks for a tool.
        
        Args:
            tool: Tool name
            
        Returns:
            List of CodeQualityCheck objects
        """
        with get_session() as session:
            return session.query(CodeQualityCheck).filter(CodeQualityCheck.tool == tool).all()