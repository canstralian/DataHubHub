"""
Database models for the ML Dataset & Code Generation Manager.
"""
from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Float, 
    DateTime, 
    Text, 
    Boolean,
    ForeignKey,
    LargeBinary,
    JSON,
    create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
import json

# Create engine and session
DATABASE_URL = "sqlite:///database/data/mlmanager.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Dataset(Base):
    """
    Model for storing dataset information.
    """
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(String(50), nullable=False)  # csv, json, etc.
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    source = Column(String(255), nullable=True)  # local, huggingface, etc.
    source_url = Column(String(255), nullable=True)
    additional_data = Column(JSON, nullable=True)  # Any additional metadata
    
    # Relationships
    columns_info = relationship("DatasetColumn", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format": self.format,
            "rows": self.rows,
            "columns": self.columns,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "source_url": self.source_url,
            "additional_data": self.additional_data
        }

class DatasetColumn(Base):
    """
    Model for storing information about dataset columns.
    """
    __tablename__ = "dataset_columns"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    name = Column(String(255), nullable=False)
    data_type = Column(String(50), nullable=False)  # int, float, string, etc.
    missing_values = Column(Integer, default=0)
    missing_percentage = Column(Float, default=0.0)
    unique_values = Column(Integer, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    mean_value = Column(Float, nullable=True)
    median_value = Column(Float, nullable=True)
    std_value = Column(Float, nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="columns_info")
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "data_type": self.data_type,
            "missing_values": self.missing_values,
            "missing_percentage": self.missing_percentage,
            "unique_values": self.unique_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "std_value": self.std_value
        }

class TrainingJob(Base):
    """
    Model for storing information about model training jobs.
    """
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_type = Column(String(100), nullable=False)  # CodeT5, CodeBERT, etc.
    task_type = Column(String(100), nullable=False)  # Code to Comment, Comment to Code, etc.
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="created")  # created, running, completed, failed
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    model_path = Column(String(255), nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    logs = relationship("TrainingLog", back_populates="training_job", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dataset_id": self.dataset_id,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "model_path": self.model_path
        }

class TrainingLog(Base):
    """
    Model for storing training logs.
    """
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    level = Column(String(20), default="INFO")  # INFO, WARNING, ERROR, etc.
    message = Column(Text, nullable=False)
    metrics = Column(JSON, nullable=True)  # Any metrics at this log point
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="logs")
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "message": self.message,
            "metrics": self.metrics
        }

class CodeQualityCheck(Base):
    """
    Model for storing code quality check results.
    """
    __tablename__ = "code_quality_checks"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    tool = Column(String(50), nullable=False)  # pylint, flake8, mypy, etc.
    created_at = Column(DateTime, default=datetime.now)
    score = Column(Float, nullable=True)
    issues_count = Column(Integer, default=0)
    report = Column(Text, nullable=True)
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "tool": self.tool,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "score": self.score,
            "issues_count": self.issues_count,
            "report": self.report
        }

# Create all tables
def init_db():
    """Initialize the database and create all tables."""
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DATABASE_URL.replace('sqlite:///', '')), exist_ok=True)
    Base.metadata.create_all(engine)

# Helper function to get a database session
def get_session():
    """Get a database session."""
    return Session()

if __name__ == "__main__":
    init_db()