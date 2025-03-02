"""
Dataset version control UI component for the ML Dataset & Code Generation Manager.
Provides UI for viewing, comparing, and restoring dataset versions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import hashlib
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from database import version_control

def render_version_control_ui(dataset_id: int, df: Optional[pd.DataFrame] = None):
    """
    Render the version control UI for a dataset
    
    Args:
        dataset_id: ID of the dataset
        df: Current DataFrame of the dataset (optional)
    """
    st.header("Dataset Version Control")
    
    # Get all versions of the dataset
    versions = version_control.get_versions(dataset_id)
    
    if not versions:
        st.info("No versions found for this dataset. Save changes to create the first version.")
        
        if df is not None and st.button("Create Initial Version"):
            version = version_control.create_version(
                dataset_id=dataset_id,
                df=df,
                description="Initial version"
            )
            st.success(f"Created initial version: {version.version_id}")
            st.experimental_rerun()
        
        return
    
    # Display version history
    st.subheader("Version History")
    
    version_data = []
    for v in versions:
        version_data.append({
            "Version ID": v.version_id,
            "Date": v.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Rows": v.metadata.get("rows", "N/A"),
            "Columns": v.metadata.get("columns", "N/A"),
            "Description": v.description
        })
    
    version_df = pd.DataFrame(version_data)
    st.dataframe(version_df, use_container_width=True)
    
    # Version actions section
    st.subheader("Version Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_version = st.selectbox(
            "Select Version", 
            options=[v.version_id for v in versions],
            format_func=lambda x: f"{x} - {next((v.timestamp.strftime('%Y-%m-%d %H:%M:%S') for v in versions if v.version_id == x), '')}"
        )
        
        # Get selected version object
        selected_v = next((v for v in versions if v.version_id == selected_version), None)
        
        if selected_v:
            st.write(f"**Description:** {selected_v.description}")
            st.write(f"**Created:** {selected_v.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display metadata
            if selected_v.metadata:
                with st.expander("Version Metadata"):
                    for key, value in selected_v.metadata.items():
                        if key != "column_names":  # Show column names separately
                            st.write(f"**{key}:** {value}")
                    
                    if "column_names" in selected_v.metadata:
                        st.write("**Columns:**")
                        st.write(", ".join(selected_v.metadata["column_names"]))
    
    with col2:
        st.write("**Actions:**")
        
        if selected_v:
            # Load selected version
            if st.button("View Version Data"):
                version_df = version_control.load_version_data(selected_v)
                st.session_state["viewing_version_df"] = version_df
                st.session_state["viewing_version_id"] = selected_v.version_id
            
            # Restore version
            if st.button("Restore This Version"):
                if df is not None:
                    description = st.session_state.get("restore_description", f"Restored from {selected_v.version_id}")
                    new_version = version_control.restore_version(
                        dataset_id=dataset_id,
                        version_id=selected_v.version_id,
                        description=description
                    )
                    st.success(f"Restored version {selected_v.version_id} as new version {new_version.version_id}")
                    st.experimental_rerun()
                else:
                    st.error("Cannot restore version: No dataset provided")
        
        # Compare versions
        if len(versions) > 1:
            st.write("**Compare Versions:**")
            compare_v1 = st.selectbox("Version 1", options=[v.version_id for v in versions], key="compare_v1")
            compare_v2 = st.selectbox("Version 2", options=[v.version_id for v in versions], key="compare_v2")
            
            if st.button("Compare Versions"):
                if compare_v1 != compare_v2:
                    comparison = version_control.compare_versions(
                        dataset_id=dataset_id,
                        version_id1=compare_v1,
                        version_id2=compare_v2
                    )
                    st.session_state["version_comparison"] = comparison
                else:
                    st.warning("Please select different versions to compare")
    
    # Show version data if requested
    if "viewing_version_df" in st.session_state:
        st.subheader(f"Data for Version: {st.session_state['viewing_version_id']}")
        st.dataframe(st.session_state["viewing_version_df"], use_container_width=True)
        
        if st.button("Clear Version View"):
            del st.session_state["viewing_version_df"]
            del st.session_state["viewing_version_id"]
            st.experimental_rerun()
    
    # Show version comparison if requested
    if "version_comparison" in st.session_state:
        comparison = st.session_state["version_comparison"]
        st.subheader(f"Version Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Version 1:** {comparison['version1']}")
            st.write(f"**Date:** {comparison['version1_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.write(f"**Version 2:** {comparison['version2']}")
            st.write(f"**Date:** {comparison['version2_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.write(f"**Rows Changed:** {comparison['rows_diff']} ({'+' if comparison['rows_diff'] > 0 else ''}{comparison['rows_diff']})")
        
        if comparison["columns_added"]:
            st.write("**Columns Added:**")
            for col in comparison["columns_added"]:
                st.write(f"- {col}")
        
        if comparison["columns_removed"]:
            st.write("**Columns Removed:**")
            for col in comparison["columns_removed"]:
                st.write(f"- {col}")
        
        if comparison["columns_diff"]:
            st.write("**Columns Changed:**")
            for col, diff in comparison["columns_diff"].items():
                if diff.get("type_changed", False):
                    st.write(f"- {col}: Type changed from {diff['type1']} to {diff['type2']}")
                elif diff.get("values_changed", False):
                    st.write(f"- {col}: Values changed")
        
        if st.button("Clear Comparison"):
            del st.session_state["version_comparison"]
            st.experimental_rerun()

def render_save_version_ui(dataset_id: int, df: pd.DataFrame):
    """
    Render UI for saving a new version of a dataset
    
    Args:
        dataset_id: ID of the dataset
        df: DataFrame to save
    """
    st.subheader("Save Current Version")
    
    # Get latest version if any
    latest_version = version_control.get_latest_version(dataset_id)
    
    # Calculate changes if a previous version exists
    if latest_version:
        try:
            prev_df = version_control.load_version_data(latest_version)
            rows_diff = len(df) - len(prev_df)
            cols_diff = len(df.columns) - len(prev_df.columns)
            
            st.write(f"Changes from last version:")
            st.write(f"- Rows: {'+' if rows_diff > 0 else ''}{rows_diff}")
            st.write(f"- Columns: {'+' if cols_diff > 0 else ''}{cols_diff}")
            
            # Check content hash
            current_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:8]
            if current_hash == latest_version.metadata.get("content_hash"):
                st.info("No changes detected in the data content since the last version.")
        except:
            st.warning("Could not compare with previous version.")
    
    # Input for version description
    description = st.text_area("Version Description", placeholder="Describe the changes in this version", key="version_description")
    
    # Save button
    if st.button("Save Version"):
        version = version_control.create_version(
            dataset_id=dataset_id,
            df=df,
            description=description
        )
        st.success(f"Created new version: {version.version_id}")
        
        return version
    
    return None

def render_version_visualization(dataset_id: int):
    """
    Render visualization of dataset versions
    
    Args:
        dataset_id: ID of the dataset
    """
    versions = version_control.get_versions(dataset_id)
    
    if not versions:
        st.info("No versions available to visualize.")
        return
    
    st.subheader("Version Metrics Visualization")
    
    # Prepare data for visualization
    viz_data = []
    for version in versions:
        viz_data.append({
            "Version": version.version_id[:8] + "...",  # Truncated ID for display
            "Date": version.timestamp,
            "Rows": version.metadata.get("rows", 0),
            "Columns": version.metadata.get("columns", 0),
            "Full Version ID": version.version_id,  # For tooltip
            "Description": version.description
        })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Visualize row counts over versions
    fig1 = px.line(
        viz_df, 
        x="Date", 
        y="Rows", 
        title="Dataset Size (Rows) Across Versions",
        markers=True,
        hover_data=["Full Version ID", "Description"]
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualize column counts over versions
    fig2 = px.line(
        viz_df,
        x="Date",
        y="Columns",
        title="Dataset Structure (Columns) Across Versions",
        markers=True,
        hover_data=["Full Version ID", "Description"]
    )
    st.plotly_chart(fig2, use_container_width=True)