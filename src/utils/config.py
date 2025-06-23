"""
Configuration utility for the Dask ML Pipeline.

This module provides functions for loading and validating configuration settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the configuration file is not valid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check required sections
    required_sections = ["data", "dask", "preprocessing", "model", "evaluation", "visualization"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data configuration
    if "dataset" not in config["data"]:
        raise ValueError("Missing dataset specification in data configuration")
    
    # Validate dask configuration
    if "cluster" not in config["dask"]:
        raise ValueError("Missing cluster configuration in dask configuration")
    
    # Validate model configuration
    if "type" not in config["model"] or "algorithm" not in config["model"]:
        raise ValueError("Missing type or algorithm in model configuration")


def get_dataset_path(config: Dict[str, Any], split: str) -> Optional[str]:
    """
    Get the path for a specific dataset split.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        split (str): Dataset split (train, test, validation)
        
    Returns:
        Optional[str]: Path to the dataset split, or None if not found
    """    
    dataset_paths = config["data"]["paths"]
    
    if split not in dataset_paths:
        return None
    
    return dataset_paths[split]


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        output_path (str): Path to save the configuration file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
