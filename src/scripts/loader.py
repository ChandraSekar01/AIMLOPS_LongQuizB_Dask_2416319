"""
Data loader module for the Dask ML Pipeline.

This module provides classes for loading different types of datasets using Dask.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

logger = logging.getLogger("dask_ml_pipeline")


class DataLoader:
    """
    Data loader for different types of datasets using Dask.
    """
    
    def __init__(self, data_config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            data_config (Dict[str, Any]): Data configuration dictionary
        """
        self.data_config = data_config
        logger.info(f"Initializing data loader for dataset")
    
    def load_data(self) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        """
        Load a tabular dataset using Dask dataframes.
        
        Returns:
            Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]: Train, test, and validation datasets
        """
        from src.utils.config import get_dataset_path
        
        train_path = get_dataset_path(self.data_config, "train")
        test_path = get_dataset_path(self.data_config, "test")
        
        logger.info(f"Loading tabular dataset from {train_path}, {test_path}")
        
        # Check if paths exist
        for path in [train_path, test_path]:
            if path and not os.path.exists(path):
                logger.warning(f"Dataset path does not exist: {path}")
        
        # Load tabular data using Dask dataframes
        # For demonstration purposes, we'll create dummy data
        # In a real implementation, you would load actual tabular data
        
        # read csv file
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # convert df to dask df
        train_ddf = dd.from_pandas(train_df, npartitions=4)
        task_ddf = dd.from_pandas(test_df, npartitions=4)
        
        logger.info(f"Loaded dataset with {len(train_df)} training and {len(test_df)} testing")
        
        return train_df, test_df, train_ddf, task_ddf

