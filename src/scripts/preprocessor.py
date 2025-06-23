"""
Data preprocessor module for the Dask ML Pipeline.

This module provides classes for preprocessing different types of data using Dask.
"""

import logging
from typing import Dict, Any, Tuple, Union, Optional

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from dask_ml.model_selection import train_test_split as dask_train_test_split



logger = logging.getLogger("dask_ml_pipeline")


class Preprocessor:
    """
    Preprocessor for NYC taxi trip data using Dask.
    """
    
    def __init__(self, preprocessing_config: Dict[str, Any]):
        """
        Initialize the preprocessor.
        
        Args:
            preprocessing_config (Dict[str, Any]): Preprocessing configuration dictionary
        """
        self.config = preprocessing_config
        logger.info("Initializing data preprocessor")
        
        # Initialize scalers and encoders
        self.numerical_scaler = None
        self.categorical_encoder = None
    
    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        """
        Vectorized Haversine distance calculation for Dask DataFrames.
        Returns distance in kilometers.
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km

    def preprocess(
            self, 
            # datetime_columns, 
            # features, 
            data, 
            is_train: bool = True
            ) -> dd.DataFrame:
        """
        Preprocess the taxi trip data.
        Args:
            data (dd.DataFrame): Raw taxi trip data
            is_train (bool): Whether this is training data (for fitting preprocessors)
        Returns:
            dd.DataFrame: Preprocessed data
        """
        logger.info(f"Preprocessing taxi data with {len(data.columns)} columns")

        # Detect DataFrame type
        is_dask = isinstance(data, dd.DataFrame)
        is_pandas = isinstance(data, pd.DataFrame)
        if not (is_dask or is_pandas):
            raise ValueError("Input data must be a pandas or Dask DataFrame.")

        # Ensure datetime columns are parsed
        for col in ['pickup_datetime', 'dropoff_datetime']:
            if not np.issubdtype(data[col].dtype, np.datetime64):
                data[col] = dd.to_datetime(data[col])

        # Feature engineering: hour of pickup
        data['hour'] = data['pickup_datetime'].dt.hour

        # Feature engineering: trip distance using Haversine formula
        data['trip_distance'] = self._haversine(
            data['pickup_longitude'], data['pickup_latitude'],
            data['dropoff_longitude'], data['dropoff_latitude']
        )

        # Data cleaning: filter outliers
        data = data[
            (data['trip_duration'] > 60) &
            (data['trip_duration'] < 7200) &
            (data['passenger_count'] > 0) &
            (data['passenger_count'] < 8)
        ]

        # Select relevant features for ML
        features = ['hour', 'passenger_count', 'trip_distance', 'trip_duration']
        result = data[features]


        logger.info(f"Preprocessed taxi data to {len(result.columns)} features")
        return result