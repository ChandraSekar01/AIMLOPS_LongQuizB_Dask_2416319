"""
Model factory module for the Dask and Sklearn ML Pipeline.

This module provides a factory class for creating different types of machine learning models.
"""

import logging
from typing import Dict, Any

from dask_ml.linear_model import LinearRegression as DaskLR
from dask_ml.xgboost import XGBRegressor as DaskXGBR
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.ensemble import RandomForestRegressor as SklearnRFR


logger = logging.getLogger("dask_ml_pipeline")


class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError


class DaskModel(BaseModel):
    def __init__(self, estimator, hyperparameters: Dict[str, Any]):
        self.estimator = estimator(**hyperparameters)
        logger.info(f"Initialized Dask model: {self.estimator}")

    def fit(self, X, y):
        logger.info("Fitting Dask model...")
        self.estimator.fit(X, y)
        logger.info("Dask Model fit complete.")

    def predict(self, X):
        logger.info("Predicting with Dask model...")
        return self.estimator.predict(X)


class SklearnModel(BaseModel):
    def __init__(self, estimator, hyperparameters: Dict[str, Any]):
        self.estimator = estimator(**hyperparameters)
        logger.info(f"Initialized scikit-learn model: {self.estimator}")

    def fit(self, X, y):
        logger.info("Fitting scikit-learn model...")
        self.estimator.fit(X, y)
        logger.info("Sklearn Model fit complete.")

    def predict(self, X):
        logger.info("Predicting with scikit-learn model...")
        return self.estimator.predict(X)
    

class ModelFactory:
    """
    Factory class for creating machine learning models for Dask or scikit-learn.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the model factory.
        Args:
            model_config (Dict[str, Any]): Model configuration dictionary
        """
        self.config = model_config
        logger.info(f"Initializing model factory for {model_config['algorithm']} model")

    def create_model(self) -> BaseModel:
        """
        Create a model based on the configuration.
        Returns:
            BaseModel: The created model
        Raises:
            ValueError: If the model algorithm is not supported
        """
        algorithm = self.config["algorithm"]
        library = self.config.get("library", "dask")  # "dask" or "sklearn"
        hyperparameters = self.config.get("hyperparameters", {}).get(algorithm, {})

        logger.info(f"Creating {algorithm} model using {library}")

        # Dask-based models
        if library == "dask":
            if algorithm == "linear_regression":
                return DaskModel(DaskLR, hyperparameters)
            elif algorithm == "xgboost":
                return DaskModel(DaskXGBR, hyperparameters)
            else:
                raise ValueError(f"Unsupported Dask algorithm: {algorithm}")

        # scikit-learn-based models
        elif library == "sklearn":
            if algorithm == "linear_regression":
                return SklearnModel(SklearnLR, hyperparameters)
            elif algorithm == "random_forest":
                return SklearnModel(SklearnRFR, hyperparameters)
            else:
                raise ValueError(f"Unsupported scikit-learn algorithm: {algorithm}")

        else:
            raise ValueError(f"Unsupported library: {library}")
