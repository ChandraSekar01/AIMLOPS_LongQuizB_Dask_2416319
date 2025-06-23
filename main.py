#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the Dask ML Pipeline.

This script orchestrates the entire machine learning pipeline using Dask
for distributed computing, from data loading to model evaluation and visualization.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.data.loader import DataLoader
from src.scripts.preprocessor import Preprocessor
from src.scripts.model_factory import ModelFactory
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer



def main():
    """Main execution function."""
    config_path = "config/config.yaml"
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_level = config["logging"]["level"]
    logger = setup_logger(log_level, config["logging"]["save_path"])
    
    # Create output directory
    results_dir = Path(config['paths']['results'])
    results_dir.mkdirs(parents=True, exist_ok=True)
    
    logger.info("Starting Dask ML Pipeline")
    logger.info(f"Using dataset: {config['paths']['data']}")
    
    start_time = time.time()
    
    try:
        # Initialize Dask cluster
        from src.utils.dask_setup import setup_dask_cluster
        client = setup_dask_cluster(config["dask"])
        logger.info(f"Dask cluster initialized with {config['dask']['cluster']['n_workers']} workers")
        
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(config["data"])
        train_df, test_df, train_ddf, task_ddf = data_loader.load_data()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        # pandas df
        preprocessor = Preprocessor(config["preprocessing"])
        train_data = preprocessor.preprocess(train_data, is_train=True)
        test_data = preprocessor.preprocess(test_data, is_train=False)
        # dask df
        train_data = preprocessor.preprocess(train_data, is_train=True)
        test_data = preprocessor.preprocess(test_data, is_train=False)

        
        # Train model
        logger.info(f"Training {config['model']['algorithm']} model...")
        model_factory = ModelFactory(config["model"])
        model = model_factory.create_model()
        model.train(train_data, validation_data)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = Evaluator(config["evaluation"])
        evaluation_results = evaluator.evaluate(model, test_data)
        
        # Compare with traditional approach
        if config["evaluation"]["comparison"]["enabled"]:
            logger.info("Comparing with traditional approach...")
            comparison_results = evaluator.compare_with_traditional(
                train_data, test_data, config["model"]
            )
            
        # Visualize results
        logger.info("Generating visualizations...")
        visualizer = Visualizer(config["visualization"])
        visualizer.visualize(model, evaluation_results, results_dir)
        
        if config["visualization"]["dashboard"]["enabled"]:
            logger.info(f"Starting dashboard on port {config['visualization']['dashboard']['port']}...")
            visualizer.launch_dashboard(model, evaluation_results)
        
        # Save results
        logger.info(f"Saving results to {results_dir}...")
        evaluator.save_results(evaluation_results, results_dir / "evaluation_results.json")
        model.save(config['paths']['model'])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {e}")
        raise
    finally:
        # Close Dask client
        if 'client' in locals():
            client.close()
            logger.info("Dask client closed")


if __name__ == "__main__":
    main()
