"""
Dask setup utility for the Dask ML Pipeline.

This module provides functions for setting up and configuring Dask clusters.
"""

import logging
import os
from typing import Dict, Any, Optional

import dask
from dask.distributed import Client, LocalCluster
from distributed.deploy.spec import SpecCluster

logger = logging.getLogger("dask_ml_pipeline")


def setup_dask_cluster(dask_config: Dict[str, Any]) -> Client:
    """
    Set up a Dask cluster based on the configuration.
    
    Args:
        dask_config (Dict[str, Any]): Dask configuration dictionary
        
    Returns:
        Client: Dask distributed client
        
    Raises:
        ValueError: If the cluster type is not supported
    """
    cluster_type = dask_config["cluster"]["type"]
    
    if cluster_type == "local":
        return setup_local_cluster(dask_config["cluster"])
    elif cluster_type == "distributed":
        return setup_distributed_cluster(dask_config["distributed"])
    elif cluster_type == "kubernetes":
        return setup_kubernetes_cluster(dask_config["kubernetes"])
    else:
        raise ValueError(f"Unsupported cluster type: {cluster_type}")


def setup_local_cluster(cluster_config: Dict[str, Any]) -> Client:
    """
    Set up a local Dask cluster.
    
    Args:
        cluster_config (Dict[str, Any]): Local cluster configuration
        
    Returns:
        Client: Dask distributed client
    """
    n_workers = cluster_config.get("n_workers", 4)
    threads_per_worker = cluster_config.get("threads_per_worker", 2)
    memory_limit = cluster_config.get("memory_limit", "4GB")
    
    logger.info(f"Setting up local Dask cluster with {n_workers} workers")
    
    # Set up Dask configuration
    dask.config.set({
        "distributed.worker.memory.target": 0.8,  # Target 80% memory utilization
        "distributed.worker.memory.spill": 0.9,   # Spill to disk at 90% memory
        "distributed.worker.memory.pause": 0.95,  # Pause worker at 95% memory
        "distributed.worker.memory.terminate": 0.98,  # Terminate worker at 98% memory
    })
    
    # Create local cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        silence_logs=logging.INFO,
    )
    
    # Create client
    client = Client(cluster)
    
    logger.info(f"Local Dask cluster dashboard available at: {client.dashboard_link}")
    
    return client


def get_client_info(client: Client) -> Dict[str, Any]:
    """
    Get information about the Dask client and cluster.
    
    Args:
        client (Client): Dask distributed client
        
    Returns:
        Dict[str, Any]: Client and cluster information
    """
    info = {
        "dashboard_link": client.dashboard_link,
        "scheduler_info": client.scheduler_info(),
        "n_workers": len(client.scheduler_info()["workers"]),
        "total_threads": sum(w["nthreads"] for w in client.scheduler_info()["workers"].values()),
        "total_memory": sum(w["memory_limit"] for w in client.scheduler_info()["workers"].values()),
    }
    
    return info
