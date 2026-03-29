#!/usr/bin/env python3
"""
Simplified data loading utilities for activation analysis.
"""

import pickle
import os
from pathlib import Path
from typing import Tuple, Optional, List, Any
import numpy as np
from tqdm import tqdm


def load_activations_from_pickle(data_path: str, max_samples: Optional[int] = None) -> List[Any]:
    """
    Load activations from pickle file or directory containing pickle files.
    
    Args:
        data_path: Path to pickle file or directory
        max_samples: Maximum number of samples to load
        
    Returns:
        List of activation data
    """
    if os.path.isfile(data_path):
        # Direct pickle file
        with open(data_path, 'rb') as f:
            activations = pickle.load(f)
        print(f"Loaded {len(activations)} samples from {data_path}")
    else:
        # Directory - look for pickle files
        data_dir = Path(data_path)
        pickle_files = list(data_dir.glob("*.pkl"))
        
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in {data_path}")
        
        # Use the first pickle file found
        pickle_file = pickle_files[0]
        print(f"Loading from {pickle_file}")
        
        with open(pickle_file, 'rb') as f:
            activations = pickle.load(f)
        print(f"Loaded {len(activations)} samples from {pickle_file}")
    
    # Limit samples if requested
    if max_samples and len(activations) > max_samples:
        activations = activations[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    return activations


def load_input_ids_from_pickle(data_path: str, max_samples: Optional[int] = None) -> Optional[List[Any]]:
    """
    Load input IDs from pickle file in the same directory as activations.
    
    Args:
        data_path: Path to activation data
        max_samples: Maximum number of samples to load
        
    Returns:
        List of input IDs or None if not found
    """
    if os.path.isfile(data_path):
        data_dir = Path(data_path).parent
    else:
        data_dir = Path(data_path)
    
    # Look for input IDs pickle file
    input_ids_files = list(data_dir.glob("*input_ids*.pkl"))
    
    if not input_ids_files:
        print("No input IDs pickle file found")
        return None
    
    input_ids_file = input_ids_files[0]
    print(f"Loading input IDs from {input_ids_file}")
    
    try:
        with open(input_ids_file, 'rb') as f:
            input_ids = pickle.load(f)
        print(f"Loaded input IDs for {len(input_ids)} samples")
        
        # Limit samples if requested
        if max_samples and len(input_ids) > max_samples:
            input_ids = input_ids[:max_samples]
            print(f"Limited input IDs to {max_samples} samples")
        
        return input_ids
    except Exception as e:
        print(f"Error loading input IDs: {e}")
        return None


def load_activation_data(data_path: str, max_samples: Optional[int] = None, 
                        load_input_ids: bool = True) -> Tuple[List[Any], Optional[List[Any]]]:
    """
    Load both activations and input IDs from pickle files.
    
    Args:
        data_path: Path to activation data
        max_samples: Maximum number of samples to load
        load_input_ids: Whether to load input IDs
        
    Returns:
        Tuple of (activations, input_ids)
    """
    print(f"Loading data from: {data_path}")
    
    # Load activations
    activations = load_activations_from_pickle(data_path, max_samples)
    
    # Load input IDs if requested
    input_ids = None
    if load_input_ids:
        input_ids = load_input_ids_from_pickle(data_path, max_samples)
    
    return activations, input_ids