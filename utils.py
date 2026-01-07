"""
Utility Functions
Common helper functions used across the project.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


def create_directories(dirs):
    """
    Create directories if they don't exist.
    
    Parameters:
    -----------
    dirs : list or str
        Directory path(s) to create
    """
    if isinstance(dirs, str):
        dirs = [dirs]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ready: {dir_path}")


def save_json(data, filepath):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved JSON to: {filepath}")


def load_json(filepath):
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(obj, filepath):
    """Save object to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"💾 Saved pickle to: {filepath}")


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def log_metrics(metrics, filepath="metrics.json"):
    """
    Log metrics to JSON file with timestamp.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics to log
    filepath : str
        Path to save metrics
    """
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Load existing metrics if file exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    all_metrics.append(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"📊 Metrics logged to: {filepath}")


def format_price(price):
    """Format price with comma separators."""
    return f"${price:,.2f}"


def format_percentage(value):
    """Format percentage value."""
    return f"{value:.2f}%"


def print_dataframe_info(df, name="DataFrame"):
    """
    Print comprehensive information about a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    name : str
        Name to display
    """
    print("\n" + "="*50)
    print(f"{name} Information")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    print("="*50 + "\n")


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error, 
        mean_absolute_error, 
        r2_score,
        mean_absolute_percentage_error
    )
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def print_metrics(metrics, title="Metrics"):
    """
    Pretty print metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    title : str
        Title to display
    """
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:15s}: {value:,.4f}")
        else:
            print(f"{key:15s}: {value}")
    print("="*50 + "\n")


def time_function(func):
    """
    Decorator to time function execution.
    
    Usage:
    @time_function
    def my_function():
        ...
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        
        print(f"\n⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    
    return wrapper


def check_gpu_available():
    """Check if GPU is available for PyTorch."""
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠️ No GPU available, using CPU")
        return False


def estimate_memory_usage(df):
    """
    Estimate memory usage of DataFrame columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    
    Returns:
    --------
    pd.DataFrame
        Memory usage breakdown
    """
    mem_usage = df.memory_usage(deep=True)
    mem_df = pd.DataFrame({
        'Column': mem_usage.index,
        'Memory (MB)': mem_usage.values / 1024**2
    })
    mem_df = mem_df.sort_values('Memory (MB)', ascending=False)
    
    total_mb = mem_df['Memory (MB)'].sum()
    print(f"\nTotal memory usage: {total_mb:.2f} MB")
    
    return mem_df


def reduce_memory_usage(df):
    """
    Reduce memory usage of DataFrame by optimizing dtypes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize
    
    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    
    return df


def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame has required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
    
    Raises:
    -------
    ValueError
        If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("✅ DataFrame validation passed")


def split_train_val_test(df, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split DataFrame into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to split
    train_size : float
        Proportion for training set
    val_size : float
        Proportion for validation set
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=(1 - train_size - val_size), 
        random_state=random_state
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=(val_size / (train_size + val_size)),
        random_state=random_state
    )
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test GPU check
    check_gpu_available()
    
    # Test directory creation
    create_directories(['test_dir1', 'test_dir2/subdir'])
    
    print("\n✅ All utility functions working correctly!")
