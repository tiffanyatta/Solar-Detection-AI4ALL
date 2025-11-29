"""
Data preprocessing functions for the Solar Detection project.
Handles data loading, cleaning, and feature transformation.
"""

import pandas as pd
import numpy as np
import os


def convert_time(time):
    """
    Convert time from HHMM format to decimal hours.
    
    Args:
        time: Time in HHMM format (e.g., 1149 for 11:49)
    
    Returns:
        Time in decimal hours (e.g., 11.8167 for 11:49)
    """
    hours = int(time / 100)
    minutes = time % 100
    total_minutes = hours * 60 + minutes
    total_hours = total_minutes / 60
    return total_hours


def convert_date(date):
    """
    Convert date from YYYYMMDD format to total days from year 0.
    
    Args:
        date: Date in YYYYMMDD format (e.g., 20171021)
    
    Returns:
        Total days from year 0
    """
    years = int(date / 10000)
    days = date % 100
    months = int((date % 10000) / 100)
    total_days = (years * 365.25) + (months * 30.5) + days
    return total_days


def day_from_zero(date, min_date):
    """
    Convert date to days from the minimum date in the dataset.
    
    Args:
        date: Date in YYYYMMDD format
        min_date: Minimum date in the dataset (YYYYMMDD format)
    
    Returns:
        Days from the minimum date
    """
    return convert_date(date) - convert_date(min_date)


def preprocess_data(df):
    """
    Preprocess the raw dataset for model training.
    
    Args:
        df: Raw DataFrame with all columns
    
    Returns:
        Preprocessed DataFrame ready for model training
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop unnecessary columns
    df_processed = df_processed.drop(columns=['YRMODAHRMI', 'Season', 'Location', 'Hour'], errors='ignore')
    
    # Convert Time from HHMM to decimal hours
    if 'Time' in df_processed.columns:
        df_processed['Time'] = df_processed['Time'].apply(convert_time)
    
    # Convert Date to days from first date
    if 'Date' in df_processed.columns:
        min_date = df_processed['Date'].min()
        df_processed['Date'] = df_processed['Date'].apply(lambda x: day_from_zero(x, min_date))
    
    return df_processed


def load_data(file_path=None, use_google_sheets_fallback=False):
    """
    Load data from a CSV file and preprocess it.
    
    Args:
        file_path: Path to the CSV file (optional, defaults to data/solar_data.csv)
        use_google_sheets_fallback: If True and file not found, try loading from Google Sheets URL
    
    Returns:
        Preprocessed DataFrame
    """
    if file_path is None:
        file_path = os.path.join("data", "solar_data.csv")
    
    # Try to load from local file
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    elif use_google_sheets_fallback:
        # Fallback to Google Sheets URL (original data source)
        import warnings
        warnings.warn(
            f"Local file not found at {file_path}. "
            "Loading from Google Sheets URL as fallback. "
            "For better performance, download the data locally."
        )
        url = "1QZ16T9cNLx-2pKAqvwO_BYOIx5mi3Cvyb4K4LHfo46M"
        df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{url}/export?format=csv")
    else:
        raise FileNotFoundError(
            f"Data file not found at {file_path}. "
            "Please download the dataset and place it in the data/ directory."
        )
    
    df_processed = preprocess_data(df)
    return df_processed


def prepare_features(df):
    """
    Prepare feature matrix X and target vector y from preprocessed DataFrame.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
    """
    # Drop target and any classification columns
    X = df.drop(columns=['PolyPwr', 'PolyPwr_class', 'PolyPwr_class_num'], errors='ignore')
    
    # Ensure only numerical types are included
    X = X.select_dtypes(include='number')
    
    # Extract target
    y = df['PolyPwr']
    
    return X, y


def get_feature_names():
    """
    Get the expected feature names in the correct order.
    
    Returns:
        List of feature names
    """
    return [
        'Date', 'Time', 'Latitude', 'Longitude', 'Altitude', 'Month',
        'Humidity', 'AmbientTemp', 'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling'
    ]

