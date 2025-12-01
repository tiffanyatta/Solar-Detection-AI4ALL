"""
Model training and persistence functions for the Solar Detection project.
"""

import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from data_processing import prepare_features, preprocess_data, load_data


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "solar_model.pkl")
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "solar_data.csv")


def train_model(df=None, data_path=None, save_model=True):
    """
    Train the XGBoost regression model.
    
    Args:
        df: Preprocessed DataFrame (optional, if None, loads from data_path)
        data_path: Path to CSV file (optional, if df is provided)
        save_model: Whether to save the trained model
    
    Returns:
        Trained model, X_test, y_test (for evaluation)
    """
    # Load data if DataFrame not provided
    if df is None:
        if data_path is None:
            data_path = DATA_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                f"Please place your CSV file in the '{DATA_DIR}' directory."
            )
        df = load_data(data_path)
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model trained successfully!")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Save model if requested
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    return model, X_test, y_test


def load_model(model_path=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model (optional, uses default if None)
    
    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first by running train_model()."
        )
    
    model = joblib.load(model_path)
    return model


def get_target_mean(data_path=None):
    """
    Compute the mean of the target variable (PolyPwr) from the training data.

    This is used to interpret predictions (e.g., to decide if a location is
    above or below average suitability).

    Args:
        data_path: Optional path to the CSV data file. Defaults to DATA_PATH.

    Returns:
        Float mean of the target variable.
    """
    if data_path is None:
        data_path = DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            f"Please place your CSV file in the '{DATA_DIR}' directory."
        )

    df = load_data(data_path)
    _, y = prepare_features(df)
    return float(y.mean())


def get_or_train_model(df=None, data_path=None, force_retrain=False):
    """
    Get the trained model, loading from disk if available, or training if not.
    
    Args:
        df: Preprocessed DataFrame (optional)
        data_path: Path to CSV file (optional)
        force_retrain: If True, retrain even if model exists
    
    Returns:
        Trained model
    """
    if not force_retrain and os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        return load_model()
    else:
        print("Training new model...")
        model, _, _ = train_model(df=df, data_path=data_path, save_model=True)
        return model


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained XGBRegressor model
        X: Feature matrix (DataFrame or array)
    
    Returns:
        Predictions (array)
    """
    # Ensure X is a DataFrame with correct column order
    if isinstance(X, np.ndarray):
        feature_names = get_feature_names()
        X = pd.DataFrame(X, columns=feature_names)
    elif isinstance(X, pd.DataFrame):
        # Ensure correct column order
        feature_names = get_feature_names()
        X = X[feature_names]
    
    predictions = model.predict(X)
    return predictions


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


