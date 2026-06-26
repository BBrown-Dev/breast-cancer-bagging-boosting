import numpy as np
from src.utils import load_dataset, train_test_split_scaled


def preprocess_data(data_path: str = "data/data.csv"):
    """ Loads dataset and preprocesses it """
    X, y, df = load_dataset(data_path)

    # Remove zero-variance columns to avoid StandardScaler warnings
    non_constant_cols = X.columns[X.nunique() > 1]
    X = X[non_constant_cols]

    # Scale the features and split the dataset into training and testing sets
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = train_test_split_scaled(X, y)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df