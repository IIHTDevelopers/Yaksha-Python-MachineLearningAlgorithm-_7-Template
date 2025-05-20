import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path="fruit_data.csv"):
    """
    Load and preprocess the fruit dataset.
    
    TODO:
    1. Load the fruit data from the CSV file specified by data_path
    2. Separate the features (X) and target variable (y)
       - X should contain: 'mass', 'width', 'height', 'color_score'
       - y should contain: 'fruit_name'
    3. Return X and y
    
    Parameters:
    -----------
    data_path : str, default="fruit_data.csv"
        Path to the CSV file containing fruit data
        
    Returns:
    --------
    X : pandas.DataFrame
        Features for fruit classification
    y : pandas.Series
        Target variable (fruit_name)
    """
    # Your code here
    return pd.DataFrame(), pd.Series()
    pass  

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    TODO:
    1. Use train_test_split to split the data into training and testing sets
    2. Use the provided test_size and random_state parameters
    3. Return the split data
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features for fruit classification
    y : pandas.Series
        Target variable (fruit_name)
    test_size : float, default=0.2
        Proportion of the data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    """
    # Your code here
    return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    pass  


def train_and_save_model(X_train, y_train, model_path="fruit_rf_model.pkl"):
    """
    Train a Random Forest classifier and save it to disk.
    
    TODO:
    1. Create a RandomForestClassifier with appropriate parameters
    2. Train the model on the provided training data
    3. Save the trained model to disk using pickle
    4. Return the trained model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    model_path : str, default="fruit_rf_model.pkl"
        Path where the trained model will be saved
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained Random Forest classifier
    """
    # Your code here
    return RandomForestClassifier()
    pass  

def load_model(model_path="fruit_rf_model.pkl"):
    """
    Load a trained model from disk.
    
    TODO:
    1. Load the model from the specified path using pickle
    2. Return the loaded model
    
    Parameters:
    -----------
    model_path : str, default="fruit_rf_model.pkl"
        Path to the saved model file
        
    Returns:
    --------
    model : RandomForestClassifier
        Loaded Random Forest classifier
    """
    # Your code here
    return RandomForestClassifier()
    pass  

def predict_from_json(model, json_file="fruit_item.json"):
    """
    Make a prediction for a fruit based on features in a JSON file.
    
    TODO:
    1. Load the fruit data from the JSON file
    2. Extract the features (mass, width, height, color_score)
    3. Use the model to predict the fruit type
    4. Return the predicted fruit type as a string
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model to use for prediction
    json_file : str, default="fruit_item.json"
        Path to the JSON file containing fruit features
        
    Returns:
    --------
    prediction : str
        Predicted fruit type (e.g., "apple", "orange", "banana", etc.)
    """
    # Your code here
    return ""
    pass  # Remove this line when implementing the function


if __name__ == "__main__":
    # Example usage of the functions
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and save model
    model = train_and_save_model(X_train, y_train)
    
    # Load model
    loaded_model = load_model()
    
    # Make prediction from JSON
    prediction = predict_from_json(loaded_model)
    print(f"Predicted fruit: {prediction}")
