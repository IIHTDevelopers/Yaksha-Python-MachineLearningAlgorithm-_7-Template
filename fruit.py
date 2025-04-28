import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load real-world dataset
def load_fruit_data():
    """
    TODO: Load the fruit dataset from 'fruit_data.csv'.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # TODO: Load the dataset from 'fruit_data.csv'
    # TODO: Print loading information
    pass

# 2. Preprocess data
def preprocess_fruit_data(df):
    """
    TODO: Preprocess the fruit dataset.
    Separate features and target.
    
    Args:
        df (pandas.DataFrame): The fruit dataset
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    # TODO: Separate features (X) and target (y)
    # TODO: Print preprocessing information
    pass

# 3. Split the data
def split_fruit_data(X, y, test_size=0.2):
    """
    TODO: Split the data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of the dataset to include in the test split
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Split the data using train_test_split
    # TODO: Print splitting information
    pass

# 4. Create model
def create_model(n_estimators=100, max_depth=None):
    """
    TODO: Create a Random Forest model.
    
    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        
    Returns:
        RandomForestClassifier: The created model
    """
    # TODO: Create a RandomForestClassifier with random_state=42
    # TODO: Print model creation information
    pass

# 5. Train model
def train_model(model, X_train, y_train):
    """
    TODO: Train the Random Forest model.
    
    Args:
        model (RandomForestClassifier): The model to train
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        RandomForestClassifier: The trained model
    """
    # TODO: Train the model
    # TODO: Print training information
    pass

# 6. Save model
def save_model(model, filename="fruit_rf_model.pkl"):
    """
    TODO: Save the trained model to a file.
    
    Args:
        model (RandomForestClassifier): The trained model
        filename (str): Path to save the model
    """
    # TODO: Save the model using joblib.dump
    # TODO: Print saving information
    pass

# 7. Load model
def load_model(filename="fruit_rf_model.pkl"):
    """
    TODO: Load a trained model from a file.
    
    Args:
        filename (str): Path to the saved model
        
    Returns:
        RandomForestClassifier: The loaded model
    """
    # TODO: Load the model using joblib.load
    # TODO: Print loading information
    pass

# 8. Predict new fruit from JSON
def check_new_data_from_json(model, json_file="fruit_item.json"):
    """
    TODO: Predict the fruit type of a new fruit from JSON data.
    
    Args:
        model (RandomForestClassifier): The trained model
        json_file (str): Path to the JSON file with new fruit data
    """
    # TODO: Load the JSON file
    # TODO: Extract fruit data
    # TODO: Create a dataframe with the new fruit data
    # TODO: Make prediction
    # TODO: Print the prediction result
    pass

# --- Pipeline Execution ---
# TODO: Uncomment and complete the following code to execute the pipeline
# df = load_fruit_data()
# X, y = preprocess_fruit_data(df)
# X_train, X_test, y_train, y_test = split_fruit_data(X, y)
# model = create_model()
# trained_model = train_model(model, X_train, y_train)
# save_model(trained_model)
# check_new_data_from_json(trained_model)
