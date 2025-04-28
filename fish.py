import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Load synthetic fish disease dataset
def load_fish_disease_data():
    """
    TODO: Load the fish disease dataset from 'fish_disease_data.csv'.
    The dataset should be limited to the first 1000 records.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # TODO: Load the dataset from 'fish_disease_data.csv'
    # TODO: Limit to first 1000 records
    # TODO: Print loading information
    pass

# 2. EDA Function to count fish with age > 1 year
def perform_eda_on_age(df):
    """
    TODO: Perform exploratory data analysis on the Age column.
    Count the number of fish with age > 1 year.
    
    Args:
        df (pandas.DataFrame): The fish disease dataset
    """
    # TODO: Check if 'Age' column exists
    # TODO: Count fish with age > 1 year
    # TODO: Print the count
    pass

# 3. Preprocess data
def preprocess_fish_data(df):
    """
    TODO: Preprocess the fish disease dataset.
    Convert categorical variables to dummy variables.
    Separate features and target.
    
    Args:
        df (pandas.DataFrame): The fish disease dataset
        
    Returns:
        tuple: (X, y, df_encoded) where X is features, y is target, and df_encoded is the encoded dataframe
    """
    # TODO: Convert categorical variables to dummy variables
    # TODO: Check if 'Disease_Status_Healthy' column exists after encoding
    # TODO: Separate features (X) and target (y)
    # TODO: Print preprocessing information
    pass

# 4. Split the data
def split_fish_data(X, y, test_size=0.2):
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

# 5. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    """
    TODO: Create and train a Decision Tree model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        DecisionTreeClassifier: The trained model
    """
    # TODO: Create a DecisionTreeClassifier with random_state=42
    # TODO: Train the model
    # TODO: Print training information
    pass

# 6. Calculate entropy
def calculate_entropy(y):
    """
    TODO: Calculate the entropy of the target variable.
    
    Args:
        y (pandas.Series): Target variable
    """
    # TODO: Calculate value counts with normalize=True
    # TODO: Calculate entropy using the formula: -sum(p * log2(p))
    # TODO: Print the entropy value
    pass

# 7. Predict new fish data from JSON
def check_new_data_from_json(model, df_encoded, json_file="fish_data.json"):
    """
    TODO: Predict the disease status of a new fish from JSON data.
    
    Args:
        model (DecisionTreeClassifier): The trained model
        df_encoded (pandas.DataFrame): The encoded dataframe
        json_file (str): Path to the JSON file with new fish data
    """
    # TODO: Load the JSON file
    # TODO: Extract fish data
    # TODO: Create a temporary dataframe with the new fish data
    # TODO: Combine with original data for consistent encoding
    # TODO: Extract features for the new fish
    # TODO: Make prediction
    # TODO: Print the prediction result
    pass

# --- Pipeline Execution ---
# TODO: Uncomment and complete the following code to execute the pipeline
# df = load_fish_disease_data()
# perform_eda_on_age(df)
# X, y, df_encoded = preprocess_fish_data(df)
# X_train, X_test, y_train, y_test = split_fish_data(X, y)
# model = create_and_train_model(X_train, y_train)
# joblib.dump(model, 'decision_tree_fish_disease_model.pkl')
# print("💾 Model saved as 'decision_tree_fish_disease_model.pkl'")
# calculate_entropy(y)
# check_new_data_from_json(model, df_encoded)
