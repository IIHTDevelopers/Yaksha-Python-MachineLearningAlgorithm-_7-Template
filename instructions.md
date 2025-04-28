# Machine Learning Assessment Instructions

This assessment tests your ability to implement machine learning models for classification tasks. You will work with two different datasets and implement the required functions to pass the test cases.

## Overview

You are provided with two skeleton Python files:
1. `fish.py` - For implementing a Decision Tree model to predict fish disease status
2. `fruit.py` - For implementing a Random Forest model to classify fruits

Your task is to complete the functions in these files by replacing the `pass` statements with the appropriate code. Each function has detailed TODO comments to guide you.

## Datasets

### Fish Disease Dataset (`fish_disease_data.csv`)

This dataset contains information about fish and their disease status:

- **Columns**:
  - `Age`: Age of the fish in years (numeric)
  - `Species`: Type of fish (categorical: Catfish, Trout, Tilapia, Goldfish, Salmon)
  - `Water_Temperature`: Temperature of the water in Celsius (numeric)
  - `Feeding_Behavior`: Feeding behavior of the fish (categorical: Normal, Reduced)
  - `Coloration`: Color appearance of the fish (categorical: Bright, Faded)
  - `Swimming_Behavior`: Swimming pattern of the fish (categorical: Normal, Erratic)
  - `Disease_Status`: Health status of the fish (categorical: Healthy, Diseased)

- **Task**: Predict whether a fish is healthy or diseased based on its features.
- **Model**: Decision Tree Classifier

### Fruit Classification Dataset (`fruit_data.csv`)

This dataset contains measurements of different fruits:

- **Columns**:
  - `mass`: Mass of the fruit in grams (numeric)
  - `width`: Width of the fruit in cm (numeric)
  - `height`: Height of the fruit in cm (numeric)
  - `color_score`: Color score of the fruit (numeric between 0 and 1)
  - `fruit_name`: Type of fruit (categorical: Apple, Orange, Pear, Grape, Banana)

- **Task**: Classify fruits based on their physical measurements.
- **Model**: Random Forest Classifier

## JSON Files

The assessment also includes JSON files for testing predictions:

1. `fish_data.json`: Contains data for a single fish to predict its disease status.
2. `fruit_item.json`: Contains data for a single fruit to predict its type.

## Requirements

1. Complete all functions in both `fish.py` and `fruit.py` files.
2. Follow the TODO comments in each function.
3. Make sure your code passes all the test cases in the `test` directory.
4. Do not modify the function signatures or return types.
5. Ensure your code handles the data correctly and makes accurate predictions.

## Testing Your Code

The test cases will check if:
1. Your code can load and process the datasets correctly.
2. The preprocessing steps are implemented properly.
3. The models are trained correctly.
4. The models can make predictions on new data from JSON files.
5. Additional specific requirements for each dataset (e.g., calculating entropy for the fish dataset).

## Tips

1. Read the TODO comments carefully to understand what each function should do.
2. Use pandas for data manipulation and scikit-learn for the machine learning models.
3. Pay attention to the expected return values of each function.
4. Make sure to handle categorical variables appropriately (e.g., using one-hot encoding).
5. For the fish dataset, ensure that the target variable is correctly encoded as 'Disease_Status_Healthy'.
6. For the fruit dataset, ensure that you're correctly separating the 'fruit_name' as the target variable.

Good luck!
