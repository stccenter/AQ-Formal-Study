import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# Function to train the XGBoost model and return RMSE and R^2
def train_model(X_train, y_train, X_test, y_test, eta, max_depth, n_estimators):
    # Initialize the XGBRegressor model with specified hyperparameters
    model = XGBRegressor(eta=eta, max_depth=max_depth, n_estimators=n_estimators)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE and R^2 score for evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2


# Load and concatenate the dataset from multiple files
pa_files = os.listdir(
    'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join(
    'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f))
                  for f in pa_files])

# Filter data based on Pearson correlation coefficient threshold
data = data[data['PearsonR'] >= 0.7]

# Select features and target variable
X = data[["pm25_cf_1", "temperature", "humidity"]]  # Example feature set
y = data['epa_pm25']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model multiple times and collect metrics
rmses = []
r2s = []
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, eta=0.1, max_depth=6, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print the average RMSE and R^2 scores
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the evaluation metrics for each model run to a log file
with open('xgboost_model_evaluation_80_20_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i + 1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
