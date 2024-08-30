import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def train_model(X_train, y_train, X_test, y_test, eta, max_depth, n_estimators):
    model = XGBRegressor(eta=eta, max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Load dataset
directory = 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData'
pa_files = os.listdir(directory)
data = pd.concat([pd.read_csv(os.path.join(directory, f)) for f in pa_files])

# Filter data
data = data[data['PearsonR'] >= 0.7]

# Define features and target
X = data[["pm25_cf_1", "temperature", "humidity"]]
y = data['epa_pm25']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
rmses, r2s = [], []

# Example: training 10 models with the same parameters
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, eta=0.1, max_depth=6, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print aggregate RMSE and R^2
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the log of all values
with open('ols_70_30_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
