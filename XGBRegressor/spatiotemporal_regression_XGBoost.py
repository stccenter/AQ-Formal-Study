import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Function to train the model and return RMSE and R^2
def train_model(X_train, y_train, X_test, y_test, eta, max_depth, n_estimators):
    model = XGBRegressor(eta=eta, max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Load the dataset
pa_files = os.listdir('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Assume the dataset has a timestamp and spatial coordinates
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['year'] = data['timestamp'].dt.year

# Example set of variables - including spatial and temporal features
X = data[["longitude", "latitude", "hour", "day", "month", "year", "temperature", "humidity"]]
y = data['target_variable']  # Specify your target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
rmses = []
r2s = []
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, eta=0.1, max_depth=6, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print aggregate RMSE and R^2
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the log of all values
with open('spatiotemporal_xgboost_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
