import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to train the model and return RMSE and R^2
def train_model(X_train, y_train, X_test, y_test, alpha):
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Load the dataset
pa_files = os.listdir('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Filter data based on a fixed PearsonR threshold (example: 0.7)
data = data[data['PearsonR'] >= 0.7]

X = data[["pm25_cf_1", "temperature", "humidity"]]  # Example set of variables
y = data['epa_pm25']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
rmses = []
r2s = []
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, alpha=1.0)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print aggregate RMSE and R^2
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the log of all values
with open('lasso_regressor_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
