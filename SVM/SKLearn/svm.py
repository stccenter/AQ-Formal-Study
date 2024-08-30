import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Function to train the model and return RMSE and R² score
def train_model(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', gamma='scale', epsilon=0.1):
    model = SVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Load the dataset
pa_files = os.listdir( 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join( 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Set your target variable for regression
X = data[["pm25_cf_1", "temperature", "humidity"]]
y = data['epa_pm25']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
rmses = []
r2_scores = []
total_models = 10
for i in range(total_models):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test)
    rmses.append(rmse)
    r2_scores.append(r2)
    print(f"Model {i+1}/{total_models} trained: RMSE = {rmse}, R² = {r2}")

# Calculate and print aggregate RMSE and R² score
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2_scores)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R²: {avg_r2}")

# Save the log of all values
with open('svr_80_20_model_log.txt', 'w') as file:
    for i in range(total_models):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R² = {r2_scores[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R²: {avg_r2}\n")
