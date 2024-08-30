import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor

# Function to train the hybrid model and return RMSE and R^2
def train_hybrid_model(X_train, y_train, X_test, y_test, eta, max_depth, n_estimators):
    # Train Naive Bayes and predict probabilities
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_predictions = nb.predict_proba(X_train)[:, 1]  # Assuming y is binary

    # Add Naive Bayes predictions as a feature for XGBoost
    X_train_with_nb = np.hstack((X_train, nb_predictions.reshape(-1, 1)))

    # Train XGBoost with Naive Bayes predictions as a feature
    xgb = XGBRegressor(eta=eta, max_depth=max_depth, n_estimators=n_estimators)
    xgb.fit(X_train_with_nb, y_train)

    # Predict using XGBoost with Naive Bayes predictions as a feature for the test set
    nb_test_predictions = nb.predict_proba(X_test)[:, 1]
    X_test_with_nb = np.hstack((X_test, nb_test_predictions.reshape(-1, 1)))
    y_pred = xgb.predict(X_test_with_nb)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Load the dataset
pa_files = os.listdir('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Preprocess the data
# Filter data based on a fixed PearsonR threshold (example: 0.7)
data = data[data['PearsonR'] >= 0.7]

X = data[["pm25_cf_1", "temperature", "humidity"]]  # Example set of features
y = data['epa_pm25']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and calculate aggregate RMSE and R^2
rmses = []
r2s = []
for i in range(10):
    rmse, r2 = train_hybrid_model(X_train_scaled, y_train, X_test_scaled, y_test, eta=0.1, max_depth=6, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)

avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the log of all values
with open('hybrid_naivebayes_xgboost_70_30_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
