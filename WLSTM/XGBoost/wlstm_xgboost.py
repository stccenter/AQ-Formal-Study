import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error, r2_score

# Load and concatenate the dataset from multiple files
pa_files = os.listdir(
    'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join(
    'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f))
                  for f in pa_files])

# Convert 'datetime' to datetime object and sort the data
data['datetime'] = pd.to_datetime(data['datetime'])
data.sort_values('datetime', inplace=True)

# Selecting relevant features
features = ['temperature', 'humidity', 'pm25_cf_1']
X = data[features]
y = data['epa_pm25']  # Assuming we want to predict 'epa_pm25'

# Normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create sequences
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        if i + time_steps < len(y):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Creating sequences with the past 24 hours to predict the next hour
time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Function to create WLSTM model with Attention
def create_wlstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to calculate RMSE and R^2
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Train the WLSTM model multiple times and collect metrics
rmses = []
r2s = []
for i in range(10):
    wlstm_model = create_wlstm_model((time_steps, X_seq.shape[2]))
    wlstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    rmse, r2 = evaluate_model(wlstm_model, X_test, y_test)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print the average RMSE and R^2 scores
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the evaluation metrics for each model run to a log file
with open('wlstm_model_evaluation_80_20_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i + 1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
