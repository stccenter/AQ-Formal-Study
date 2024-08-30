import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
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
features = ['epa_pm25', 'temperature', 'humidity']
X = data[features]
y = data['epa_pm25']  # Assuming we want to predict 'epa_pm25'

# Normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create sequences
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Add a check to ensure i + time_steps is within the range of y
        if i + time_steps < len(y):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y.iloc[i + time_steps])  # Use iloc for safe indexing
    return np.array(Xs), np.array(ys)


# Creating sequences with the past 24 hours to predict the next hour
time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)

# Creating the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X_seq.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# Function to create a new LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
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


# Train the LSTM model multiple times and collect metrics
rmses = []
r2s = []
for i in range(10):
    # Create a new model instance for each run
    lstm_model = create_lstm_model((time_steps, X_seq.shape[2]))

    # Train the model
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Set verbose to 0 to reduce output

    # Evaluate the model
    rmse, r2 = evaluate_model(lstm_model, X_test, y_test)
    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print the average RMSE and R^2 scores
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")

# Save the evaluation metrics for each model run to a log file
with open('lstm_model_evaluation_70_30_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i + 1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")