import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score

# Load data from .npy files
X_train = np.load('x_train.npy')
X_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

with open("rnn_7030.txt", "w") as results_file:
    all_rmse = []
    all_r2 = []
    all_times = []
    
    for i in range(10):
        start_time = time.time()  # start time

        model = Sequential()
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(SimpleRNN(50, return_sequences=False))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
        
        history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
        
        # use the X_test and y_test to test
        y_pred = model.predict(X_test).flatten()
        y_test_numpy = y_test.flatten()  # Flatten y_test to match the shape of y_pred
        mse = mean_squared_error(y_test_numpy, y_pred)
        r2 = r2_score(y_test_numpy, y_pred)

        end_time = time.time()  # end time
        elapsed_time = end_time - start_time  # calculate seconds

        # results
        result_string = f"Model {i + 1}: RMSE = {math.sqrt(mse)}, R² = {r2}, Time: {elapsed_time} seconds\n"
        print(result_string)
        results_file.write(result_string)

        all_rmse.append(math.sqrt(mse))
        all_r2.append(r2)
        all_times.append(elapsed_time)

    average_rmse = np.mean(all_rmse)
    average_r2 = np.mean(all_r2)
    average_time = np.mean(all_times)

    with open("rnn_7030aggregate.txt", "w") as aggregate_results_file:
        average_result_string = f"Average RMSE: {average_rmse}, Average R²: {average_r2}\nAverage time: {average_time} seconds\n"
        print(average_result_string)
        aggregate_results_file.write(average_result_string)

print("done")