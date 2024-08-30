import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time  

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# check the versions
print(pd.__version__)
print(np.__version__)
print(tf.__version__)

# directory with csv files
directory_path = "TrainData/"

# list of all files
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# initialize dataframe for combined data
combined_dataset = pd.DataFrame()

# loop through files to concat data
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    current_dataset = pd.read_csv(file_path)
    combined_dataset = pd.concat([combined_dataset, current_dataset], ignore_index=True)
    

# define the function to split test and train
def split_dataset(dataset, test_ratio=0.20):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

# for calculating average
all_rmse = []
all_r2 = []
all_times = []  # Added list to store individual model run times

with open("snn_8020.txt", "w") as results_file:
    for i in range(10):
        start_time = time.time()  # start time

        # split the data
        train_ds_pd, test_ds_pd = split_dataset(combined_dataset)
        print("{} examples in training, {} examples for testing.".format(len(train_ds_pd), len(test_ds_pd)))        
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_ds_pd[['temperature', 'humidity', 'pm25_cf_1']].values)
        y_train = train_ds_pd['epa_pm25'].values
        x_test = scaler.transform(test_ds_pd[['temperature', 'humidity', 'pm25_cf_1']].values)
        y_test = test_ds_pd['epa_pm25'].values  
        
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)))
        # model.add(layers.Dense(, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))

        # minimizing mse
        model.compile(optimizer='adam', loss='mse')
        
        # batch size is automatically 32
        model.fit(x_train, y_train, epochs=30)

        # use the x_test and y_test to test
        y_pred = model.predict(x_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

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

with open("snn_8020aggregate.txt", "w") as results_file:
    average_result_string = f"Average RMSE: {average_rmse}, Average R²: {average_r2}\nAverage time: {average_time} seconds\n"
    print(average_result_string)
    results_file.write(average_result_string)

print("done8020snn")