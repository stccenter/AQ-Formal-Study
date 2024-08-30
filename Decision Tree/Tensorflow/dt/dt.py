import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import tensorflow_decision_forests as tfdf
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Check the versions
print(pd.__version__)
print(np.__version__)
print(tf.__version__)
print(tfdf.__version__)

# Path to the directory containing your CSV files
directory_path = "TrainData/"

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# Initialize an empty DataFrame to store the combined data
combined_dataset = pd.DataFrame()

# Loop through each CSV file and concatenate its data to the combined dataset
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    current_dataset = pd.read_csv(file_path)
    combined_dataset = pd.concat([combined_dataset, current_dataset], ignore_index=True)

dataset_df = combined_dataset[['temperature', 'humidity', 'pm25_cf_1', 'epa_pm25']]
print(dataset_df)

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

all_rmse = []
all_r2 = []
all_times = [] 

# txt file for logging
with open("dtregressor_8020.txt", "w") as results_file:
    for i in range(10):
        start_time = time.time()  # start time
        
        # split data
        train_ds_pd, test_ds_pd = split_dataset(dataset_df)
        print("{} examples in training, {} examples for testing.".format(len(train_ds_pd), len(test_ds_pd)))

        label = "epa_pm25"
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

        # type of model
        regression_model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)

        # training
        regression_model.fit(train_ds)

        # mse
        regression_model.compile(metrics=["mse"])
        evaluation = regression_model.evaluate(test_ds, return_dict=True)
        mse = evaluation['mse']

        # r^2
        y_true = test_ds_pd[label]
        y_pred = regression_model.predict(test_ds).flatten()
        r2 = r2_score(y_true, y_pred)

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
with open("dtregressor_8020aggregate.txt", "a") as results_file:  # Append to the log file
    average_result_string = f"Average RMSE: {average_rmse}, Average R²: {average_r2}\nAverage time: {average_time} seconds\n"
    print(average_result_string)
    results_file.write(average_result_string)
