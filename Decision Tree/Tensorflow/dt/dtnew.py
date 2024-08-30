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

# PREPROCESSING
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

print(combined_dataset.columns)

df = combined_dataset[['temperature', 'humidity', 'pm25_cf_1', 'epa_pm25', 'PearsonR']]
print(df)

# ADDED, filter the data based on pearsonr threhhold
df = df[df['PearsonR'] >= 0.7]

# drop na values
df.dropna(subset=['temperature', 'humidity', 'pm25_cf_1', 'epa_pm25', 'PearsonR'], inplace=True)

print("pre\n", df)

x = df[['pm25_cf_1', 'temperature', 'humidity']].values
y = df['epa_pm25'].values

# Normalize the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

# put x and y back together to split later
df = np.concatenate((scaled_x, y.reshape(-1, 1)), axis=1)

# Convert the combined array into a DataFrame
df = pd.DataFrame(df, columns=['pm25_cf_1', 'temperature', 'humidity', 'epa_pm25'])

print("post\n", df)

all_rmse = []
all_r2 = []
all_times = [] 

def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

# split data
train_ds_pd, test_ds_pd = split_dataset(df)
print("{} examples in training, {} examples for testing.".format(len(train_ds_pd), len(test_ds_pd)))
label = "epa_pm25"
        
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# txt file for logging
with open("dt_7030.txt", "w") as results_file:
    for i in range(10):
        start_time = time.time()  # start time
        
        # type of model, set regression, classification is automatic
        model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)

        # training
        model.fit(train_ds)

        # mse
        model.compile(metrics=["mse"])
        evaluation = model.evaluate(test_ds, return_dict=True)
        mse = evaluation['mse']

        # r^2
        y_test = test_ds_pd[label]
        y_pred = model.predict(test_ds).flatten()
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

with open("dt_7030aggregate.txt", "w") as results_file:  
    average_result_string = f"Average RMSE: {average_rmse}, Average R²: {average_r2}\nAverage time: {average_time} seconds\n"
    print(average_result_string)
    results_file.write(average_result_string)

print("done")
