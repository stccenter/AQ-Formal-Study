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

# Loop through each CSV file and concatenate its data to the combined dataset
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    current_dataset = pd.read_csv(file_path)
    combined_dataset = pd.concat([combined_dataset, current_dataset], ignore_index=True)

print(combined_dataset.columns)

df = combined_dataset[['temperature', 'humidity', 'pm25_cf_1', 'epa_pm25', 'PearsonR']]

# ADDED, filter the data based on PearsonR
df = df[df['PearsonR'] >= 0.7]

# drop na values
df.dropna(subset=['temperature', 'humidity', 'pm25_cf_1', 'epa_pm25'], inplace=True)

print("pre\n", df)

x = df[['pm25_cf_1', 'temperature', 'humidity']].values
y = df['epa_pm25'].values

# Normalize the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

# put x and y back together to split later
df_normalized = np.concatenate((scaled_x, y.reshape(-1, 1)), axis=1)

# Convert the combined array into a DataFrame
df_normalized = pd.DataFrame(df_normalized, columns=['pm25_cf_1', 'temperature', 'humidity', 'epa_pm25'])

print("post\n", df_normalized)

all_rmse = []
all_r2 = []
all_times = [] 

# define the function to split test and train
def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

# split the data
train_ds, test_ds = split_dataset(df_normalized)
print("{} examples in training, {} examples for testing.".format(len(train_ds), len(test_ds)))        
x_train = train_ds[['temperature', 'humidity', 'pm25_cf_1']].values
y_train = train_ds['epa_pm25'].values
x_test = test_ds[['temperature', 'humidity', 'pm25_cf_1']].values
y_test = test_ds['epa_pm25'].values  

# txt file for logging
with open("ols_7030.txt", "w") as results_file:
    for i in range(10):
        start_time = time.time()  # start time

        model = models.Sequential()
        model.add(layers.Dense(1, input_shape=(x_train.shape[1],)))

        # minimizing mse
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='mse')
        
        # batch size is automatically 32
        model.fit(x_train, y_train, epochs=30, batch_size=32)

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

with open("ols_7030aggregate.txt", "w") as results_file:
    average_result_string = f"Average RMSE: {average_rmse}, Average R²: {average_r2}\nAverage time: {average_time} seconds\n"
    print(average_result_string)
    results_file.write(average_result_string)

print("done")
