"""REQUIRED PACKAGES"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.linalg import lstsq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torcheval.metrics import R2Score
from math import sqrt
import time

""" DATA PREPROCESSING (Skip if train and test CSV files are downloaded) """


class PM25Dataset(Dataset):
    def __init__(self, features, labels):
        """
        Convert features and labels from Pandas DataFrames to Numpy arrays for faster access.
        """
        self.features = features.to_numpy(dtype=np.float32)
        self.labels = labels.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Efficiently fetch a single data point with its label.
        """
        sample_features = torch.tensor(self.features[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample_features, sample_label


def get_data(data_directory):
    # List all files in the specified directory
    pa_files = os.listdir(data_directory)
    # Read each CSV file and concatenate them into one DataFrame
    data = pd.concat([pd.read_csv(os.path.join(data_directory, f)) for f in pa_files])
    # Convert the 'datetime' column to a datetime object
    data["datetime"] = pd.to_datetime(data["datetime"])
    # Set the 'datetime' column as the index
    data.set_index("datetime", inplace=True)
    # filter data baased on PearsonR (e.i.: 0.7)
    data = data[data["PearsonR"] >= 0.7]
    # Ensure columns are numeric, coercing errors to NaN
    numeric_columns = ["epa_pm25", "pm25_cf_1", "temperature", "humidity"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    """
    Aggregate the data to hourly granularity and drop rows with missing values
    """
    # hourly_data = data[numeric_columns].groupby(pd.Grouper(freq='h')).mean().dropna(subset=['epa_pm25', 'pm25_cf_1'])
    # Extract features and target variable

    new_X = data[["pm25_cf_1", "temperature", "humidity"]]
    y = data["epa_pm25"]

    scaler = MinMaxScaler()
    new_X_scaled = scaler.fit_transform(new_X)
    new_X = pd.DataFrame(new_X_scaled, columns=new_X.columns, index=new_X.index)
    # print(new_X)
    X_train, X_test, y_train, y_test = train_test_split(
        new_X, y, test_size=0.2, shuffle=False
    )

    train_dataset = PM25Dataset(X_train, y_train)
    test_dataset = PM25Dataset(X_test, y_test)

    return train_dataset, test_dataset


data_directory = "<local>/SystematicStudy/TrainData"
train_dataset, test_dataset = get_data(data_directory)
train_size = len(train_dataset)
test_size = len(test_dataset)

print("Size of training dataset:", train_size)
print("Size of testing dataset:", test_size)


"""MODEL ARCHITECTURE AND CONFIGURATION"""


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


config = dict(
    epochs=100,
    batch_size=25,
    learning_rate=0.001,
    input_size=3,
    output_size=1,
    data_dir="/Users/shyralarea/SystematicStudy/TrainData",
)

"""TRAINING LOOP"""


def training_loop(
    train_loader, test_loader, device, n_epochs, model, optimiser, loss_fn
):
    train_r2_score_metric = R2Score()
    test_r2_score_metric = R2Score()
    for i in range(n_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        mse_train = 0.0
        for train_input, train_target in train_loader:
            train_input = train_input.unsqueeze(dim=1)
            train_input, train_target = train_input.to(device), train_target.to(device)
            optimiser.zero_grad()
            out = model(train_input)
            out = out.squeeze()
            loss = loss_fn(out, train_target)
            loss.backward()
            optimiser.step()

            # Train metrics calculation
            train_loss += loss.item() * train_input.size(0)
            mse_train += ((out - train_target) ** 2).sum().item()
            train_r2_score_metric.update(out, train_target)

        # Overall train metrics
        train_loss /= len(train_loader.dataset)
        mse_train /= len(train_loader.dataset)
        train_rmse = sqrt(mse_train)
        train_r2_score = train_r2_score_metric.compute()

        # Testing phase
        model.eval()
        test_loss = 0.0
        mse_test = 0.0
        with torch.no_grad():
            for test_input, test_target in test_loader:
                test_input = test_input.unsqueeze(dim=1)
                test_input, test_target = test_input.to(device), test_target.to(device)
                out = model(test_input)
                out = out.squeeze()
                loss = loss_fn(out, test_target)

                # Test metrics calculation
                test_loss += loss.item() * test_input.size(0)
                mse_test += ((out - test_target) ** 2).sum().item()
                test_r2_score_metric.update(out, test_target)

        # Overall test metrics
        test_loss /= len(test_loader.dataset)
        mse_test /= len(test_loader.dataset)
        test_rmse = sqrt(mse_test)
        test_r2_score = test_r2_score_metric.compute()

    return train_rmse, train_r2_score, test_rmse, test_r2_score


"""EXTRACT RESULTS (save as CSV to later use for visualizations)"""

# Lists to store results for each run
all_results = []

for run in range(10):
    start_time = time.time()

    # Get train and test data for the model
    train_dataset, test_dataset = get_data(config["data_dir"])

    # Data loader
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Building the model
    model = LinearRegressionModel(config["input_size"], config["output_size"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    # Defining the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # Training phase
    train_rmse, train_r2_score, test_rmse, test_r2_score = training_loop(
        train_loader, test_loader, device, config["epochs"], model, optimizer, criterion
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Append results to list
    all_results.append(
        {
            "Run": run + 1,
            "Train RMSE": train_rmse,
            "Train R2": train_r2_score,
            "Test RMSE": test_rmse,
            "Test R2": test_r2_score,
            "Time Elapsed": elapsed_time,
        }
    )

    # Print and save results
    print(
        f"Run {run + 1}: Train RMSE = {train_rmse:.4f}, Train R^2 = {train_r2_score:.4f}, Test RMSE = {test_rmse:.4f}, Test R^2 = {test_r2_score:.4f}, Time Elapsed = {elapsed_time:.2f} seconds"
    )

# Calculate aggregated results
results_df = pd.DataFrame(all_results)
mean_results = results_df.mean()

# Save results to CSV
results_df.to_csv(
    "/Users/shyralarea/SystematicStudy/ExperimentResult/NewPreprocessing/OLSresults.csv",
    index=False,
)

# Print mean results
print("\nAggregated Results over 10 Runs:")
print(
    f'Mean Train RMSE = {mean_results["Train RMSE"]:.4f}, Mean Train R^2 = {mean_results["Train R2"]:.4f}, Mean Test RMSE = {mean_results["Test RMSE"]:.4f}, Mean Test R^2 = {mean_results["Test R2"]:.4f}, Mean Time Elapsed = {mean_results["Time Elapsed"]:.2f} seconds'
)
