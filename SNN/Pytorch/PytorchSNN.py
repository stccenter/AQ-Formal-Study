"""REQUIRED PACKAGES"""

import math, os
from math import sqrt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import relu
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score

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


## Load and preprocess the data
pa_files = os.listdir("<Local>/SystematicStudy/TrainData")
data = pd.concat(
    [
        pd.read_csv(os.path.join("<Local>/SystematicStudy/TrainData", f))
        for f in pa_files
    ]
)

# Convert the 'datetime' column to a datetime object
data["datetime"] = pd.to_datetime(data["datetime"])

# set the 'datetime' column as the index
data.set_index("datetime", inplace=True)

# aggregate the data to hourly granularity
hourly_data = data.groupby(pd.Grouper(freq="1H")).mean()
hourly_data = hourly_data.dropna(subset=["epa_pm25", "pm25_cf_1"])

## Preprocess the data to handle missing values
new_X = hourly_data[["pm25_cf_1", "temperature", "humidity"]]
y = hourly_data["epa_pm25"]


def get_data(data_directory):
    # List all files in the specified directory
    pa_files = os.listdir(data_directory)
    # Read each CSV file and concatenate them into one DataFrame
    data = pd.concat([pd.read_csv(os.path.join(data_directory, f)) for f in pa_files])
    # Convert the 'datetime' column to a datetime object
    data["datetime"] = pd.to_datetime(data["datetime"])
    # Set the 'datetime' column as the index
    data.set_index("datetime", inplace=True)
    # Ensure columns are numeric, coercing errors to NaN
    numeric_columns = ["epa_pm25", "pm25_cf_1", "temperature", "humidity"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Aggregate the data to hourly granularity and drop rows with missing values
    hourly_data = (
        data[numeric_columns]
        .groupby(pd.Grouper(freq="h"))
        .mean()
        .dropna(subset=["epa_pm25", "pm25_cf_1"])
    )
    # Extract features and target variable
    new_X = hourly_data[["pm25_cf_1", "temperature", "humidity"]]
    y = hourly_data["epa_pm25"]
    # scaler = MinMaxScaler()
    # new_X_scaled = scaler.fit_transform(new_X)
    # new_X = pd.DataFrame(new_X_scaled, columns=new_X.columns, index=new_X.index)
    # print(new_X)
    X_train, X_test, y_train, y_test = train_test_split(
        new_X, y, test_size=0.3, shuffle=False
    )

    train_dataset = PM25Dataset(X_train, y_train)
    test_dataset = PM25Dataset(X_test, y_test)

    return train_dataset, test_dataset


## Split the data
X_train, X_test, y_train, y_test = train_test_split(
    new_X, y, test_size=0.2, shuffle=False
)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()

"""MODEL ARCHITECTURE AND CONFIGURATION"""


# Define neural network model using PyTorch
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


config = dict(
    epochs=50,
    batch_size=25,
    learning_rate=0.0001,
    input_size=3,
    hidden_size=32,
    output_size=1,
    num_layers=3,
    weight_decay=1e-4,
    data_dir="/Users/shyralarea/SystematicStudy/TrainData",
)


"""TRAINING LOOP AND RESULTS EXTRACTION"""


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
        # total_variance = 0.0
        # explained_variance = 0.0
        for train_input, train_target in train_loader:
            train_input = train_input.unsqueeze(dim=1)
            train_input, train_target = train_input.to(device), train_target.to(device)
            optimiser.zero_grad()
            out = model(train_input)
            out = out.squeeze()
            loss = loss_fn(out, train_target)
            loss.backward()
            optimiser.step()

            # train metrics calculation
            train_loss += loss.item() * train_input.size(0)
            mse_train += ((out - train_target) ** 2).sum().item()
            train_r2_score_metric.update(out, train_target)

            # Update R-squared variables
            # total_variance += ((train_target - train_target.mean()) ** 2).sum().item()
            # explained_variance += ((train_target - out) ** 2).sum().item()

        # Overall train metrics
        train_loss /= len(train_loader.dataset)
        mse_train /= len(train_loader.dataset)
        train_rmse = sqrt(mse_train)
        train_r2_score = train_r2_score_metric.compute()
        # Calculate RMSE for both train and test sets
        # train_r_squared = 1 - (explained_variance / total_variance)

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

                # test metrics calculation
                test_loss += loss.item() * test_input.size(0)
                mse_test += ((out - test_target) ** 2).sum().item()
                test_r2_score_metric.update(out, test_target)

        # Overall test metrics
        test_loss /= len(test_loader.dataset)
        mse_test /= len(test_loader.dataset)
        test_rmse = sqrt(mse_test)
        test_r2_score = test_r2_score_metric.compute()

        print(
            f"Epoch {i+1}, Training Loss: {train_loss:.4f}, Training MSE: {mse_train:.4f}, "
            f"Training RMSE: {train_rmse:.4f}, Training R²: {train_r2_score:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test MSE: {mse_test:.4f}, "
            f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2_score:.4f}"
        )


print("Get train and test data for the model")
train_dataset, test_dataset = get_data(config["data_dir"])

print("Data loader")
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

print("Building the model...")
model = MLPModel(config["input_size"], config["hidden_size"], config["output_size"])
