"""REQUIRED PACKAGES"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import math
import torch.optim as optim
import torch.utils.data as data
from torcheval.metrics import R2Score
from math import sqrt

""" DATA PREPROCESSING (Skip if train and test CSV files are downloaded) """


class PM25Dataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label based on the index.
        """
        sample_features = self.features[idx]
        sample_label = self.labels[idx]

        return sample_features, sample_label

X_train = np.load('<Local>/SystematicStudy/updt_seq_npy_arrays/x_train.npy')
y_train = np.load('<Local>/SystematicStudy/updt_seq_npy_arrays/y_train.npy')
X_test = np.load('<Local>/SystematicStudy/updt_seq_npy_arrays/x_test.npy')
y_test = np.load('<Local>/SystematicStudy/updt_seq_npy_arrays/y_test.npy')


train_dataset = PM25Dataset(X_train, y_train)
test_dataset = PM25Dataset(X_test, y_test)

"""MODEL ARCHITECTURE AND CONFIGURATION"""

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        
        # Assuming num_layers is intended for stacking LSTMCells, here's a basic adaptation
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize LSTM for each layer using pytorch built in package 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        
        # Fully connected layers
        self.fc_1 = nn.Linear(hidden_size, 64)  # First fully connected layer
        self.fc_2 = nn.Linear(64, 128)  # Second fully connected layer
        self.fc_3 = nn.Linear(128, 256)  # Second fully connected layer
        self.fc = nn.Linear(256, output_size)  # Final fully connected layer

        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Batch size for dynamic allocation
        #batch_size = x.size(0)
        
        # LSTM input shape: (batch_size, seq_len, input_size)
        # Output shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        # Initial states
        #h_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        #c_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        # Container for outputs
        #outputs = []
        
        """
        # Process each time step
        for i in range(x.size(1)):  # Assuming x is of shape (batch, seq_len, input_size)
            input_t = x[:, i, :]
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_cells[layer](input_t, (h_t[layer], c_t[layer]))
                input_t = h_t[layer]  # Pass the output as the next input
                
            outputs.append(h_t[-1].unsqueeze(1))  # Take the output from the last layer
        """
        
        # Concatenate outputs along the time dimension
        #outputs = torch.cat(outputs, dim=1) 
        # Taking the last time step's output for prediction
        #x = outputs[:, -1, :]
        

        x = self.fc_1(lstm_out)  # Pass through the first fully connected layer
        x = self.relu(x)  # Apply ReLU activation
        # x = self.dropout(x)
        x = self.fc_2(x)  # Pass through the second fully connected layer
        x = self.relu(x)
        x = self.fc_3(x)  # Pass through the second fully connected layer
        x = self.relu(x)
        # x = self.dropout(x)# Apply ReLU activation
        x = self.fc(x)    # Final output through the third fully connected layer
        return x

config = dict(
    epochs = 30,
    batch_size = 32,
    learning_rate=0.001,
    input_size = 3, 
    hidden_size = 32,
    output_size = 1, 
    units1 = 50, 
    units2 = 50, 
    num_layers = 3,
    weight_decay = 1e-4,
    )

"""TRAINING LOOP & EXTRACT RESULTS"""

def training_loop(train_loader, test_loader, device, n_epochs, model, optimiser, loss_fn):
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
            #train_input = train_input.unsqueeze(dim=2)
            
            # Reshape the input tensor to have the correct dimensions
            train_input = train_input.view(train_input.size(0), -1, train_input.size(-1))
            
            train_input, train_target = train_input.to(device), train_target.to(device)  
            optimiser.zero_grad()
            out = model(train_input)
            out = out.squeeze()
            loss = loss_fn(out, train_target)
            loss.backward()
            optimiser.step()
            
            #train metrics calculation
            train_loss += loss.item() * train_input.size(0)
            mse_train += ((out - train_target) ** 2).sum().item()
            train_r2_score_metric.update(out, train_target)
       
            # Update R-squared variables
            # total_variance += ((train_target - train_target.mean()) ** 2).sum().item()
            # explained_variance += ((train_target - out) ** 2).sum().item()
            
        #Overall train metrics
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
                #test_input = test_input.unsqueeze(dim=2)
                
                # Reshape the input tensor to have the correct dimensions
                test_input = test_input.view(test_input.size(0), -1, test_input.size(-1))

                test_input, test_target = test_input.to(device), test_target.to(device)
                out = model(test_input)
                out = out.squeeze()
                loss = loss_fn(out, test_target)
                
                #test metrics calculation
                test_loss += loss.item() * test_input.size(0)
                mse_test += ((out - test_target) ** 2).sum().item()
                test_r2_score_metric.update(out, test_target)
            
        # Overall test metrics
        test_loss /= len(test_loader.dataset)
        mse_test /= len(test_loader.dataset)
        test_rmse = sqrt(mse_test)
        test_r2_score = test_r2_score_metric.compute()
        
        print(f'Epoch {i+1}, Training Loss: {train_loss:.4f}, Training MSE: {mse_train:.4f}, '
              f'Training RMSE: {train_rmse:.4f}, Training R²: {train_r2_score:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test MSE: {mse_test:.4f}, '
              f'Test RMSE: {test_rmse:.4f}, Test R²: {test_r2_score:.4f}')
        
        # print(f'Epoch {i+1}, Training Loss: {train_loss:.4f}, Training MSE: {mse_train:.4f}, Training RMSE: {train_rmse:.4f}, Training R²: {train_r2_score:.4f}')
        # print(f'Epoch {i+1}, Test Loss: {test_loss:.4f}, Test MSE: {mse_test:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2_score:.4f}')
