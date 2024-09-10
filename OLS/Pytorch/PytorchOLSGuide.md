# Ordinary Least Squares Pytorch Guide

## Datasets

The datasets are located in the `Training Data` folder. This folder contains the sensor pairs used for the training.
Sequential dataset is required for the LSTM model, this dataset can be downloaded from the `XX` folder.

## Downloading the Full Dataset

All training data files are stored in the `Training Data` folder. Each dataset is in CSV format and includes necessary variables such as sensor readings, timestamps, and temperature and relative humidity.

After downloading the data, navigate to the directory where the files are saved. This file path will be used later to populate the train and test data objects in the appropriate code blocks.

## Environment Setup

The repository is organized into architecture-specific folders. The scripts can be downloaded from each top level folder. Pytorch has combatible models (SNN, DNN, OLD, LSTM, and RNN). Navigate to the model architecture/s for your specific use case, and download scripts

### Install the Required Software

The models were built using PyTorch and other supporting packages. To install using pip package manager, run the following commands:

1. Pandas and NumPy
```bash
pip install pandas numpy
```

2. PyTorch and TorchEval
```bash
pip install torch torcheval
```

3. Scikit - Learn
```bash
pip install scikit-learn
```

4. Matplotlib
```bash
pip install matplotlib
```

To use an IDE to create these models we use Jupyter Notebook. Install and open Jupyter Notebook using the following commands:

```bash
pip install jupyterlab
jupyter lab
```
## Models

Each folder inside the models directory contains models trained with different air quality sensor calibration algorithms. These models include approaches using LSTM, DNN, and other machine learning techniques. The scripts are provided as .py files. 

### Required Modifications
The datasets inside these folders are similar, yet LSTM uses sequential data. The primary differences are in the constants that configure the models to ensure standardization across software, such as:

Number of Channels and Filters
Epochs
Learning Rate 
Activation and Optimization Functions 

### To use the models:
- Populate the Train and Test objects: Define the path to the dataset you downloaded earlier.
- Use the default Hyperparameters: These are specified in the 'config' call within each model's script unless modifications are required for your specific use case.

### Running the Models

Navigate to the the Top Level OLS module then download the PytorchOLS.py file and in your terminal run:

```bash
Python3 PytorchOLS.py
```

To integrate all Results and visualize results refer to general guide 

After running the models, the results will be saved to a <specified file name>.csv file. This file will include key metrics such as:

- R² (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- Training time