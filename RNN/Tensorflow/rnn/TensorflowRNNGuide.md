# Recurrent Neural Network Tensorflow Guide

## Datasets

The datasets are located in the `Training Data` folder. This folder contains the sensor pairs used for the training.
Sequential dataset is required for the LSTM model, this dataset can be downloaded from the `XX` folder.

## Downloading the Full Dataset

All training data files are stored in the `Training Data` folder. Each dataset is in CSV format and includes necessary variables such as sensor readings, timestamps, and temperature and relative humidity.

After downloading the data, navigate to the directory where the files are saved. This file path will be used later to populate the train and test data objects in the appropriate code blocks.

## Environment Setup

The repository is organized into architecture-specific folders. The scripts can be downloaded from each top level folder. Tensorflow has combatible models (DT, RF, SNN, DNN, OLS, LSTM, and RNN). Navigate to the model architecture/s for your specific use case, and download scripts

### Install the Required Software

The models were built using Tensorflow and other supporting packages. To install using pip package manager, run the following commands:

1. Pandas, Scikit - Learn, and NumPy 
```bash
pip install numpy pandas scikit-learn
```

2. Ensure Python Version Compatibility & Tensorflow Installation 
TensorFlow and other libraries may have version constraints depending on your Python version. TensorFlow 2.x is typically compatible with Python 3.6–3.9

```bash
python --version
```

Install Tensorflow with Pip
```bash
python3 -m pip install tensorflow[and-cuda]
```
### Verify the installation:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
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

Navigate to the the Top Level SNN module then in the Tensorflow directory download the rnn.py file and in your terminal run:

```bash
rnn.py
```

To integrate all Results and visualize results refer to general guide 

After running the models, the results will be saved to a <specified file name>.csv file. This file will include key metrics such as:

- R² (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- Training time