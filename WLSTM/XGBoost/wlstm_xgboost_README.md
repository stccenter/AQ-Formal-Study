1. Install Necessary Libraries
Make sure you have the required libraries installed before running the script. You will need pandas, numpy, scikit-learn, keras, and keras_self_attention. You can install them using the following pip commands:

```
pip install pandas numpy scikit-learn keras keras-self-attention
```

2. Prepare Your Data
Ensure that your dataset is located in the correct directory.

Directory Structure: Make sure the directory C:/~/TrainData contains the training data files in CSV format. The script concatenates these files into a single dataframe for model training.

Datetime and Sorting: The script requires that the dataset includes a 'datetime' column to correctly order the time-series data. Ensure this column exists and is in the appropriate format.

3. Configure the Script
This script builds an LSTM model with attention to predict PM2.5 levels. The script uses sequences of 24 hours of historical data to predict the PM2.5 level for the next hour.

Edit Hyperparameters: You can adjust various model parameters, including time_steps, epochs, and batch_size. For example:

```
time_steps = 24  # How many hours of data to use as input
epochs = 10  # Number of training epochs
batch_size = 32  # Size of training batches
```

You may also modify the LSTM units or the model architecture as needed. For instance, increasing the units in the LSTM layer:

```
model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
```

4. Run the Training Script
Once you have configured the script to your needs:

Locate the Training Script: Ensure you are in the directory where the script is saved.

Run the Training Script:

Open your terminal or command prompt.
Navigate to the directory where the script is stored.
Execute the script with the following command:
```
python wlstm_xgboost.py
```

The script will load the data from the C:/~/TrainData directory, preprocess it, and train the WLSTM model with self-attention multiple times (10 iterations in this case).

5. Retrieve the Results
Once model training is complete, retrieve the results:

Results Location: The RMSE and R² values for each model iteration, along with aggregate metrics, are printed to the terminal. Additionally, the script saves these metrics to a log file named wlstm_model_evaluation_80_20_log.txt.

Save or Copy Results: The log file will contain the evaluation metrics for each of the 10 model runs, as well as the aggregate RMSE and R² scores. You can use this file for later analysis or reference.

6. Integrate Results
Once you have the results:

Consolidate Results: Open your main results file (e.g., sample_results.csv) where you aggregate results from all models.

Append the WLSTM Results: Manually copy the WLSTM model results from the log file and append them to your results file. Include key metrics such as the model name (WLSTM with Attention), RMSE, R², and other relevant details.

Ensure Consistency: Make sure the formatting of the results is consistent with other models for easy analysis and comparison later on.
