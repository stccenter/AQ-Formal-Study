1. Install Necessary Libraries
Before running the script, ensure you have the required libraries installed. You can install them using pip if you haven't already:

```
pip install pandas numpy scikit-learn
```

2. Prepare Your Data
Ensure your dataset is properly set up and located in the directory specified in the script.

Directory Structure: Make sure that the directory C:/~/TrainData contains the training data files in CSV format.

Data Filtering: The script filters the data based on a Pearson correlation threshold (e.g., 0.7). Ensure your dataset has a 'PearsonR' column for this filtering step.

3. Configure Hyperparameters in Script
The Linear Regression model in this script is straightforward and doesn’t have many hyperparameters like other models. If you need to adjust the script:

Locate the Script File: Open the script file where the model is defined.

Adjust Data Handling: For example, you can change the features being used for training. In this section, you can modify the variables in the dataset:

```
X = data[["pm25_cf_1", "temperature", "humidity"]]  # Adjust the variables here as needed
y = data['epa_pm25']
```

4. Run the Training Script
After ensuring the data is correctly loaded and formatted:

Locate the Training Script: Ensure you are in the directory where the script is located.

Run the Training Script:

Open your terminal or command prompt.
Navigate to the directory where the script is stored.
Execute the script with the following command:
```
python ols_scikit.py
```
The script will load the data, filter it based on the PearsonR threshold, apply the configured model, and train the Linear Regression model multiple times.

5. Retrieve the Results
Once the model training is complete, retrieve the results:

Results Location: The results, including the RMSE and R² for each model iteration, will be printed directly in the terminal. The script also calculates aggregate RMSE and R² values.

Save or Copy Results: The script saves the log of all values (RMSE and R² for each iteration and their aggregate values) to a text file named ols_linear_regressor_scikit_80_20_model_log.txt. You can find this file in the directory where the script was executed.

6. Integrate Results
Once you have the results:

Consolidate Results: Open your main results file (e.g., sample_results.csv) used for aggregating results from multiple models.

Append the Linear Regression Results: Manually copy the Linear Regression results from the log file and append them to the results file. Ensure you include key metrics such as the model name (Linear Regression), RMSE, R², and any other relevant information.

Ensure Consistency: Make sure the results are formatted consistently with results from other models to simplify future analysis and visualization.
