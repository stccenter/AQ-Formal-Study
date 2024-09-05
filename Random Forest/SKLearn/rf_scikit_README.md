1. Install Necessary Libraries
Ensure you have the necessary Python libraries installed before running the script. The script uses pandas, numpy, and scikit-learn. You can install these via pip if needed:

```
pip install pandas numpy scikit-learn
```

2. Prepare Your Data
Make sure your dataset is located in the specified directory.

Directory Structure: Ensure the directory C:/~/TrainData contains the training data files in CSV format. The script assumes that all files in this folder will be concatenated into one dataframe.

Data Filtering: The script filters the data based on a Pearson correlation threshold (e.g., 0.7). Ensure your dataset has a 'PearsonR' column to support this filtering.

3. Configure Hyperparameters in Script
The Random Forest model has tunable hyperparameters like n_estimators and max_depth. You may want to adjust these to fit your specific use case.

Locate the Script File: Open the script file where the model is defined.

Edit Hyperparameters: Find the section where the RandomForestRegressor is initialized. For example:

```
rmse, r2 = train_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)
```

You can change the values of n_estimators (number of trees) or max_depth (maximum depth of each tree) to suit your needs. For instance, you could try increasing the number of trees:

```
rmse, r2 = train_model(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=15)
```

4. Run the Training Script
After configuring the hyperparameters:

Locate the Training Script: Ensure you are in the directory where the script is located.

Run the Training Script:

Open your terminal or command prompt.
Navigate to the directory where the script is stored.
Execute the script with the following command:
```
python scikit_random_forest.py
```
The script will load the data from the C:/~/TrainData directory, apply the configured model, and train the Random Forest regressor multiple times.

5. Retrieve the Results
Once the model training is complete, retrieve the results:

Results Location: The results, including the RMSE and R² for each model iteration, will be printed directly in the terminal. The script also calculates the aggregate RMSE and R² values.

Save or Copy Results: The script saves the log of all values (RMSE and R² for each iteration, and the aggregate values) to a text file named random_forest_regressor_scikit_80_20_model_log.txt. This file will be created in the same directory as the script.

6. Integrate Results
Once you have the results:

Consolidate Results: Open the main results file (e.g., sample_results.csv) used for aggregating results from multiple models.

Append the Random Forest Results: Manually copy the Random Forest regression results from the log file and append them to your results file. Include the key metrics, such as the model name (Random Forest), RMSE, R², and any other relevant information.

Ensure Consistency: Ensure the results are formatted consistently with those from other models for easy analysis and visualization later.

