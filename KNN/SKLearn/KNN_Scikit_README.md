1. Install Necessary Libraries
Before running the script, ensure you have all the required libraries installed. The script uses pandas, numpy, and scikit-learn. Install them using pip if you haven't already:

```
pip install pandas numpy scikit-learn
```

2. Prepare Your Data
Make sure your dataset is ready and located in the specified directory.

Directory Structure: Ensure that the directory C:/~/TrainData contains the training data files in CSV format.

Data Filtering: The script filters the data based on a Pearson correlation threshold (e.g., 0.7). Ensure your data has a 'PearsonR' column for this filtering step.

3. Configure Hyperparameters in Script
Since the script does not use a configuration file, you will need to manually adjust the hyperparameters directly within the script.

Locate the Script File: Open the script file that you want to modify.

Edit Hyperparameters: Find the section where the KNeighborsRegressor is initialized. For example:

```def train_model(X_train, y_train, X_test, y_test, n_neighbors):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    # Other model configurations
```
Adjust the hyperparameters as needed. For example, change the number of neighbors:

```n_neighbors = 5  # Change this value as needed
```

4. Run the Training Script
After configuring the hyperparameters:

Locate the Training Script: Ensure you are in the directory where the script is located.

Run the Training Script:

Open your terminal or command prompt.
Navigate to the script's directory.
Execute the script with the command:
```
python scikit_knn.py
```

The script will load the data, apply the configured hyperparameters, and train the KNN regression model multiple times.

5. Retrieve the Results
Once the model training is complete, retrieve the results:

Results Location: The results, including the RMSE and R² for each model, are printed directly to the terminal. Aggregate values for RMSE and R² are also displayed.

Save or Copy Results: The script saves the log of all values, including individual model performance and aggregate metrics, to a text file named knn_regressor_scikit_80_20_model_log.txt. You can find this file in the directory where the script was executed.

6. Integrate Results
Once you have the results:

Consolidate Results: Open the main results file (e.g., sample_results.csv) used for aggregating results from all models.

Append the KNN Results: Manually copy the KNN regression results from the log file and append them to the results file. Include key metrics like model name, RMSE, R², and any other relevant information.

Format Consistently: Ensure the results are formatted consistently with other models for easy analysis and visualization later.

By following these steps, you can effectively install the necessary libraries, configure your KNN regression script, run the model training, and integrate the results into your overall analysis.
