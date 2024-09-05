Install Required Libraries
To use the provided script, you will need to install the required Python libraries. If you don’t have them installed, you can install them using pip.

```
pip install pandas numpy scikit-learn
```

Configure Hyperparameters in the Script
Since there are no separate configuration files, you will need to edit the hyperparameters directly within the script.

Locate the Script File:
Navigate to the directory where the script is located.

Edit Hyperparameters:
Find the section in the script where the SVR model is initialized. You can adjust the parameters like C, kernel, gamma, and epsilon to suit your experiment needs. For example:

```
model = SVR(C=1.0, kernel='rbf', gamma='scale', epsilon=0.1)
```

You can modify these values to experiment with different settings:

```
model = SVR(C=10.0, kernel='linear', gamma='auto', epsilon=0.01)
```

Run the Training Script
After you have configured the hyperparameters, you can proceed to run the script.

Locate the Training Script:
Make sure you are in the directory where the SVR training script is saved.

Run the Script:
Open your terminal or command prompt and navigate to the script’s directory. Run the script by typing the following command:

```
python svm.py
```

The script will load the dataset, split it into training and test sets, standardize the features, and train multiple SVR models. It will print the RMSE and R² score for each model and calculate the aggregate RMSE and R² values.

Retrieve the Results
Once the model training is complete, the results will be displayed in the terminal.

Results Location:
Check the terminal for individual model metrics such as RMSE and R² scores. In addition, the script writes all model performance logs to a file named svr_80_20_model_log.txt.

Save or Copy Results:
The svr_80_20_model_log.txt file will contain the RMSE and R² values for each model as well as the aggregate RMSE and R² for all models. You can find this file in the same directory as the script.

Integrate Results
After obtaining the results:

Consolidate Results:
If you are aggregating results from multiple experiments, open a results file (such as a sample_results.csv) and append the SVR model performance metrics (e.g., RMSE, R² scores) to the file.

Format Results Consistently:
Ensure the results are formatted consistently for easier analysis. For example, include columns for model configuration (hyperparameters used), RMSE, and R² for each run.
