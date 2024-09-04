1. Install XGBoost
To install XGBoost, follow the online guidance for your specific environment:

Visit the XGBoost Installation Guide for detailed instructions based on your operating system and preferred installation method.
For most users, you can install XGBoost using pip:
```pip install xgboost```

2. Configure Hyperparameters in Scripts
Since there are no configuration files and hyperparameters are set directly in the scripts, you will need to manually edit the necessary scripts to adjust the model settings.

Locate the Script Files:

Navigate to the XGBoost directory (example): XGBRegressor/XGBoost/.

Edit Hyperparameters:
Open the relevant script file (often named something like train.py or xgboost_train.py).
Search for the section where hyperparameters are defined. This might look like:
```
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    # Add other parameters here
)
```
Adjust these parameters according to your experiment requirements. For example:
```
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    # Add other parameters here
)
```

3. Run the Training Script
After configuring the hyperparameters:
Locate the Training Script:
Make sure you are in the XGBRegressor/XGBoost/ directory where the xgboost_text.py script is located.
Run the Training Script:

Open your terminal or command prompt.
Navigate to the XGBRegressor/XGBoost/ directory if you haven't already.
Type in CMD: ```python xgboost_test.py```

The script will load the data, apply the configured hyperparameters, and train the XGBoost model.

4. Retrieve the Results
Once the model training is complete, retrieve the results:

Results Location:
Check the terminal or command output. The results (such as R², RMSE, and elapsed time) are typically printed directly to the screen.
Save or Copy Results:
Manually copy the results from the terminal and paste them into a text file or directly into the overall sample_results.csv file if it is being used for aggregation.

5. Integrate Results
Once you have the results:
Consolidate Results:
Open the main sample_results.csv file used for aggregating results from all models.
Append the XGBoost results manually, including the model name, R², RMSE, time elapsed, and any other relevant metrics.
Ensure that the results are formatted consistently for easy analysis and visualization later.
