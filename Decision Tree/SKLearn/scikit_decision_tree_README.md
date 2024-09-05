1. Install Required Libraries
Before running the script, make sure you have all the necessary Python libraries installed. You can install the required libraries using pip:

```
pip install pandas numpy scikit-learn
```

2. Configure Hyperparameters in the Script
Unlike XGBoost, the DecisionTreeRegressor does not use external configuration files for hyperparameters. Instead, you can directly modify them within the script.

Locate the Hyperparameters:
Open the script file where the model is defined.
Find the section where the DecisionTreeRegressor model is instantiated. It should look something like this:
```
model = DecisionTreeRegressor(random_state=42)
```

Edit Hyperparameters:
Modify the hyperparameters based on your experiment requirements. For example, you could change the maximum depth of the tree:

```
model = DecisionTreeRegressor(max_depth=5, random_state=42)
```

3. Run the Training Script
After configuring the hyperparameters, run the training script to train the model.

Locate the Training Script:
Ensure you are in the directory where the script is saved.

Run the Script:
Open your terminal or command prompt.
Navigate to the directory containing your script.
Execute the script by typing:
```
python scikit_decision_tree.py
```

4. Retrieve the Results
Once the model training is complete, the script will output the results.

Results Location:
The results, including the RMSE (Root Mean Squared Error) and R² (R-squared) for each model run, will be printed directly to the terminal or command prompt.

Save or Copy Results:
The script is designed to log all values to a text file named scikit_decision_tree_regressor_80_20_model_log.txt. This file will include RMSE and R² for each model run, as well as the aggregate RMSE and R².

5. Integrate Results
Once you have the results, you may want to consolidate them for further analysis.

Consolidate Results:
Open the scikit_decision_tree_regressor_80_20_model_log.txt file to view the detailed log of each run.

If you have an overall sample_results.csv file for aggregating results from multiple models, append the DecisionTreeRegressor results manually.

Ensure you include the model name (DecisionTreeRegressor), RMSE, R², and any other relevant metrics.
Format the results consistently with other entries in your sample_results.csv file to facilitate easy analysis and visualization.
