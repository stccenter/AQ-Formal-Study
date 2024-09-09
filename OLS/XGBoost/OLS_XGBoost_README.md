1. Install Required Libraries
First, ensure you have the necessary libraries installed. Run the following command:

```
pip install pandas numpy scikit-learn xgboost
```
2. Load and Prepare Data
The script loads multiple CSV files from a specified directory and concatenates them into a single dataset.

Step-by-step to load data:

Directory: Ensure all your data files are in 'C:/~/TrainData'.

Filter Data by PearsonR: The script filters rows where PearsonR is below 0.7:

```
data = data[data['PearsonR'] >= 0.7]
```
Features and Labels: The features used for training (X) include PM2.5, temperature, and humidity, while the target variable (y) is epa_pm25.
3. Data Splitting and Standardization
The dataset is split into training (70%) and testing (30%) sets:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Before training, the features are standardized using StandardScaler to ensure that all features are on the same scale:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
4. Train the XGBoost Model
The XGBoost model (XGBRegressor) is trained with the following hyperparameters:

eta: 0.1 (learning rate)
max_depth: 6 (maximum tree depth)
n_estimators: 100 (number of boosting rounds)
The model is trained 10 times, and for each iteration, the RMSE and R² values are computed and stored:

```
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, eta=0.1, max_depth=6, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)
```
5. Aggregate Results
Once the model has been trained 10 times, the script calculates the average RMSE and R² across all iterations and prints them:

```
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")
```
6. Save the Results
The script saves both the individual results for each model and the aggregate results in a text file named ols_70_30_model_log.txt:

```
with open('ols_70_30_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
```
7. Running the Script
To run the script:

Open your terminal or command prompt.
Navigate to the directory where your Python script is located.
Run the script using the following command:
```
python ols_xgboost.py
```

Once the script finishes execution, the RMSE and R² values will be printed in the terminal, and the log file containing individual and aggregate results will be saved.
