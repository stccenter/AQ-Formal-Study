1. Install Required Libraries
To ensure all dependencies are installed, run the following command:

```
pip install pandas numpy scikit-learn xgboost
```
2. Load and Prepare Data
Your data is stored in multiple CSV files within a directory, and the script automatically loads and combines these files into a single dataset.

Step-by-step to load data:

Directory: Make sure all data files are in 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData'.

Filter by PearsonR: The script filters out rows with PearsonR below 0.7:

```
data = data[data['PearsonR'] >= 0.7]
```
Features and Labels: The features used for training (X) are PM2.5, temperature, and humidity, while the target variable (y) is epa_pm25.
3. Data Splitting and Standardization
Before training, the data is split into a training set (80%) and a test set (20%).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
To ensure consistent performance, the data is standardized using StandardScaler:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
4. Train the XGBoost Model as a Random Forest
This code configures the XGBoost model (XGBRegressor) to function like a Random Forest by adjusting key hyperparameters, such as:

booster: Set to 'gbtree' to use decision trees.
eta: The learning rate is set to 0.1.
max_depth: The maximum depth of each tree is 9.
n_estimators: The number of trees is set to 100.
subsample: Set to 1 to use all data points for each tree.
colsample_bytree: Set to 1 to use all features for each tree.
```
model = XGBRegressor(booster='gbtree', eta=0.1, max_depth=9, n_estimators=100, subsample=1, colsample_bytree=1)
```
The script trains the model 10 times, computing the RMSE and R² for each iteration:

```
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, eta=0.1, max_depth=9, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)
```
5. Aggregate Results
After the 10 iterations, the script calculates and prints the aggregate RMSE and R²:

```
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")
```
6. Save the Results
The results for each model, as well as the aggregate results, are saved to a text file named random_forest_xgboost_80_20_model_log.txt:

```
with open('random_forest_xgboost_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
```
7. Running the Script
To run the script:

Open a terminal or command prompt.
Navigate to the directory where the Python file is located.
Run the script using:
```
python random_forest_XGBoost.py
```
Once the script finishes, the RMSE and R² will be printed in the terminal, and the log file containing all the results will be saved.

