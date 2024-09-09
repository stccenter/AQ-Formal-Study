1. Install Required Libraries
Ensure the necessary Python libraries are installed. You can do this using the following command:

```
pip install pandas numpy scikit-learn xgboost
```
2. Load and Prepare Data
Your data is stored in multiple CSV files within a directory. The script loads and combines all the files into a single dataset.

Step-by-step to load data:

Directory: Ensure all your data files are in 'C:/~/TrainData'.

Filter by PearsonR: The script filters out rows where PearsonR is less than 0.7:

```
data = data[data['PearsonR'] >= 0.7]
```
Features and Labels: The features used for training (X) include PM2.5, temperature, and humidity, while the target variable (y) is epa_pm25.
3. Data Splitting and Standardization
The dataset is split into a training set (80%) and a test set (20%).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Before training, the data is standardized using StandardScaler to ensure that the features have a similar scale, which is important for both KNN and XGBoost models:

```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
4. Train the Hybrid Model
In this hybrid approach:

KNN: The KNN model is trained first, and its predictions on the training set are added as a new feature for XGBoost.
XGBoost: The XGBoost model is then trained using the original features along with the KNN predictions as an additional feature.
Parameters:

KNN: n_neighbors is set to 5.
XGBoost: Uses eta=0.1, max_depth=9, and n_estimators=100.
```
rmse, r2 = train_hybrid_model(X_train_scaled, y_train, X_test_scaled, y_test, n_neighbors=5, eta=0.1, max_depth=9, n_estimators=100)
```
The training process is repeated 10 times, and for each iteration, the RMSE and R² values are computed and stored:

```
for i in range(10):
    rmse, r2 = train_hybrid_model(X_train_scaled, y_train, X_test_scaled, y_test, n_neighbors=5, eta=0.1, max_depth=9, n_estimators=100)
    rmses.append(rmse)
    r2s.append(r2)
```
5. Aggregate Results
After 10 iterations, the script calculates the average RMSE and R² values and prints them:

```
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")
```
6. Save the Results
The results of all the individual models as well as the aggregate results are saved to a text file named hybrid_knn_xgboost_80_20_model_log.txt:

```
with open('hybrid_knn_xgboost_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
```
7. Running the Script
To run the script:

Open your terminal or command prompt.
Navigate to the directory where the Python file is located.
Run the script using the following command:
```
python knn_xgboost.py
```

Once the script finishes execution, the RMSE and R² values will be printed in the terminal, and the log file containing the individual and aggregate results will be saved for future reference.
