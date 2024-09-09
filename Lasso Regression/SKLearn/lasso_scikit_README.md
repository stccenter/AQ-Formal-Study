1. Install Required Libraries
Ensure that the necessary libraries are installed in your Python environment. You can install them using the following command:

```
pip install pandas numpy scikit-learn
```
2. Load and Prepare Data
Your data is stored in multiple CSV files within a directory, and the code automatically loads and combines these files into a single dataset.

Step-by-step to load data:

Directory: All data files should be located in 'C:/~/TrainData'.

Filter by PearsonR: Rows with a PearsonR value below 0.7 will be excluded:

```
data = data[data['PearsonR'] >= 0.7]
```
Features and Labels: The features for training (X) include PM2.5, temperature, and humidity, while the target variable (y) is epa_pm25.
3. Data Splitting and Standardization
Before training, the dataset is split into a training set (80%) and a test set (20%).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
To standardize the data and ensure that all features have the same scale, the StandardScaler is used:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
4. Train the Lasso Regression Model
The code uses the Lasso regression model to fit the training data and make predictions. It uses an alpha value of 1.0 for regularization, and the model is trained and tested 10 times.

```
model = Lasso(alpha=1.0, random_state=42)
```
For each iteration, it computes the RMSE and R² values for the model's performance and stores them in the rmses and r2s lists:

```
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, alpha=1.0)
    rmses.append(rmse)
    r2s.append(r2)
```
5. Aggregate Results
After running the training and testing 10 times, the script calculates and prints the average RMSE and R² across all the iterations:

```
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")
```
6. Save the Results
The script also saves both individual results for each iteration and the aggregate results into a log file called lasso_regressor_80_20_model_log.txt:

```
with open('lasso_regressor_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
```
7. Running the Script
To run the script:

Open your terminal or command prompt.
Navigate to the directory where your Python script is saved.
Run the script using the following command:
```
python lasso_scikit.py
```
Once the script completes execution, the RMSE and R² values will be printed in the terminal, and the log file will be saved with the results for future reference.
