1. Install Required Libraries
To run this code, you need to ensure that the required Python libraries are installed.

Installing Required Libraries:

```
pip install pandas numpy scikit-learn
```

2. Load and Prepare Data
Your data is located in multiple CSV files within a directory. The script automatically loads these and combines them into a single dataset.

Step-by-step to load data:

Directory: Ensure all data files are stored in the directory 'C:/~/TrainData'.

Filter by PearsonR: The script filters out rows with a PearsonR value below 0.7:

```
data = data[data['PearsonR'] >= 0.7]
```
Features and Labels: Features used for training (X) are PM2.5, temperature, and humidity. The target variable (y) is epa_pm25.
3. Data Splitting and Standardization
Before training, the data is split into a training set (80%) and a testing set (20%).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Then, data is standardized using StandardScaler to ensure the neural network can perform optimally:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

4. Train the MLPRegressor Model
This part of the code trains a neural network (MLPRegressor) using the specified hyperparameters:

Hidden Layers: Two layers with 100 neurons each.
Activation: ReLU activation function.
Iterations: A maximum of 500 iterations.
```
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=500, random_state=42)
```
It runs the model training 10 times and computes the RMSE and R² for each model, storing the results:

```
for i in range(10):
    rmse, r2 = train_model(X_train, y_train, X_test, y_test, hidden_layer_sizes=(100, 100), activation='relu', max_iter=500)
    rmses.append(rmse)
    r2s.append(r2)
```
5. Aggregate Results
After training, the script calculates and prints the aggregate results across all 10 runs:

```
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2s)
print(f"Aggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}")
```
6. Save the Results
Finally, the script logs all individual and aggregate results in a text file named dnn_regressor_scikit_80_20_model_log.txt:

```
with open('dnn_regressor_scikit_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: RMSE = {rmses[i]}, R^2 = {r2s[i]}\n")
    file.write(f"\nAggregate RMSE: {avg_rmse}, Aggregate R^2: {avg_r2}\n")
```

7. Running the Script
To run the script:

Open a terminal or command prompt.
Navigate to the directory where the Python file is saved.
Run the script:
```
python scikit_dnn.py
```
After the script completes execution, you’ll have the RMSE and R² values printed in the terminal and saved in a log file for future reference.
