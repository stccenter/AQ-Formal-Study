import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Function to train the model and return accuracy and F1 score
def train_model(X_train, y_train, X_test, y_test, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  # Change average as needed
    return accuracy, f1

# Load the dataset
pa_files = os.listdir( 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join( 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Assume 'y' is a categorical variable for classification
X = data[["pm25_cf_1", "temperature", "humidity"]]  # Example set of variables
y = data['epa_pm25']  # Replace with your categorical target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
accuracies = []
f1_scores = []
for i in range(10):
    accuracy, f1 = train_model(X_train, y_train, X_test, y_test, max_depth=5)  # Example max_depth
    accuracies.append(accuracy)
    f1_scores.append(f1)

# Calculate and print aggregate accuracy and F1 score
avg_accuracy = np.mean(accuracies)
avg_f1 = np.mean(f1_scores)
print(f"Aggregate Accuracy: {avg_accuracy}, Aggregate F1 Score: {avg_f1}")

# Save the log of all values
with open('decision_tree_classifier_80_20_model_log.txt', 'w') as file:
    for i in range(10):
        file.write(f"Model {i+1}: Accuracy = {accuracies[i]}, F1 Score = {f1_scores[i]}\n")
    file.write(f"\nAggregate Accuracy: {avg_accuracy}, Aggregate F1 Score: {avg_f1}\n")
