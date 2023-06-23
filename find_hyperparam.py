import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, ConvLSTM2D, Flatten
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.svm import SVR


# Load the Purple Air PM2.5 data
pa_files = os.listdir('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData')
data = pd.concat([pd.read_csv(os.path.join('C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/TrainData', f)) for f in pa_files])

# Convert the 'datetime' column to a datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Set the 'datetime' column as the index
data.set_index('datetime', inplace=True)

# Aggregate the Purple Air data to the hourly granularity
hourly_data = data.groupby(pd.Grouper(freq='1H')).mean()
hourly_data = hourly_data.dropna(subset=['epa_pm25', 'pm25_cf_1'])


# Load the EPA and PA sensor pairs
sensor_pairs_file = 'C:/Users/Seren Smith/Documents/George Mason University/STC_Work/Purple Air/Pre-Processing/output/Pair_List.csv'
sensor_pairs_df = pd.read_csv(sensor_pairs_file, sep='\t')
sensor_pairs = sensor_pairs_df.values.tolist()
sensor_pairs = [tuple(map(int, pair_str[0].split(','))) for pair_str in sensor_pairs]

# Preprocess the data to handle missing values
#imputer = SimpleImputer(strategy='median')
new_X = hourly_data[['pm25_cf_1', 'temperature', 'humidity']]
#new_X = hourly_data['calibrated']
#X_imputed = imputer.fit_transform(new_X)
y = hourly_data['epa_pm25']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=42)

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}
svr_model = SVR()
grid_search = GridSearchCV(svr_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("grid search fit")
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# #List Hyperparameters that we want to tune.
#leaf_size = list(range(1,50))
#n_neighbors = list(range(1,30))
#p=[1,2]
# #Convert to dictionary
#hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# #Create new KNN object
#knn_2 = KNeighborsRegressor()
# #Use GridSearch
#model = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
#best_model = model.fit(X_train,y_train)
# #Print The value of best Hyperparameters
#print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
#print('Best p:', best_model.best_estimator_.get_params()['p'])
#print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])