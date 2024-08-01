import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# read the CSV file for the patient
def readCSVFile(fileName):
    stride_step = []
    try:
        with open(fileName, 'r') as file:
            header = next(file)
            for line in file:
                values = line.strip().split(',')
                try:
                    stride_step.append(tuple(map(float, values)))
                except ValueError as ve:
                    print(f"ValueError: {ve} for line: {line}")
    except FileNotFoundError:
        print("The file does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    stride_step_array = np.array(stride_step)
    #print(stride_step_array)
    return stride_step_array

#initalize the patient from the CSV file
patient = readCSVFile('/Users/arsshbajpai/PycharmProjects/Final Project AI/Gait Data/o1-76-si.csv')

# seperate the arrays into features and the target labels for predicitons
X = patient[:, :-1]
Y = patient[:, -1]

#split the arrays into testing and training with a 20-80 split for it
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#initalizing the regressor function wre using squared error and root mean square error
trainer = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

#train the model using the training data
trainer.fit(X_train, Y_train)

#make predicitions off of the testing data
y_pred = trainer.predict(X_test)

#calculate the residuals the difference between the actual and predictions for the data
residual = Y_test - y_pred

#reshape the residuals into a 2d array
residuals = residual.reshape(-1, 1)

#initalize the isolation tree for anomaly detection
anomaly_detector = IsolationForest(contamination= 0.2)

#push the residuals through the tree for anomaly detection
anomaly_detector.fit(residuals)

#make a prediction based on which points are anomalies
y_pred = anomaly_detector.predict(residuals)

print(y_pred)