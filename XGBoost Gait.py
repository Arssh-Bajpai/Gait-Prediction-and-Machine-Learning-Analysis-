import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# Read the CSV file for the patient
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
    return stride_step_array


# Initialize the patient from the CSV file
patient = readCSVFile('/Users/arsshbajpai/PycharmProjects/Final Project AI/Gait Data/pd1-si.csv')


# Feature Engineering: Creating additional features
def feature_engineering(data):
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1] - 1)] + ["target"])

    # Statistical features
    df['feature_mean'] = df.mean(axis=1)
    df['feature_std'] = df.std(axis=1)
    df['feature_min'] = df.min(axis=1)
    df['feature_max'] = df.max(axis=1)

    # Moving averages
    df['moving_avg_3'] = df.iloc[:, :-1].T.rolling(window=3).mean().T.mean(axis=1)
    df['moving_avg_5'] = df.iloc[:, :-1].T.rolling(window=5).mean().T.mean(axis=1)

    # Peak values
    df['feature_peak'] = df.iloc[:, :-1].apply(np.max, axis=1)

    # Frequency domain features using FFT
    def compute_fft(row):
        fft_vals = np.fft.fft(row)
        fft_power = np.abs(fft_vals) ** 2
        return fft_power[:len(fft_power) // 2]  # Return only the positive frequencies

    fft_features = df.iloc[:, :-1].apply(compute_fft, axis=1)
    fft_features_df = pd.DataFrame(fft_features.tolist(), index=df.index)
    fft_features_df.columns = [f"fft_{i}" for i in range(fft_features_df.shape[1])]
    df = pd.concat([df, fft_features_df], axis=1)

    return df


# Apply feature engineering
patient_df = feature_engineering(patient)
X = patient_df.drop(columns=['target']).values
Y = patient_df['target'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the arrays into testing and training with a 20-80 split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialize the regressor function using squared error and root mean square error
trainer = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

# Use cross-validation to evaluate the model
cv_scores = cross_val_score(trainer, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {-cv_scores}")
print(f"Mean Cross-Validation MSE: {-cv_scores.mean()}")

# Train the model using the training data
trainer.fit(X_train, Y_train)

# Make predictions off of the testing data
y_pred = trainer.predict(X_test)

# Calculate the residuals: the difference between the actual and predictions for the data
residual = Y_test - y_pred

# Reshape the residuals into a 2D array
residuals = residual.reshape(-1, 1)

# Initialize the isolation forest for anomaly detection
anomaly_detector = IsolationForest(contamination=0.2, random_state=42)

# Fit the residuals into the isolation forest
anomaly_detector.fit(residuals)

# Predict anomalies based on the residuals
anomalies = anomaly_detector.predict(residuals)

# Post-process the results to map -1 to True (anomaly) and 1 to False (normal)
anomalies = anomalies == -1

# Print the anomalies
print("Anomalies detected:", anomalies)

# Additional Evaluation Metrics
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Feature Importance
importance = trainer.feature_importances_
feature_names = patient_df.drop(columns=['target']).columns.astype(
    str)  # Ensure feature names are strings

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_names, importance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Optional: Tune hyperparameters using GridSearchCV (for demonstration purposes, not comprehensive)
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                           param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)

# Print the best parameters and the best score from the grid search
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
# Feature Importance Plot
plt.figure(figsize=(12, 8))
plt.barh(feature_names, importance, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualization of the Residuals and Anomalies
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='blue', label='Residuals', alpha=0.7)
plt.scatter(np.where(anomalies)[0], residuals[anomalies], color='red', label='Anomalies', marker='x')
plt.xlabel('Index')
plt.ylabel('Residual')
plt.title('Residuals and Anomalies')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualization of Predicted vs. Actual Values
plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Actual', color='blue', linestyle='-', marker='o', markersize=5)
plt.plot(y_pred, label='Predicted', color='red', linestyle='-', marker='x', markersize=5)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



# Diagnostic Report
def diagnostic_report(y_test, y_pred, anomalies):
    report = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': y_test - y_pred,
        'Anomaly': anomalies
    })
    report['Diagnosis'] = np.where(report['Residual'] > np.percentile(report['Residual'], 75),
                                   'Severe Impairment',
                                   np.where(
                                       report['Residual'] > np.percentile(report['Residual'], 50),
                                       'Mild Impairment', 'Normal'))
    return report


diagnosis_report = diagnostic_report(Y_test, y_pred, anomalies)
print(diagnosis_report)

# Save the model and scaler for future use
joblib.dump(trainer, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save preprocessed data
preprocessed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'Y_train': Y_train,
    'Y_test': Y_test
}
joblib.dump(preprocessed_data, 'preprocessed_data.pkl')
