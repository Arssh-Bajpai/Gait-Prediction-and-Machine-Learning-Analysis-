import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network Definition
class GaitAnomalyNet(nn.Module):
    def __init__(self, input_size):
        super(GaitAnomalyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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

# Function to run the model and generate the graph and report
def run_model(file_name):
    try:
        patient = readCSVFile(file_name)
        if patient.size == 0:
            print("Error: The data array is empty.")
            return

        patient_df = feature_engineering(patient)
        if patient_df.shape[0] == 0 or patient_df.shape[1] == 0:
            print("Error: The DataFrame is empty after feature engineering.")
            return

        X = patient_df.drop(columns=['target']).values
        Y = patient_df['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        # XGBoost Model
        trainer = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        cv_scores = cross_val_score(trainer, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"Cross-Validation MSE Scores: {-cv_scores}")
        print(f"Mean Cross-Validation MSE: {-cv_scores.mean()}")
        trainer.fit(X_train, Y_train)

        y_pred_train_xgb = trainer.predict(X_train)
        y_pred_test_xgb = trainer.predict(X_test)

        residual_train_xgb = Y_train - y_pred_train_xgb
        residual_test_xgb = Y_test - y_pred_test_xgb

        # Neural Network for Time-Sensitive Anomaly Detection
        input_size = X_train.shape[1]
        net = GaitAnomalyNet(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

        # Train the Neural Network
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = net(X_train_tensor)
            loss = criterion(outputs, Y_train_tensor)
            loss.backward()
            optimizer.step()

        y_pred_train_nn = net(X_train_tensor).detach().numpy().flatten()
        y_pred_test_nn = net(X_test_tensor).detach().numpy().flatten()

        residual_train_nn = Y_train - y_pred_train_nn
        residual_test_nn = Y_test - y_pred_test_nn

        # Combine XGBoost and Neural Network residuals
        combined_residuals_train = residual_train_xgb + residual_train_nn
        combined_residuals_test = residual_test_xgb + residual_test_nn

        anomaly_detector = IsolationForest(contamination=0.2, random_state=42)
        anomaly_detector.fit(combined_residuals_train.reshape(-1, 1))

        anomalies_train = anomaly_detector.predict(combined_residuals_train.reshape(-1, 1)) == -1
        anomalies_test = anomaly_detector.predict(combined_residuals_test.reshape(-1, 1)) == -1

        diagnosis_report_train = diagnostic_report(Y_train, y_pred_train_xgb, anomalies_train)
        diagnosis_report_test = diagnostic_report(Y_test, y_pred_test_xgb, anomalies_test)

        # Display Diagnostic Report
        text_report.delete(1.0, tk.END)
        text_report.insert(tk.END, "\nTraining Data Diagnostic Report:\n")
        text_report.insert(tk.END, diagnosis_report_train.to_string())
        text_report.insert(tk.END, "\n\nTesting Data Diagnostic Report:\n")
        text_report.insert(tk.END, diagnosis_report_test.to_string())

        # Plotting the graph for testing data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(combined_residuals_test)), combined_residuals_test, color='blue', label='Residuals (Test)', alpha=0.7)
        ax.scatter(np.where(anomalies_test)[0], combined_residuals_test[anomalies_test], color='red', label='Anomalies (Test)',
                   marker='x')
        ax.set_xlabel('Index')
        ax.set_ylabel('Residual')
        ax.set_title(f'Residuals and Anomalies (Testing Data) - {file_name}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Display the plot in the GUI
        for widget in frame_graph.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        print(f"An error occurred in run_model: {e}")

# Create the GUI
root = tk.Tk()
root.title("Gait Anomaly Detection")

# Frame for dropdown and run button
frame_top = ttk.Frame(root)
frame_top.pack(pady=10)

# Dropdown for selecting the file
label = ttk.Label(frame_top, text="Select Patient File:")
label.pack(side=tk.LEFT)

gait_data_dir = '/Users/arsshbajpai/PycharmProjects/Final Project AI/Gait Data'
files = [f for f in os.listdir(gait_data_dir) if f.endswith('.csv')]

selected_file = tk.StringVar()
dropdown = ttk.Combobox(frame_top, textvariable=selected_file, values=files)
dropdown.pack(side=tk.LEFT, padx=10)
dropdown.current(0)

# Run button
button_run = ttk.Button(frame_top, text="Run Model", command=lambda: run_model(os.path.join(gait_data_dir, selected_file.get())))
button_run.pack(side=tk.LEFT)

# Frame for graph
frame_graph = ttk.Frame(root)
frame_graph.pack(pady=10)

# Frame for diagnostic report
frame_report = ttk.Frame(root)
frame_report.pack(pady=10)

text_report = tk.Text(frame_report, height=15, width=80)
text_report.pack()

root.mainloop()
