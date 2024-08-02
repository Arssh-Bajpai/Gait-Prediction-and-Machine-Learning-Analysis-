import xgboost as xgb
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import pandas as pd
import os

# Define the directory containing the CSV files
csv_directory = './csv_files/'  # Update this to your directory

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Read each CSV file into a DataFrame, add a column for the file name (without .csv extension), and store them in a list
dataframes = []
for file in csv_files:
    file_path = os.path.join(csv_directory, file)
    df = pd.read_csv(file_path)
    file_name_without_ext = os.path.splitext(file)[0]  # Remove the .csv extension
    df['File_Name'] = file_name_without_ext  # Add the file name without extension as a new column
    df['Has_PD'] = "Pt" in file_name_without_ext
    dataframes.append(df)

# Concatenate all DataFrames into one big DataFrame
big_df = pd.concat(dataframes, ignore_index=True)

print(f"Combined DataFrame shape: {big_df.shape}")
print(big_df)
