
## Gait Anomaly Detection

### Overview
The **Gait Anomaly Detection** project uses **machine learning** and **neural networks** to analyze gait patterns in **Huntington’s and Parkinson’s disease patients**. The model is trained on **normative walking data** and detects deviations indicative of neurological disorders.

### Features
- **Real-time gait anomaly detection** using XGBoost and Isolation Forest.
- **Pre-trained neural network integration** for model optimization.
- **Automated data cleaning and preprocessing** using NumPy and Pandas.
- **Tested against diagnosed patient gait data** for validation.

### Technologies Used
- **Python** (Core programming language)
- **PyTorch** (Deep learning framework)
- **XGBoost** (Gradient boosting for anomaly detection)
- **NumPy, Pandas** (Data preprocessing)

### Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Arssh-Bajpai/Gait-Prediction-and-Machine-Learning-Analysis-
   cd Gait-Prediction-and-Machine-Learning-Analysis-
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas torch xgboost scikit-learn
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Run the anomaly detection:
   ```bash
   python detect.py --input gait_data.csv
   ```

### Future Improvements
- Expanding dataset to include more diverse gait patterns.
- Deploying as a **web application** for clinical use.

