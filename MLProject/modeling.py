import argparse
import pandas as pd
import joblib
import mlflow
import numpy as np
import time
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)

# Import custom preprocessor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# ---------------------------
# MLflow remote tracking
# ---------------------------
mlflow.set_tracking_uri("https://dagshub.com/Sintasitinuriah/my-first-repo.mlflow")

# Optional: Uncomment if running manually without secrets/env setup
# os.environ["MLFLOW_TRACKING_USERNAME"] = "Sintasitinuriah"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_token_here"

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv(args.data_path)
X = data.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = np.log(data['Item_Outlet_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# ---------------------------
# Preprocessing & Pipeline
# ---------------------------
preprocessor = SklearnPreprocessor(
    num_columns=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year'],
    ordinal_columns=['Item_Fat_Content', 'Outlet_Size'],
    nominal_columns=['Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Identifier'],
    degree=2
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# ---------------------------
# Training with MLflow
# ---------------------------
with mlflow.start_run() as run:
    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()

    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    training_time = end - start

    # Log params & metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("degree_poly", 2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Explained_Variance", explained_var)
    mlflow.log_metric("Max_Error", max_err)
    mlflow.log_metric("Training_Time", training_time)

    # Log model
    mlflow.sklearn.log_model(pipeline, "model")

    # ✅ Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")

    # Print ringkasan
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}, Max Error: {max_err:.4f}, Training Time: {training_time:.2f}s")

# ---------------------------
# Save local model copy (optional)
# ---------------------------
output_path = "./models/linear_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Model saved to: {output_path}")
