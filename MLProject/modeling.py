import argparse
import pandas as pd
import joblib
import mlflow
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)

# Custom preprocessor import
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# 🚫 JANGAN pakai mlflow.set_experiment() kalau pakai `mlflow run`
# mlflow.set_experiment("Big Mart Sales Prediction")  <-- HAPUS

# Load dataset
data = pd.read_csv(args.data_path)

X = data.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = np.log(data['Item_Outlet_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Preprocessing pipeline
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

# Gunakan run aktif dari MLflow CLI
with mlflow.start_run():
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

    # Log
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("degree_poly", 2)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Explained_Variance", explained_var)
    mlflow.log_metric("Max_Error", max_err)
    mlflow.log_metric("Training_Time", training_time)

    mlflow.sklearn.log_model(pipeline, "model")

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}, Max Error: {max_err:.4f}, Training Time: {training_time:.2f}s")

# Save model to outputs folder for Git LFS
output_path = "../models/linear_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Model saved to: {output_path}")