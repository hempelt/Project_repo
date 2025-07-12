import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import mlflow.sklearn

# Set the MLflow tracking URI and experiment
mlflow.set_experiment("tm_prediction_experiment")

# Read the latest run ID from the file
with open("latest_run.txt", "r") as f:
    run_id = f.read().strip()

# Load model from  MLflow via Run-ID
model_uri = f"runs:/{run_id}/gradient_boosting_model"
model = mlflow.sklearn.load_model(model_uri)

# Import the processed test data
df_test = pd.read_csv('data/processed/df_test_processed.csv')

# Define features (X) and target (y)
X = df_test.drop(columns=['tm_c'])  # Drop the target variable 'tm_c' from the features
y = df_test['tm_c']   

# Make predictions using the loaded model
y_pred = model.predict(X)

# Calculate RMSE and R²
rmse = root_mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Log validation results to MLflow
with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print(f"✅ RMSE: {rmse:.3f}")
    print(f"✅ R²:   {r2:.3f}")
    print(f"📌 Validation results are logged as metrics within the previous run: {run_id}")


print("Validation results are successcully logged!")



