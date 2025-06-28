import pandas as pd
from sklearn.model_selection import cross_validate
import mlflow.sklearn

mlflow.set_experiment("tm_prediction_experiment")

# Read the latest run ID from the file
with open("latest_run.txt", "r") as f:
    run_id = f.read().strip()

# Load model from  MLflow via Run-ID
model_uri = f"runs:/{run_id}/gradient_boosting_model"
model = mlflow.sklearn.load_model(model_uri)

# import the CSV file data\Coded_FS_Data.csv
df_test = pd.read_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\processed\df_test_processed.csv')

#Define input (X) and target (y)
X = df_test.drop(columns=['tm_c'])  # Drop the target variable 'tm_c' from the features
y = df_test['tm_c']   

# Perform 5-fold cross-validation
cv = cross_validate(model, X, y, cv=5, scoring=('r2', 'neg_root_mean_squared_error'))
 
# Convert scores to positive RMSE
rmse_scores = -cv['test_neg_root_mean_squared_error']
r2_scores = cv['test_r2']

# Cross-Validation Ergebnisse in DataFrame speichern
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'RMSE': rmse_scores,
    'R2': r2_scores
})

# Save CSV locally
csv_path = r'C:/Users/hempe/Studium/Real_Project/Project_repo/models/validation_results.csv'
cv_results.to_csv(csv_path, index=False)

# Log validation metrics in MLflow
with mlflow.start_run(nested=True) as run:  # nested=True um einen Unterlauf zu starten
    mlflow.log_metric("validation_rmse_mean", rmse_scores.mean())
    mlflow.log_metric("validation_rmse_std", rmse_scores.std())
    mlflow.log_metric("validation_r2_mean", r2_scores.mean())
    mlflow.log_metric("validation_r2_std", r2_scores.std())


print("Validation results are successcully logged!")



