import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from mlflow.models.signature import infer_signature

# import the processed training data
df = pd.read_csv('data/processed/df_train_processed.csv')


# Define features (X) and target (y)
X = df.drop(columns=['tm_c'])
y = df['tm_c']   

# Set MLflow tracking URI and experiment
# Make sure to change the path to your local MLflow tracking server
mlflow.set_tracking_uri("file:///C:/Users/hempe/Studium/Real_Project/Project_repo/mlruns")
mlflow.set_experiment("tm_prediction_experiment")

# Start MLflow Run
with mlflow.start_run() as run:
    # Train gradient boosting regressor model with optimized hyperparamter
    model = GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.8,
        random_state=42
    )
    model.fit(X, y)

 # Log parameters
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("random_state", 42)

    # Log the model with input/output schema
    signature = infer_signature(X, model.predict(X))
    input_example = X.head(5)
    mlflow.sklearn.log_model(
        model,
        "gradient_boosting_model",
        signature=signature,
        input_example=input_example
    )

 # Save the Run ID to a latest_run.txt
    with open("latest_run.txt", "w") as f:
        f.write(run.info.run_id)

    print("✅ Model successfully logged!")
    print("Run ID:", run.info.run_id)
    print("Run ID saved to latest_run.txt")

