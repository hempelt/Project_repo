import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# import the CSV file data\Coded_FS_Data.csv
df = pd.read_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\processed\df_train_processed.csv')


#Define input (X) and target (y)
X = df.drop(columns=['tm_c'])  # Drop the target variable 'tm_c' from the features
y = df['tm_c']   

# Train gradient boosting regressor model with optimized hyperparamter
model = GradientBoostingRegressor(learning_rate= 0.1, max_depth= 4, n_estimators= 200, subsample= 0.8, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'C:/Users/hempe/Studium/Real_Project/Project_repo/models/gradient_boosting_model.pkl')
