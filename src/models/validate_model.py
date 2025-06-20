import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

# Modell und Scaler laden
model = joblib.load('C:/Users/hempe/Studium/Real_Project/Project_repo/models/gradient_boosting_model.pkl')

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

# Print mean and std
print(f"Average RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
print(f"Average R²: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")

# Create DataFrame for plotting
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'RMSE': rmse_scores,
    'R2': r2_scores
})


# CSV speichern
cv_results.to_csv(r'C:/Users/hempe/Studium/Real_Project/Project_repo/models/validation_results.csv', index=False)


