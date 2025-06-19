import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# import the CSV file data\Coded_FS_Data.csv
df = pd.read_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\processed\df_train_processed.csv')
df.head() #Display the first 5 rows of the DataFrame

#Define input (X) and target (y)
X = df.drop(columns=['tm_c'])  # Drop the target variable 'tm_c' from the features
y = df['tm_c']   

# --- Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modell trainieren
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Modell speichern
joblib.dump(model, 'C:/Users/hempe/Studium/Real_Project/Project_repo/models/gradient_boosting_model.pkl')

# Optional: auch den Scaler speichern
joblib.dump(scaler, 'C:/Users/hempe/Studium/Real_Project/Project_repo/models/modelsscaler.pkl')