import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression       
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# import the CSV file data\raw_data.csv
df = pd.read_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\raw\raw_data_all_features.csv')

#-----------------------------------------------------------------------------------
# Data Correction

# For a better overview data set is reduced to the most interessting variables we want to examine. Therefore id columns are dropped.
df = df.drop(columns=['product', 'formulation_title'])

# change data type in order to make pandas functions more efficient
df['protein_format'] = pd.Categorical(df['protein_format'])

# Eliminate spaces in all column names
df.columns = df.columns.str.replace(' ', '_')   
# Make sure column names are lower case a
df.columns = df.columns.str.lower()

#----------------------------------------------------------------------------------
# Error handling for missing values in column 'isoelectric_point' by using the mean value


# Count how many values are available
available_count = df['isoelectric_point'].notna().sum()
print(f"Available data points: {available_count} rows")

# Count missing values
missing_count = df['isoelectric_point'].isna().sum()
print(f"Missing values to be filled: {missing_count} rows")

# Fill missing values with the mean of the existing ones
if missing_count > 0:
    mean_value = df['isoelectric_point'].mean()
    df['isoelectric_point'] = df['isoelectric_point'].fillna(mean_value)
    print(f"Missing values were filled using the mean: {mean_value:.2f}")
else:
    print("No missing values – nothing to fill.")
#-------------------------------------------------------------------------------------------------- 
# Encoding categorical variables using one-hot encoding

# Create dummy variables for the 'protein_format' column
df = pd.get_dummies(df, columns=['protein_format'], drop_first=True)

# Identify only the dummy columns
dummy_cols = [col for col in df.columns if col.startswith('protein_format_')]

# Convert only those dummy columns to int type
df[dummy_cols] = df[dummy_cols].astype(int)

# Make sure column names are lower case a
df.columns = df.columns.str.lower()

#-----------------------------------------------------------------------------------
#export the DataFrame to a new CSV file and overwrite the existing one 
df.to_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\processed\preprocessed_data.csv', index=False)