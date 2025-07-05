import pandas as pd
import re     
from sklearn.model_selection import train_test_split

# import the CSV file data/raw_data.csv
df = pd.read_csv('data/raw/raw_data.csv')

#Extract pH value from column 'coposition' and create new feature 'ph'
df['ph'] = df['composition'].str.split(',').str[-1].str.extract(r'(\d+\.?\d*)').astype(float)

# ----Create a numeric feature for the concentration of every excipient in the 'composition' -----------------------------------------------------

#Create new column with all excipients and without pH value called 'composition_without_ph'
df['composition_without_ph'] = df['composition'].str.split(',').str[0]
df.drop('composition', axis=1, inplace=True)

# Create a list to hold all excipients 
excipients = []

# For every value in the column 'composition_without_ph', split the string by '+' and strip whitespace and put them into a list
for composition_without_ph in df['composition_without_ph']:
    list_conc_excipient = [c.strip() for c in composition_without_ph.split('+')] 
    # For every value in the list, split the string by ' ' and take the last part as the excipient name
    for conc_excipient in list_conc_excipient:
        parts = conc_excipient.split()
        if parts:  # Ignore empty strings
            excipients.append(parts[-1])

unique_excipients = list(set(excipients))  # Get unique excipients

# Function to extract the concentration of a specific excipient from a string
def extract_value(excipient_str,excipient):
    # Suche nach dem KCl-Teil in der Zeichenkette
    list_conc_excipient = [c.strip() for c in excipient_str.split('+')]
    for conc_excipient in list_conc_excipient:
        if excipient in conc_excipient:
            # Extrahiere den numerischen Wert
            match = re.search(r'(\d+(\.\d+)?)', conc_excipient)
            if match:
                return float(match.group(1))  
    return 0  # If now match is found, return 0

# Create new columns for each unique excipient and extract their concentrations
for excipient in unique_excipients:
    new_column_name = excipient + '_conc' 
    # Apply the extract_value function to the 'composition_without_ph' column for each excipient
    df[new_column_name] = df['composition_without_ph'].apply(lambda x: extract_value(x, excipient))
# Drop the 'composition_without_ph' column as it is no longer needed
df.drop('composition_without_ph', axis=1, inplace=True)

#-----------------------------------------------------------------------------------
# Data Correction

# For a better overview data set is reduced to the most interessting variables we want to examine. Therefore id columns are dropped.
df = df.drop(columns=['product', 'formulation_title'])

# change data type in order to make pandas functions more efficient
df['protein_format'] = pd.Categorical(df['protein_format'])

# Eliminate spaces and in all column names
df.columns = df.columns.str.replace(' ', '_')   
# Make sure column names are lower case a
df.columns = df.columns.str.lower()
# Remove special characters from column names
df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True) 

#Define input (X) and target (y)
X = df.drop(columns=['tm_c'])  # Drop the target variable 'tm_c' from the features
y = df['tm_c']   

#-----------------------------------------------------------------------------------
# For the followeing preporessing steps the data set is split into a training and test set.
# Only the training set is used for preprocessing, while the test set remains untouched until the final evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#----------------------------------------------------------------------------------

# Merge features + Target in one DataFrame
train_df = X_train.copy()
train_df['tm_c'] = y_train

test_df = X_test.copy()
test_df['tm_c'] = y_test

# Save train und test data as seperate CSV files
train_df.to_csv('data/raw/all_features_train_data.csv', index=False)
test_df.to_csv('data/raw/all_features_test_data.csv', index=False)
