import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression       
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# im port the CSV file data\raw_data.csv
df = pd.read_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\raw\raw_data.csv')
df.head() #Display the first 5 rows of the DataFrame


#Create new feature ph
df['ph'] = df['composition'].str.split(',').str[-1].str.extract(r'(\d+)').astype(float)

# ----Create a numeric feature for the concentration of every excipient in the 'composition' -----------------------------------------------------

#Create new column Excipients
df['composition_without_ph'] = df['composition'].str.split(',').str[0]
df.drop('composition', axis=1, inplace=True)

# Create a list to hold unique excipients 
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
    return 0  # Wenn kein Wert gefunden wurde, None zurückgeben

# Create new columns for each unique excipient and extract their concentrations
for excipient in unique_excipients:
    new_column_name = excipient + '_conc' 
    # replace all uppercase letters with lowercase letters in the new column name
    new_column_name = new_column_name.lower()
    # Apply the extract_value function to the 'composition_without_ph' column for each excipient
    df[new_column_name] = df['composition_without_ph'].apply(lambda x: extract_value(x, excipient))
# Drop the 'composition_without_ph' column as it is no longer needed
df.drop('composition_without_ph', axis=1, inplace=True)


pd.set_option('display.max_colwidth', None)
df.head()  # Display the first 5 rows of the DataFrame with new columns for excipients

#export the DataFrame to a new CSV file and overwrite the existing one 
df.to_csv(r'C:\Users\hempe\Studium\Real_Project\Project_repo\data\raw\raw_data_all_features.csv', index=False)