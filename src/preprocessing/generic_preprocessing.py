import pandas as pd      
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# import the train data with all features
df_train = pd.read_csv('data/raw/all_features_train_data.csv')

# Write all numeric features in a list
numeric_features = df_train.select_dtypes(include=['number']).columns.tolist()
# Write all categorical features in a list
categorical_features = df_train.select_dtypes(include=['object','category']).columns.tolist()

# Define the imputer for numeric features
num_imputer = SimpleImputer(strategy='mean')

# Define the imputer for categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')

train_numeric = pd.DataFrame(num_imputer.fit_transform(df_train[numeric_features]),
                             columns=numeric_features,
                             index=df_train.index)

train_cat = pd.DataFrame(cat_imputer.fit_transform(df_train[categorical_features]),
                                 columns=categorical_features,
                                 index=df_train.index)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_cat_encoded = pd.DataFrame(ohe.fit_transform(train_cat),
                                 columns=ohe.get_feature_names_out(categorical_features),
                                 index=df_train.index)

# Make sure new column names are lower case
train_cat_encoded.columns = train_cat_encoded.columns.str.lower()

# Combine the processed numeric and categorical features into a single DataFrame
df_train_processed = pd.concat([train_numeric, train_cat_encoded], axis=1)

# Export the DataFrame to a new CSV file and overwrite the existing one 
df_train_processed.to_csv('data/processed/df_train_processed.csv', index=False)

# Print success message and display the shape of the preprocessed Train data
print(f'✅ Train data were preprocessed successfully and saved in data/processed. Size: {df_train_processed.shape}') 

#------------------------------------------------------------------------------------
# Import the test data with all features
df_test = pd.read_csv('data/raw/all_features_test_data.csv')

# Apply the same preprocessing steps to the test data
test_numeric = pd.DataFrame(num_imputer.transform(df_test[numeric_features]),
                            columns=numeric_features,
                            index=df_test.index)
test_cat = pd.DataFrame(cat_imputer.transform(df_test[categorical_features]),
                        columns=categorical_features,
                        index=df_test.index)    
test_cat_encoded = pd.DataFrame(ohe.transform(test_cat),
                                 columns=ohe.get_feature_names_out(categorical_features),
                                 index=df_test.index)
# Make sure new column names are lower case
test_cat_encoded.columns = test_cat_encoded.columns.str.lower()

# Combine the processed numeric and categorical features into a single DataFrame
df_test_processed = pd.concat([test_numeric, test_cat_encoded], axis=1) 

# Export the DataFrame to a new CSV file and overwrite the existing one
df_test_processed.to_csv('data/processed/df_test_processed.csv', index=False)

 # Print success message and display the shape of the preprocessed Test data
print(f'✅ Test data were preprocessed successfully and saved in data/processed. Size: {df_test_processed.shape}')


