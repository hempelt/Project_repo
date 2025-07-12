import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, TypeAdapter, ValidationError

# Load database connection parameters from .env file
load_dotenv()

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "table": os.getenv("DB_TABLE"),
}

# Create PostgreSQL connection string
connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Query the database and load data into a DataFrame
df = None
try:
    query = f"SELECT * FROM {DB_CONFIG['table']}"
    df = pd.read_sql_query(query, engine)
    print(f"✅ Data loaded from database successfully: {len(df)} rows")
except SQLAlchemyError as e:
    print("❌ Error occurs during data loading:", e)
finally:
    engine.dispose()

# Validate datatypes of important variables in dataframe with pydantic
# Define a Pydantic model for the measurement data
class Measurement(BaseModel):
    isoelectric_point: float
    protein_format: str
    molecular_weight_da: float
    composition: str
    product_conc_mg_ml: float
    tm_c: float

# Create a TypeAdapter for the Measurement model
adapter = TypeAdapter(list[Measurement])

# Check if the DataFrame is not None before validation
if df is not None:
    # Validate the DataFrame against the Pydantic model
    try:
        # Convert DataFrame to a list of dictionaries for validation
        validated = adapter.validate_python(df.to_dict(orient="records"))
        print("✅ No data type validation errors found.")
        # Export the validated DataFrame to a CSV file
        output_csv_path = 'data/raw/raw_data.csv'
        df.to_csv(output_csv_path, index=False) 
        print(f"✅ Data saved as CSV file to {output_csv_path}")
    except ValidationError as e:
        print("❌ Validation failed for one or more rows:")
        print("Error:", e)
        print("❌ Data was not exported due to validation errors.")
else:
    # If df is None, print an error message
    print("❌ No data to save, DataFrame is None.")
    