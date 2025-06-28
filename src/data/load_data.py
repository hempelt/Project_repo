import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

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

#Query the database and load data into a DataFrame
df = None
try:
    query = f"SELECT * FROM {DB_CONFIG['table']}"
    df = pd.read_sql_query(query, engine)
    print(f"✅ Data loaded from database successfully: {len(df)} rows")
except SQLAlchemyError as e:
    print("❌ Error occurs during data loading:", e)
finally:
    engine.dispose()

# Export DataFrame to CSV
if df is not None:
    output_csv_path = 'data/raw/raw_data.csv'
    df.to_csv(output_csv_path, index=False) 
    print(f"✅ Data saved as CSV file to {output_csv_path}")
else:
    print("❌ No data to save, DataFrame is None.")